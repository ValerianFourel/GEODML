"""Per-job status + ETA for the probing fan-out.

Usage (on JUPITER):
    python scripts/probing_status.py

Parses logs/geodml-prob-{JID}.out and .err for every running probing job
and prints:

  JID  ELAPSED  MODEL  TREATMENT  FRAME  STEP  PROGRESS  ETA-TO-END

ETA is computed by combining the tqdm-reported "remaining" of the current
step with the typical durations of the steps that still have to happen
before the job exits. Numbers are calibrated from the runs we have in
hand on 2026-05-25; they're approximate but close.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, field

LOGS = Path("logs")
USER = os.environ.get("USER", "fourel1")

# ---------------------------------------------------------------------------
# Typical step durations (seconds), calibrated from completed jobs.
# Used to project end-of-job ETA from the current step.
# ---------------------------------------------------------------------------
TYP = {
    "load":            3 * 60,
    "digest_full":     4 * 60,
    "hidden_full":    55 * 60,
    "digest_rw":     1.5 * 60,
    "hidden_rw":      25 * 60,
    "probe_training":  5 * 60,  # logistic-regression fit per layer × treatment, after hidden-states
    "csv_write":      0.5 * 60,
}

# ---------------------------------------------------------------------------
@dataclass
class JobInfo:
    jid: str
    elapsed: str = "?"
    state: str = "?"
    model: str = "?"
    treatment: str = "?"
    frame: str = "?"
    step: str = "?"
    pct: int | None = None
    x_y: str = "?"
    tqdm_elapsed: str = "?"
    tqdm_remaining: str = "?"
    eta_text: str = "?"
    raw_last: str = ""


def parse_hms_to_seconds(s: str) -> int | None:
    """'15:52' -> 952, '1:25:30' -> 5130."""
    if not s or s in {"?", "00:00"}:
        return None
    try:
        parts = list(map(int, s.split(":")))
    except ValueError:
        return None
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


def fmt_minutes(secs: float) -> str:
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs/60:.0f}m"
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    return f"{h}h{m:02d}m"


# ---------------------------------------------------------------------------
def get_running_jobs() -> list[tuple[str, str, str]]:
    try:
        out = subprocess.check_output(
            ["squeue", "-u", USER, "-h", "-n", "geodml-prob",
             "-o", "%i|%M|%T"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) >= 3:
            rows.append((parts[0], parts[1], parts[2]))
    return rows


# ---------------------------------------------------------------------------
TQDM_RE = re.compile(
    r"""
    ^                                       # line start
    (?P<step>[A-Za-z_>][A-Za-z_> \[\]0-9\-]*?)   # step name (greedy chars excluding ':')
    :\s+                                    # colon + space
    (?P<pct>\d+)%                           # percent
    \|[^\|]*\|                              # bar between two pipes (anything not-pipe inside)
    \s*(?P<x>\d+)/(?P<y>\d+)\s*             # X/Y counts
    \[(?P<el>[^<\]]+?)<(?P<rem>[^,\]]+?)    # elapsed<remaining
    """,
    re.MULTILINE | re.VERBOSE,
)


def parse_logs(jid: str) -> JobInfo:
    info = JobInfo(jid=jid)
    out_path = LOGS / f"geodml-prob-{jid}.out"
    err_path = LOGS / f"geodml-prob-{jid}.err"

    # --- .out: model, treatment, frame markers --------------------------
    if out_path.exists():
        out_text = out_path.read_text(errors="replace")
        m = re.search(r"MODEL=(\S+)", out_text)
        if m:
            info.model = m.group(1).split("/")[-1].replace("-Instruct", "")
        m = re.search(r"treatment=(\S+)", out_text)
        if m:
            info.treatment = m.group(1)
        # most recent frame marker
        frames = re.findall(r"=== frame=([a-z_]+)", out_text)
        if frames:
            info.frame = frames[-1]

    # --- .err: last tqdm line ------------------------------------------
    if err_path.exists():
        # only read tail (last 200 KB) for speed
        try:
            with err_path.open("rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 200_000))
                tail = f.read().decode(errors="replace")
        except Exception:
            tail = ""
        matches = list(TQDM_RE.finditer(tail))
        if matches:
            mm = matches[-1]
            info.step = mm.group("step").strip()
            info.pct = int(mm.group("pct"))
            info.x_y = f"{mm.group('x')}/{mm.group('y')}"
            info.tqdm_elapsed = mm.group("el").strip()
            info.tqdm_remaining = mm.group("rem").strip()
            info.raw_last = mm.group(0)

    return info


# ---------------------------------------------------------------------------
def estimate_eta(info: JobInfo) -> str:
    """Project remaining time until job exits."""
    if not info.step or info.step == "?":
        # Model still loading, no tqdm yet → conservative
        return f"~{fmt_minutes(TYP['load'] + TYP['digest_full'] + TYP['hidden_full'] + TYP['digest_rw'] + TYP['hidden_rw'])} (no tqdm yet)"

    step = info.step.lower()
    frame = info.frame

    # remaining seconds of the CURRENT step from tqdm
    rem_now = parse_hms_to_seconds(info.tqdm_remaining) or 0

    # phase order: load → digest_full → hidden_full → digest_rw → hidden_rw → write
    if "loading weights" in step:
        return f"~{fmt_minutes(rem_now + TYP['digest_full'] + TYP['hidden_full'] + TYP['digest_rw'] + TYP['hidden_rw'])}"
    if "html->digest" in step:
        if frame == "full":
            return f"~{fmt_minutes(rem_now + TYP['hidden_full'] + TYP['digest_rw'] + TYP['hidden_rw'])}"
        else:  # robust_winners
            return f"~{fmt_minutes(rem_now + TYP['hidden_rw'])}"
    if "hidden-states" in step:
        if frame == "full":
            return f"~{fmt_minutes(rem_now + TYP['digest_rw'] + TYP['hidden_rw'])}"
        else:  # robust_winners — this is the last big phase
            return f"~{fmt_minutes(rem_now + TYP['probe_training'] + TYP['csv_write'])}"
    if "t7-keywords" in step:
        return f"~{fmt_minutes(rem_now + 5*60)}"
    return f"~{fmt_minutes(rem_now)}"


# ---------------------------------------------------------------------------
def main():
    rows = get_running_jobs()
    if not rows:
        print("No running geodml-prob jobs found.")
        return

    # header
    hdr = ("JID", "ELAPSED", "MODEL", "TREATMENT", "FRAME", "STEP", "PROGRESS", "ETA-TO-END")
    fmt = "{:<8} {:<9} {:<14} {:<25} {:<14} {:<22} {:<14} {}"
    print(fmt.format(*hdr))
    print("-" * 130)

    total_max_eta = 0
    for jid, elapsed, state in rows:
        info = parse_logs(jid)
        info.elapsed = elapsed
        info.state = state
        eta = estimate_eta(info)
        info.eta_text = eta
        prog = f"{info.pct}% {info.x_y}" if info.pct is not None else "(no tqdm)"
        print(fmt.format(
            jid, elapsed, info.model, info.treatment, info.frame,
            info.step[:22], prog, eta,
        ))

        # parse the eta string for "X min" / "Xh YYm" so we can find the max
        m = re.search(r"~(\d+)h(\d+)m", eta)
        if m:
            total_max_eta = max(total_max_eta, int(m.group(1))*3600 + int(m.group(2))*60)
        else:
            m = re.search(r"~(\d+)m", eta)
            if m:
                total_max_eta = max(total_max_eta, int(m.group(1))*60)

    print()
    print(f"Slowest job ETA: ~{fmt_minutes(total_max_eta)} until all 8 are done.")
    print()
    print("Step durations used for ETA (calibrated 2026-05-25):")
    for k, v in TYP.items():
        print(f"  {k:<18} {v/60:.1f} min")


if __name__ == "__main__":
    main()
