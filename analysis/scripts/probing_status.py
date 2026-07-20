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
    frame: str = "?"            # currently active frame inferred from === markers
    frame_arg: str = "both"     # what was passed via FRAME= (full / robust_winners / both)
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
# tqdm overwrites with \r, so we don't anchor to ^. Just find any tqdm-shaped
# fragment in the tail and take the last one.
TQDM_RE = re.compile(
    r"""
    (?P<step>[A-Za-z][A-Za-z_> \[\]0-9\-]{1,30}?)   # step name (1-30 chars)
    :\s+                                            # colon + space
    (?P<pct>\d+)%                                   # percent
    \|[^\|]*\|                                      # bar between two pipes
    \s*(?P<x>\d+)/(?P<y>\d+)\s*                     # X/Y counts
    \[(?P<el>[^<\]]+?)<(?P<rem>[^,\]]+?)[,\]]       # elapsed<remaining ,
    """,
    re.VERBOSE,
)


def parse_logs(jid: str) -> JobInfo:
    info = JobInfo(jid=jid)
    out_path = LOGS / f"geodml-prob-{jid}.out"
    err_path = LOGS / f"geodml-prob-{jid}.err"

    # --- .out: model, treatment, frame markers --------------------------
    if out_path.exists():
        out_text = out_path.read_text(errors="replace")
        # .out has lines like "[probing] model=meta-llama/Llama-3.3-70B-Instruct frame=both ..."
        m = re.search(r"\bmodel=(\S+)", out_text, re.IGNORECASE)
        if m:
            info.model = m.group(1).split("/")[-1].replace("-Instruct", "")
        m = re.search(r"\btreatment=(\S+)", out_text, re.IGNORECASE)
        if m and m.group(1) not in ("ALL", "?"):
            info.treatment = m.group(1)
        # frame=both/full/robust_winners — what the script was *invoked* with
        m = re.search(r"\bframe=([a-z_]+)", out_text)
        if m and m.group(1) in {"full", "robust_winners", "both"}:
            info.frame_arg = m.group(1)
        # most recent frame marker showing which frame the run is *currently* in
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
    """Project remaining wall-time until the slurm job exits.

    Accounts for what FRAME= was passed: with FRAME=full the job skips
    the robust_winners pass and exits ~25-30 min earlier than the both-
    frame default.
    """
    step = (info.step or "").lower()
    frame_arg = info.frame_arg          # what FRAME= was set to
    frame_now = info.frame              # what frame the script is currently in

    # Will this job do a robust_winners pass AFTER the current/full pass?
    will_do_rw = (frame_arg == "both") and (frame_now != "robust_winners")
    tail_rw = (TYP["digest_rw"] + TYP["hidden_rw"]) if will_do_rw else 0
    tail_final = TYP["probe_training"] + TYP["csv_write"]
    rem_now = parse_hms_to_seconds(info.tqdm_remaining) or 0

    if not step or step == "?":
        # No tqdm yet — assume still loading; conservative
        full_part = (TYP["digest_full"] + TYP["hidden_full"]) if frame_arg != "robust_winners" else 0
        return f"~{fmt_minutes(TYP['load'] + full_part + tail_rw + tail_final)} (no tqdm yet)"

    if "loading weights" in step:
        full_part = (TYP["digest_full"] + TYP["hidden_full"]) if frame_arg != "robust_winners" else 0
        return f"~{fmt_minutes(rem_now + full_part + tail_rw + tail_final)}"
    if "html->digest" in step:
        if frame_now == "robust_winners":
            return f"~{fmt_minutes(rem_now + TYP['hidden_rw'] + tail_final)}"
        # currently doing the full-frame digest
        return f"~{fmt_minutes(rem_now + TYP['hidden_full'] + tail_rw + tail_final)}"
    if "hidden-states" in step:
        if frame_now == "robust_winners":
            return f"~{fmt_minutes(rem_now + tail_final)}"
        # currently doing the full-frame hidden states
        return f"~{fmt_minutes(rem_now + tail_rw + tail_final)}"
    if "t7-keywords" in step:
        return f"~{fmt_minutes(rem_now + TYP['probe_training'] + TYP['csv_write'])}"
    return f"~{fmt_minutes(rem_now + tail_final)}"


# ---------------------------------------------------------------------------
def debug_one(jid: str):
    err_path = LOGS / f"geodml-prob-{jid}.err"
    out_path = LOGS / f"geodml-prob-{jid}.out"
    print(f"=== {jid}.out (tail) ===")
    if out_path.exists():
        print(out_path.read_text(errors="replace")[-2000:])
    print(f"\n=== {jid}.err (last 3000 chars, \\r → newline) ===")
    if err_path.exists():
        try:
            with err_path.open("rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 5000))
                tail = f.read().decode(errors="replace")
        except Exception:
            tail = ""
        tail = tail.replace("\r", "\n")
        print(tail[-3000:])
    print(f"\n=== regex match attempts on .err tail ===")
    info = parse_logs(jid)
    print(f"step={info.step!r}  pct={info.pct}  x_y={info.x_y!r}  rem={info.tqdm_remaining!r}")
    print(f"raw_last={info.raw_last!r}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--debug" and len(sys.argv) > 2:
        debug_one(sys.argv[2])
        return

    rows = get_running_jobs()
    if not rows:
        print("No running geodml-prob jobs found.")
        return

    # header
    hdr = ("JID", "ELAPSED", "MODEL", "TREATMENT", "FRAME(arg)", "STEP", "PROGRESS", "ETA-TO-END")
    fmt = "{:<8} {:<9} {:<14} {:<25} {:<18} {:<22} {:<14} {}"
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
        frame_col = f"{info.frame}({info.frame_arg})"
        print(fmt.format(
            jid, elapsed, info.model, info.treatment, frame_col,
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
