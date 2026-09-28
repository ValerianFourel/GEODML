# HoreKa Qwen sender handoff — 2026-09-28

Continues [2026-09-27_horeka-a100-bouts_handoff.md](2026-09-27_horeka-a100-bouts_handoff.md) for HoreKa;
the JUPITER state in [2026-09-28_jupiter-llama-final-sweep_handoff.md](2026-09-28_jupiter-llama-final-sweep_handoff.md)
is unchanged. Operator copy (same content):
`/Users/valerianfourel/Hamburg/GEODML_Unified/QwenHorekaHandoff260928.md`.
Snapshot: 28 Sep 2026, 11:18 CEST, from pasted HoreKa output; recheck before acting.

## Where things stand

- **Plan:** HoreKa owns all remaining Qwen work. The remaining cells
  (287,878 at division time) are split into 893 five-hour "bouts". Division:
  `/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/qwen-bouts/CURRENT`.
  Each bout writes to the shared HoreKa task ledger, which admits every cell
  once, so a bout can never repeat finished work.
- **Approved:** bouts 1–430. Each is 1 node × 4 A100 × 5 h. No automatic
  retries, no Hugging Face publication for now.
- **Finished:** bouts 1–30. 29 COMPLETED in about 2.3–4.7 h (≈30 s per cell).
  Bout 1 (job 5167563) hit its 5 h deadline (exit 124); its unfinished cells
  need a later leftover sweep.
- **Queued or running:** bouts 31–325, 295 jobs. At 11:18: 7 running
  (bouts 214–220, general `accelerated` nodes) and 288 pending.
- **Still to send:** bouts **326–430** (105 bouts), on the general `accelerated`
  partition without the `casualnet` reservation.
- **Why they aren't sent yet:** the account accepts at most **295 queued plus
  running jobs** (`AssocGrpSubmitJobsLimit`, measured when bout 325 went in and
  326 was refused). A slot opens only when a bout finishes.

## Simple instructions: send all remaining bouts

Page: `/Users/valerianfourel/Hamburg/GEODML_Unified/horeka-qwen-bouts.html`
(open it on the Mac with `open <that path>`). All commands go in a HoreKa
login shell.

1. **Start the automatic sender once.** Copy block **3b · Keep sending as
   room opens** from the page and paste it. It starts the tmux session
   `qwen-sender` and prints `STARTED`, the log path and the stop command.
   - Every 10 minutes, for at most 24 hours, it sends as many of bouts 326–430
     as fit under the cap of 295.
   - It stops by itself when all are sent (log line `ALL_SENT`).
   - It never sends a bout twice and only sends from the approved 326–430 list.
2. **Check it now and then:**
   `tail -n 20 <log path it printed>`. Expect one line per check,
   e.g. `queued 295 of cap 295, room 0`, and job IDs whenever a slot opened.
3. **If the log ends with `SENDER_FINISHED` before `ALL_SENT`**, the 24 hours
   ran out. Paste block 3b again.
4. **To stop sending:** `tmux kill-session -t qwen-sender`. This stops only the
   loop; queued and running bouts are untouched.

tmux sessions live on one login node. `tmux attach -t qwen-sender` works only
on the node where you started it; the `tail` of the log works from any node.

## Checking results

- Page **step 5 (Results)** shows every submitted bout: Slurm state, completed
  cells per bout from the ledger, throughput, and only the bouts needing
  attention (failed, deadline, zero cells) with their error lines.
- Quick overall view, in any HoreKa login shell:

  ```bash
  squeue --me -h -o '%T' | sort | uniq -c
  sacct -X -u "$USER" --starttime=2026-09-27 -n -P --format=State | awk '{print $1}' | sort | uniq -c
  ```

## Still open (not urgent)

1. Leftover sweep: cells still missing after their bout (e.g. bout 1's) need
   a later small list of bouts. Not implemented yet.
2. Publishing HoreKa's Qwen results to Hugging Face needs an HF **write** token
   on HoreKa (the current login there is read-only), then
   `publish_qwen_results.py` against the HoreKa dataset.
3. Bouts 431–893 are not approved yet: they need a fresh estimate and
   Valerian's approval, ideally sized from the measured ≈30 s per cell.
4. Never press Ctrl-Z on any running command (it freezes it with its locks
   and GPUs held); use Ctrl-C.
