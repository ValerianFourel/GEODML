# Separate Gemma checks and interactive allocation

User pasted the admission failure: 28 running Qwen allocations exceeded the
default five-active guard. A subsequent malformed heredoc paste was cancelled
before execution. Those shown attempts did not reach salloc. User explicitly
approved one one-hour Gemma allocation alongside the existing Qwen queue,
overriding the five-active and no-pending restrictions for this test only.
Then required no job cancellation and noted the 295-job queue limit. Latest
instruction: a separate salloc command in the HTML.

Updated only documentation. Page now has seven distinct copy blocks: download,
prepare, check, plain salloc, optional interactive shell, direct run.sh, results.
Step 3 uses Python -c instead of a heredoc, records the explicit scheduling
exception, counts account queue jobs with expanded arrays, blocks at 295, keeps
fresh storage and ten-minute observed-start checks, and refuses existing Gemma
allocations/attempts. No Slurm mutations occur in checks. Step 4 is just salloc;
the user must run it once after CHECK PASSED. The preflight no longer creates an
allocation-attempt marker because it does not execute salloc. It still refuses
an existing marker from an earlier attempt; inspect/reconcile rather than clear.
The queue may change after the snapshot; site enforcement and operator
coordination remain necessary. No cancellation, holds, requeues or automatic
retry. Existing allocations must remain open.

Inference pin and existing prepared run remain ddc93fe. No need to repeat
download or preparation: user starts at step 3 in the updated page. Tracked page
is analysis/docs/horeka-gemma-si-v3.html; identical convenient copy at workspace
root horeka-gemma-si-v3.html.

Validation: seven extracted Bash blocks passed bash -n; embedded Python parsed;
salloc block verified standalone and admission has no heredoc. Executed the
actual admission Python locally with mocked external scheduler/quota reads:
28 running + pending at 294 account jobs passes; 295 jobs, a recent start, and
unsafe storage each block without recording a passed admission. Copies match,
git diff --check passed. No HoreKa commands, allocations or job changes executed.
