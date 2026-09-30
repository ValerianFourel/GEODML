# Explicit interactive salloc workflow

Valerian requested modifying the Gemma page to use interactive salloc. The
existing page already allocated via salloc but launched run.sh as a separate
srun command. The page now explicitly opens an interactive compute terminal:
salloc runs a child command which executes srun with the allocation's explicit
SLURM_JOB_ID and --pty bash -l. From that terminal the user runs bash run.sh
directly in a child subshell, retaining progress logging and the parent shell
after success or failure. No batch submission, cancellation or release command.

The one-hour approval remains the same single allocation, not an additional
allocation. Page directs the operator to reuse an existing approved Gemma
allocation and skip allocation setup. Admission, quota and duplicate-attempt
guards remain. Code pin remains ddc93fe5342152d7c33f2abfc0658a48714d7dcc.
Code and prior HTML were successfully pushed through 5e18bc3 in the preceding
turn; removed the stale instruction to push before using the page.

Changed the tracked HTML and Markdown guide; copied HTML to workspace-root
horeka-gemma-si-v3.html. Validation: extracted shell blocks and inner interactive
shell command passed bash -n, embedded Python parsed, identical copies confirmed,
git diff --check passed. No cluster command executed and no live state observed.
