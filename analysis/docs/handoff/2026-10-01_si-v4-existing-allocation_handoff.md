# SI-v4 existing allocation detected, 1 October 2026

Valerian returned successful updates of all three sender scripts to CAP=294,
with backups stamped 20261001T090610Z-c7943077. The immediate account count was
still 295. A later SI-v4 admission check refused `SI-v4 allocation already
exists; do not request another`.

Confirmed this guard reads the current user's squeue jobs, not historical sacct
owners. It is duplicate-allocation protection, not a CAP update failure. Job ID,
state, node and remaining time are missing from the pasted output. Requested
read-only squeue output for geodml-gemma-si-v4 and the current shell's
SLURM_JOB_ID. Preserve all existing allocation shells; no new salloc, cancellation
or inference restart is indicated before identifying the existing job.

No runtime change, cluster execution or allocation ran locally. The returned
queue and CAP facts are pasted observations, not a new live query. Next step is
to identify the existing allocation and its owning shell before continuing.
