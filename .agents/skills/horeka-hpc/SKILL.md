---
name: horeka-hpc
description: Prepare, review, or troubleshoot Slurm, GPU, storage, environment, and AI inference work on NHR@KIT HoreKa. Use for HoreKa commands, job scripts, allocations, A100/H100/H200 workloads, or cluster failures.
---

# HoreKa HPC

Check the current [batch documentation](https://www.nhr.kit.edu/userdocs/horeka/batch/), [hardware overview](https://www.nhr.kit.edu/userdocs/horeka/hardware/), and [filesystem guide](https://www.nhr.kit.edu/userdocs/horeka/filesystems/) before choosing resources. Confirm the partition exists and the project can use it.

## Work safely

- Give commands for an already-open cluster shell unless Mac commands are requested. Never request passwords or MFA tokens in chat.
- Run compute through Slurm, not on login nodes.
- Before any `sbatch`, `salloc`, or allocation-bearing `srun`, estimate runtime, nodes, GPUs, CPUs, memory, resource-hours, and wall-time margin. Ask for explicit wall-time approval. Use placeholders until approved.
- Preserve live allocations. Do not cancel, exit, extend, requeue, replace, or retry without explicit authorization.
- Inspect `squeue`, `scontrol show job`, and `sacct` before diagnosing or retrying. Poll `squeue` no faster than every 30 seconds.

## HoreKa specifics

- Select the partition explicitly. Current GPU queues are `accelerated` for four A100 GPUs, `accelerated-h100` for four H100 GPUs, `accelerated-h200` for four H200 GPUs, and `accelerated-h200-8` for eight H200 GPUs.
- Development queues are for short development and testing. Only one development job may run per user; do not chain production work through them.
- Request GPUs explicitly with `--gres=gpu:N`, then verify `CUDA_VISIBLE_DEVICES` and `nvidia-smi` inside the job.
- Prefer scheduler defaults for memory unless measured need justifies an override. Request `--exclusive` when whole-node isolation or node-level energy attribution is required.
- Keep active datasets, models, and checkpoints in an allocated workspace or approved project storage. Use `$TMPDIR` only for disposable job-local data. Check space, inode quota, and workspace expiry before large writes.

Validate software and model fit on the actual node type before production. A working GH200 environment does not establish A100 compatibility. Record the Git SHA, configuration, seeds, model revision, environment, Slurm job ID, resources, timestamps, logs, and artifact paths. Never present smoke output as a scientific result.
