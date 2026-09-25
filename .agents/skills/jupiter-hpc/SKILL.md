---
name: jupiter-hpc
description: Prepare, review, or troubleshoot Slurm, GPU, storage, environment, and AI inference work on JSC JUPITER. Use for JUPITER commands, job scripts, allocations, GH200 workloads, or cluster failures.
---

# JUPITER HPC

Use current JSC documentation as authority. JUPITER is in early access, so check the [known issues](https://apps.fz-juelich.de/jsc/hps/jupiter/known-issues.html), [batch system](https://apps.fz-juelich.de/jsc/hps/jupiter/batchsystem.html), and [GPU guide](https://apps.fz-juelich.de/jsc/hps/jupiter/gpu-computing.html) before relying on partition, software, or Slurm behavior.

## Work safely

- Give commands for an already-open cluster shell unless Mac commands are requested. Never request MFA tokens in chat.
- Run compute through Slurm. Use `srun` to start work inside an allocation; JUPITER does not support `mpiexec` for MPI launch.
- Before any `sbatch`, `salloc`, or allocation-bearing `srun`, estimate runtime, nodes, GPUs, CPUs, memory, resource-hours, and wall-time margin. Ask for explicit wall-time approval. Use placeholders until approved.
- Preserve live allocations. Do not cancel, exit, extend, requeue, replace, or retry without explicit authorization.
- Inspect `squeue`, `scontrol show job`, and `sacct` before diagnosing or retrying. A submission proves neither execution nor results.

## JUPITER specifics

- The `booster` partition currently provides exclusive nodes with four GH200 superchips, each containing one H100 GPU with 96 GB HBM. Request GPUs explicitly with `--gpus=1..4`; verify `CUDA_VISIBLE_DEVICES` and `nvidia-smi` inside the job.
- Load required modules inside job scripts. This avoids environment mismatches after Slurm/backend changes.
- Treat ARM64 compatibility separately from CUDA compatibility. Verify wheels, containers, vLLM, PyTorch, and model revisions on JUPITER before scaling.
- Keep large models, datasets, checkpoints, and logs on the approved project or scratch filesystem. Check byte and inode quotas before large writes.
- For GEODML inference, use the maintained exclusive-node verifier and set `GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY=1` after environment setup. Fail closed; do not fall back to `unshare`.

Start with the smallest approved compatibility run. Record the Git SHA, configuration, seeds, model revision, environment, Slurm job ID, resources, timestamps, logs, and artifact paths. Never present smoke output as a scientific result.
