# HoreKa resources and model fit

Snapshot: 24 September 2026, from Valerian's Slurm and quota outputs.
Account: `hk-project-p0026831`; user: `uhh_bbf7367`; legacy cluster: `hk`.

## Compute

| Resource | Remaining budget or hardware |
|---|---|
| Shared project GPU budget | **376,141 GPU-hours** of 478,000 granted |
| Shared project CPU budget | **7,046,444.5 CPU-hours** of 9,082,000 granted |
| Project reservation `casualnet` | 16 accelerated nodes, 64 A100 GPUs; scheduled through 31 October 2026 at 18:00 |
| Each accelerated node | 4 × A100 40 GB, **160 GB aggregate GPU memory** |
| Host memory and CPUs per node | 512 GiB RAM; 76 physical cores, 152 threads |

Budgets and the reservation are shared across the project. Free nodes are unknown:
Slurm denied node-status access. Reservation size does not establish immediate
availability. [Accounting units](https://www.nhr.kit.edu/userdocs/horeka/accounting/),
[node specifications](https://www.nhr.kit.edu/userdocs/horeka/hardware/).

H100/H200 partitions appeared in Slurm, but access remains unverified during
[HoreKa 2 migration](https://docs.nhr.kit.edu/get-started/migration/).

GEODML permits at most five concurrent allocations per cluster, with ten-minute
start spacing and explicitly approved wall-time, normally at most one hour.
One four-GPU allocation-hour consumes 4 GPU-hours.

JUPITER is chief; HoreKa is secondary. A shared work-hour is sized against four
JUPITER GH200s, not one HoreKa wall-clock hour. A100 throughput needs measurement.
See [shared-hour rules](../analysis/docs/agentic_shared_hours.md).

## Storage and placement

Figures use quota-output GB units. Headroom is approximate space below the soft
quota, not the filesystem-wide free capacity shown by `df`.

| Storage | Used / soft quota | Headroom | Contents |
|---|---:|---:|---|
| Project home, shared | 2,955 / 10,240 GB | **7,285 GB** | Code, configuration, durable results |
| Personal work quota | 13 / 256,000 GB | **255,987 GB** | Model cache, datasets, active runs |

File-count headroom: 8,324,165 in project home; 52,390,938 in personal work.

- Keep code at `$HOME/geodml`.
- Use an allocated `/hkfs/work` workspace with `models/`, `datasets/`, `runs/`
  and `logs/` directories. No active workspace was shown; the old `ragdag`
  workspace has been unavailable since 30 August.
- Use `$TMPDIR` for disposable runtime files, roughly 800 GB per node, deleted
  after the job. Keep completed results outside it.
- Workspaces expire and have no backup. Preserve durable results in project home
  and publish verified dataset shards to the private Hugging Face repository.
  Never delete unsynchronized results. [Storage guidance](https://www.nhr.kit.edu/userdocs/horeka/filesystems/).

## Current study models

Estimated BF16 weights only, with one model distributed across four A100s:

| Model | Weight memory | One 160 GB A100 node |
|---|---:|---|
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | About 54–56 GB | Memory-feasible; serving unvalidated |
| [Nemotron Nano 30B-A3B-BF16](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf) | About 63 GB | Memory-feasible; serving unvalidated |
| [Llama 4 Scout 17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) | About 218 GB | **Does not fit** |

Estimates use roughly two bytes per parameter and exclude context caches and
runtime overhead. Scout stores 109B parameters despite activating 17B per token.
Host RAM does not increase GPU memory.

The current judge is Nano, as pinned in the
[judge preparation script](../analysis/scripts/prepare_agentic_judge_pilot.py).
The [continuation configuration](../analysis/scripts/prepare_agentic_adaptive_500.py)
records the current model revisions.

Place Qwen and Nano on HoreKa only after validating their frozen serving profiles.
Keep Scout on JUPITER. Preserve revisions, precision, context and scientific
settings; do not silently quantize, offload or switch to multiple nodes.

## Before execution

Validate the actual compute-node Slurm exclusivity boundary, model runtime,
workspace lifetime and private Hugging Face synchronization. Login-node public
HF connectivity does not establish private write access or compute-node networking.
The [HoreKa profile](../analysis/config/shared_hours/horeka.template.json) currently
lists no validated models. This document authorizes no allocation.

[Qwen download and input preparation](HOREKA_QWEN_PREPARATION.md) is the next step.
