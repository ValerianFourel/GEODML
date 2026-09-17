# Batch inference endpoint security

The July 2026 JUPITER incident demonstrated that another cluster user could call
an inference endpoint on a compute-node network address without authentication.
That request consumed GPU resources. The exact historical launch command is not
available here, so the incident is not attributed to a particular current script.

The maintained launchers already specified `127.0.0.1`, but readiness accepted an
unauthenticated server. Several controllers also passed `api_key=None`.
Binding and authentication are separate controls. A port-ownership lock prevents
cooperating launchers from colliding; it does not authenticate HTTP clients.

## Access control belongs to the server invocation

Each server start receives a fresh random credential in `VLLM_API_KEY`. The
controller inherits that same credential and `GEODML_INFERENCE_AUTH_REQUIRED=1`.
The key is not an argument, a serving-profile field, or part of scientific task
identity. Restarting a server therefore does not change the identity of completed
tasks or cause them to be regenerated.

The common client uses the inherited key even when a historical caller supplies
`api_key=None`. Managed clients reject a missing or conflicting credential.
Their URL must name literal `127.0.0.1`. Redirects and environment-provided HTTP
proxies are disabled for managed requests.

Before starting a controller, the launcher verifies the expected model with the
correct key. It then requires authentication rejection for both an anonymous
request and an incorrect key on `/v1/models` and `/v1/chat/completions`.
The chat probes contain an empty JSON object, not an inference task. A validation
error, redirect, or successful anonymous response fails the security check.

The runtime record contains the access-control result, not the credential.
Server logs are private. Failure output redacts the active credential, including
server-log tails and client exceptions. HTTP error bodies are not copied into
client failure records or exceptions. Successful JSON responses are screened
after decoding as well, so escaped credentials cannot pass through model identity
errors or saved outputs. A response that reflects the credential is rejected
rather than saved as a scientific result.

Before starting any model process, the shared launcher verifies one of two
explicit network boundaries. Its default path re-executes into a fresh Linux
user and network namespace. Server and controller share that private network,
which must have only the loopback interface. Verification reads the actual
process namespace against a retained outside-namespace descriptor, checks the
interface list, and tests a private loopback connection. An environment marker
alone is not proof.

The generator backlog uses a whole-node Slurm boundary because JUPITER Booster
nodes rejected unprivileged `unshare` with `ENOSPC` on 16 September 2026. Its
submission command includes `--exclusive`. Before loading a model, the worker
queries its own array element through `scontrol` and requires a running,
single-node job with `Exclusive=NODE`, or legacy Slurm's equivalent `Shared=0`,
on the allocated node. JUPITER's Slurm 25.05 build omits both fields; for that
representation the verifier requires `OverSubscribe=NO` and independently
matches the job's one-node CPU allocation in both `NumCPUs` and `AllocTRES`
against the positive `SLURM_CPUS_ON_NODE` value. If `Exclusive` is present it
remains authoritative, so conflicting scheduler evidence is rejected. Missing
or conflicting scheduler evidence stops the job. This path retains literal
loopback binding, per-run native authentication, and loopback-only NCCL/Gloo
transport. It does not claim that a private kernel network namespace exists.
Other launchers continue to require the namespace path unless they explicitly
request and verify the whole-node boundary.

The helper enables loopback only inside the new namespace. It does not change
the host firewall or network interfaces. NCCL/Gloo socket traffic is restricted to
loopback and NCCL RDMA is disabled; single-node CUDA/NVLink communication remains
available, subject to a real GPU compatibility test. No model download can occur
inside this isolated stage; required snapshots must already be cached.

## Maintained launch paths

`analysis/scripts/search_vllm_stage.py` owns the current generator, post-processing,
and judge server lifecycle. It applies the checks regardless of the frozen
serving profile's age. Profiles retain their original hashes and model settings.

Four older JUPITER scripts launch vLLM directly and share a shell security helper.

- `run_acl_arr_document_pilot_4gpu.sh`
- `run_acl_arr_llama_answer_4gpu.sh`
- `run_acl_arr_correction_smoke_4gpu.sh`
- `recover_acl_arr_llama_rerank_4gpu.sh`

These changes apply only to jobs launched from the updated commit. Already-running
jobs, historical worktrees, external launch commands, and SSH port forwards are
not changed by local edits.

## Native authentication is not host isolation

The selected design uses native vLLM authentication with mandatory loopback
binding. An authentication proxy was considered but rejected for this milestone.
A proxy adds another server, streaming path, and shutdown sequence. A proxy over
an unprotected TCP backend also leaves a local bypass.

vLLM documents that native API keys do not protect every management route.
Its distributed dependencies can open additional listeners, including a PyTorch
TCPStore on all interfaces. Setting `VLLM_HOST_IP=127.0.0.1` is not proof that every
dependency listener is private. See the [vLLM security guidance](https://docs.vllm.ai/en/latest/usage/security/).

The generator submission helper checks that the boundary helper exists, but
does not execute a namespace probe on the login node. Login and compute nodes
can have different namespace policies. The scheduler must confirm whole-node
exclusivity from inside each opted-in generator worker. Submission alone is not
proof. Namespace-based launchers still verify isolation on the allocated node
before loading the model or starting vLLM. See the
[Linux network namespace documentation](https://man7.org/linux/man-pages/man7/network_namespaces.7.html)
and the [Slurm `--exclusive` contract](https://slurm.schedmd.com/sbatch.html#OPT_exclusive).
There is no automatic retry or downgrade from either boundary.

Per-run credentials do not protect against administrators, another process under
the same Unix account, or a compromised model runtime. No source change can
certify existing cluster jobs without inspecting them.

## Verification boundary

CPU tests exercise an actual local HTTP server for positive and negative probes,
fake child processes for credential propagation, and the real shell helpers.
Namespace tests mock kernel facts on macOS; a real Linux namespace check is
separate and does not claim to validate JUPITER or its CUDA stack.
They do not load a model or allocate GPUs. The shared-backlog tests separately
verify durable task ownership and resume behavior.

The approved one-hour Nemotron benchmark remains capped at four GPU-hours with
no automatic resubmission. This fix does not launch that job. Cluster listener
and namespace verification is still required before calling the deployment protected.
