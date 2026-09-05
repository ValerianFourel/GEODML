# Codex cloud environment

Configure the repository in Codex cloud with these values:

- Repository: `ValerianFourel/GEODML`
- Branch for the current experiment: `testing-skills`
- Runtime: Python 3.11
- Setup script: `bash .codex/setup.sh`
- Maintenance script: `bash .codex/setup.sh`
- Agent internet access: disabled unless a task specifically needs current public sources

The setup installs the CPU analysis dependencies into `.venv`. It does not
install vLLM, download model weights, fetch research datasets, or connect to
JUPITER. GPU inference and large-data work remain separate Slurm milestones.

Codex cloud sees the complete committed repository at the selected Git branch.
It does not see uncommitted files or the live JUPITER filesystem. When code is
edited on JUPITER, commit and push it to `testing-skills` before starting or
refreshing a cloud task. Keep model weights, raw datasets, and large run outputs
on JUPITER. Commit only portable scripts, configurations, manifests, concise run
summaries, and documentation needed to understand or reproduce the work.

Use `.venv/bin/python` for Python commands inside cloud tasks. For example:

```bash
.venv/bin/python -m pytest -q analysis/tests/test_acl_arr_document_experiment.py
```
