# V4 evidence transfer for content review

Valerian requested a HoreKa command to upload the saved v4 inference file to
Hugging Face, then analyze it here. This authorizes transfer of the requested
evaluation artifacts to the existing private experiment dataset. No new model
inference or repair is requested.

Read the last three indexed handoffs. Confirmed local cached Hugging Face
authentication and live access to the private dataset
`ValerianFourel/geodml-experiment-v2-paper-private`, at revision
`73a0c123f66c7ad03c9d291c8fec9a56e247944b`. Its returned file list had no path
containing v4 plus inference/evidence/assessment. No upload or cluster command
was executed by Codex.

Provide an existing-shell command that sources the HoreKa environment, selects
the newest assessment-all directory, requires its FULL_CONSOLE completion
marker and inference dump, verifies the destination remains private, then uses
HfApi.upload_folder with an explicit allowlist: all-inference.jsonl,
assessment.json, resource-usage.json and evaluation-evidence.tar.gz. Use a
unique remote folder under reviews/si-v4-r2-cycle-20261002/development.
Return the commit-pinned URL and dump SHA-256. Use existing authentication or
a hidden terminal token prompt, never a token in chat.

Next: user runs the transfer command and returns UPLOADED. Retrieve the exact
revision locally using /Users/valerianfourel/miniconda3/bin/python and the
available huggingface_hub installation. Verify the dump hash, then review the
actual questions, maps, complete sources, support findings and grades. Keep
qualitative judgments separate from the already documented formal completion
and cost failures. Analysis remains pending the actual artifact transfer.
