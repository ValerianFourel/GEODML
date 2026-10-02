# Remove per-submission confirmation

Valerian explicitly requested removing the AGENTS.md requirement to ask before
each submission. Updated the workspace-root AGENTS.md and the active and Gemma
release checkout copies. A requested cluster task now authorizes necessary finite
submissions, retries/resumes and replacements within its scope and declared
budget, without separate per-job or wall-time confirmation. The policy explicitly
supersedes per-submission confirmation wording in HPC skills for this project.
Root milestone and wave wording was reconciled so it does not reimpose the same
confirmation under another heading.

Retained estimates and provenance, shortest defensible wall-time, default one-hour
maximum, finite budgets, five-allocation/start-spacing/storage admission rules,
saved-result reconciliation, no repeated completed work, and live-shell protections.
Longer allocations, scope/budget expansion, resource increases, and changing live
allocations still need explicit approval/instruction. The distinct JUPITER SSH
approval rule remains unchanged because this request concerns job submissions.

Files changed:
- /Users/valerianfourel/Hamburg/GEODML_Unified/AGENTS.md
- .worktrees/threehour-relaunch-fix/AGENTS.md
- .worktrees/gemma-selected-cells-release/AGENTS.md
The workspace root is outside Git; its edit persists on disk. Commit the tracked
checkout policy and handoff/index changes locally. No remote push is required for
this instructions-only request. Historical inactive checkout copies are untouched.

Read the last three active handoffs and checked all remaining allocation/approval
wording for contradictions. Diff whitespace checks pass. No executable changes
or tests are needed. No cluster command, allocation or inference was launched.
The preceding 5175121 time-left snapshot remains historical. The Gemma failed-only
recovery launcher remains outstanding; this policy change does not implement it.
