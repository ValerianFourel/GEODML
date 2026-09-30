# Gemma export retry with hidden write-token prompt

Valerian ran the exporter on HoreKa login hkn1990. Local archive completed with
no missing expected files, 1,253,914 bytes, SHA256
31f3666c64e2f44d58cc4f06411f87f828ad454fee9c1b506c6c915dba77e62f.
Path: $W/reviews/gemma-si-v3-exports/<sha256>.zip. HF preupload returned 403 with
explicit message that a write token is required. No verified upload receipt.
This is an authorization failure, not a judgment or archive-generation error.

User requested that the command ask for a write token. Updated operator HTML
to reuse and hash-check that exact archive and call the existing pinned publish
function. Python getpass prompts on the operator's terminal with echo disabled;
GetPassWarning is treated as an error to prevent an echoing fallback. Empty
input aborts. Token enters only this child process's HF_TOKEN and is not printed,
written to a file or stored as an HF login. Shell tracing disabled. User must
use a token with write access to the existing private target repository.

Existing helper pin dcd8588b4a3daee31362275167684218e6e607b0 unchanged. Upload
still uses content-addressed path, private-repo check, CAS commit and exact
revision readback. Receipt written beside ZIP only after verification. No
repacking, re-running judgments, new allocation or modification of prior runs.

Recovery Bash and embedded Python parse; both HTML copies match; whitespace
check passed. No live upload executed by Codex. Page reopened. Next user runs
the command, enters token locally and sends HF_EXPORT receipt. Codex reads that
exact archive and completes timing/error/raw-judgment review. Never ask for
tokens in chat. HF read access from local Mac still requires network access.
