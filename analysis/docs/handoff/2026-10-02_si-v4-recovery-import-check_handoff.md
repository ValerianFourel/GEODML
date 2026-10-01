# SI-v4 recovery checkout import check

User's existing-job recovery command stopped at import with ModuleNotFoundError: analysis, before reading the receipt or starting inference. No HASH_REPAIR_VERIFIED output has been supplied. Corrected checkout existence and completed preparation are not established. The exact cluster cause could be missing checkout or Python ignoring PYTHONPATH; do not assert which without evidence. Previous allocation timing is stale pasted evidence.

Updated only the HTML recovery instructions. Step 2 now explicitly checks the corrected module file and verification receipt before importing, with messages directing the user to step 1. Both rebuild and execution preflight pass the checkout as an argument and insert its resolved path into sys.path. Scientific code, execution pin and prepared artifacts are unchanged. No package installation is appropriate.

Verification follows verify-and-stop: both command blocks pass bash -n and embedded Python AST parsing; explicit checkout import passes in isolated Python mode ignoring PYTHONPATH; missing-checkout and missing-receipt cases both exit before Python/inference; exported root and Downloads copies match. git diff --check passed. Existing source-fix test results remain current because no product code changed. No remote commands or inference executed. No current allocation status inferred.

Next: user runs step 1 in updated HTML, supplies HASH_REPAIR_VERIFIED or full error. Only then use step 2, which retains job 5174124 and at least 20 minutes remaining checks. Preserve live allocation and failed preparations; do not bypass time checks or issue another allocation without approval.
