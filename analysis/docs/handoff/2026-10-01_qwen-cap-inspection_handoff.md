# Qwen CAP update refused, 1 October 2026

Valerian pasted `REFUSED unfamiliar CAP setting` for shared-workspace
`qwen-bouts/send-what-fits.sh`. The updater validates all scripts before its
write loop, so this refusal changed none of them. The live CAP assignment is
not present in the pasted output and cannot be inferred from local templates.

Requested a read-only `sed -n '/^CAP=/p'` of that exact file from the already-open
HoreKa shell. Await its output before adjusting the updater; do not bypass the
guard or assume CAP=294 took effect. No cluster command, allocation, cancellation
or runtime edit occurred in this turn. Locally confirmed the validation/write
ordering in the HTML. Earlier queue facts remain historical.
