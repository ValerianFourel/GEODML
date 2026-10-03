# Gemma preparation pending

Returned HoreKa output identifies job 5177483 as geodml-gemma-v4-prepare,
PENDING with reason Priority. Sender log contains two PENDING observations
600 seconds apart. Unrelated Qwen job 5177268 is also pending. These are pasted
observations, not live cluster queries.

Valerian proposed cancelling and submitting sbatch immediately. Inspected the
pinned launcher start path: preparation already uses sbatch, and the sender
tracks its saved receipt. Cancellation without a completed plan raises an error;
restarting the launcher does not automatically replace preparation. No cancellation,
replacement, submission, or code change was performed. Recommending preservation
of the pending preparation and scheduler start/priority inspection, because
resubmitting the same resources does not bypass scheduler priority.

After successful preparation the existing sender can submit up to 200 planned
inference jobs. Shards and plan must exist first. Read the three preceding indexed
handoffs. Documentation-only update; whitespace verification is sufficient.
