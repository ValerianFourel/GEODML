# Resume login preparation using job-specific cancellation evidence

Returned exact sacct output confirms 5177483 CANCELLED by 1025084, Start=None,
End=2026-10-03T21:02:56, ExitCode=0:0. The preparation is absent from the live
queue and scontrol no longer retains it. The old controller had stopped waiting
for terminal accounting. Login preparation has not started. hkn1990's absent tmux
server does not establish hkn1991 session state; tmux is host-local. Shared run
locks remain authoritative against a duplicate controller.

Added targeted sacct selection by preparation job ID with State%40, parsed actor
suffix and preserved raw evidence. The live snapshot is read last and still wins.
A cancelled never-started job missing from broad accounting can now be confirmed.
The exact reason the older broad query failed on the cluster is not established;
this repair uses the positive exact-ID evidence the user supplied.

Allows a narrow recovery from helper 547cb5cc2c5a211054fb0ead8b620a60a17e405f
only before any frozen inputs, shards, plan or inference sender state exists.
Archives the old takeover receipt before pin migration. Original scientific pin,
queued jobs, task identities, resource budgets and no-overwrite checks stay intact.
The resumed helper clears/verifies dependencies again, records cancelled preparation,
runs login preparation, adopts/releases existing jobs and invokes the original
finite sender. No new wave is created.

Regression first failed on the old implementation at the same cancellation wait.
Added coverage for missing broad cancellation records and resuming the actual old
helper receipt. Focused runtime/sender/prequeue suite and whitespace/shell checks
run before publishing. No cluster jobs executed by Codex. User urgently requests
running now; provide one pinned fetch/checkout/resume command, logging in tmux.
