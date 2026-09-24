"""Local receipts for unchanged immutable files, never for mutable task state.

Receipts avoid reading large payloads on every allocation. They rely on normal
filesystem change metadata, not on detecting silent corruption or hostile local
writers. A forced audit ignores receipts. Nothing here is uploaded to the Hub.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

from .agentic_audit_progress import audit_progress


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class VerificationCache:
    def __init__(self, root: Path, *, force=False):
        self.root = root.resolve()
        self.path = self.root / 'local-only/verification-cache-v1.json'
        self.force = force
        self.entries = {}
        self.checked = self.reused = 0

    def __enter__(self):
        if self.path.is_symlink() or not self.path.resolve().is_relative_to(self.root):
            raise ValueError('verification receipt escapes dataset')
        try:
            receipt = json.loads(self.path.read_bytes())
            body = receipt['body']
            if (receipt['sha256'] == _digest(body) and body['version'] == 1
                    and body['root'] == str(self.root) and isinstance(body['entries'], dict)):
                self.entries = body['entries']
        except (FileNotFoundError, ValueError, KeyError, TypeError):
            pass  # Absent/invalid receipts require real verification.
        return self

    def signature(self, path):
        path = Path(path)
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError('verification path escapes dataset or is a symlink')
        value = path.stat()
        if not stat.S_ISREG(value.st_mode):
            raise ValueError('verification requires a regular file')
        return [str(path.relative_to(self.root)), value.st_dev, value.st_ino,
                value.st_size, value.st_mtime_ns, value.st_ctime_ns]

    def check(self, key, paths, verify):
        before = [self.signature(p) for p in paths]
        if not self.force and self.entries.get(key) == before:
            self.reused += 1
            valid = True
        else:
            valid = verify()
            if before != [self.signature(p) for p in paths]:
                raise ValueError('file changed during verification; retry after writes stop')
            self.checked += 1
            if valid:
                self.entries[key] = before
            else:
                self.entries.pop(key, None)
        audit_progress(verification_checked=self.checked, verification_reused=self.reused)
        return valid

    def file(self, path, expected):
        expected = {key: expected[key] for key in ('sha256', 'bytes')}
        def verify():
            hasher = hashlib.sha256()
            size = 0
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    size += len(chunk)
                    hasher.update(chunk)
            return hasher.hexdigest() == expected['sha256'] and size == expected['bytes']
        return self.check('file-' + _digest([str(path.relative_to(self.root)), expected]), [path], verify)

    def reference(self, reference, manifest, shard, verify):
        return self.check('reference-' + _digest(reference), [manifest, shard], verify)

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            # A failed forced audit must not leave old proofs usable, including
            # when corruption was found without a filesystem metadata change.
            self.entries = {}
        if kind is None or self.path.exists():
            # Concurrent writers may lose cache hits, never validation: each entry
            # contains its own signatures, and only fully written receipts replace.
            from .agentic_hour_sync import atomic
            body = {'version': 1, 'root': str(self.root), 'entries': self.entries}
            atomic(self.path, json.dumps({'body': body, 'sha256': _digest(body)}, sort_keys=True).encode())
            os.chmod(self.path, 0o600)
