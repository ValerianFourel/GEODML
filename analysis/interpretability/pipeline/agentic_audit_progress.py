"""Opt-in audit counters on stderr; no scientific data or tokens are logged."""
from __future__ import annotations

import os
import sys
import threading
import time
from contextvars import ContextVar
from functools import wraps

_current = ContextVar('audit_progress', default=None)


def audit_progress(**counts):
    state = _current.get()
    if state is not None:
        state.update(counts)


def audit_stage(name, *, interval=10.0):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if os.environ.get('GEODML_AUDIT_PROGRESS') != '1':
                return function(*args, **kwargs)
            state = {}
            started = time.monotonic()
            stopped = threading.Event()
            def emit(status):
                fields = ' '.join(f'{k}={v}' for k, v in state.copy().items())
                print(f'AUDIT stage={name} status={status} elapsed_s={time.monotonic()-started:.1f} {fields}',
                      file=sys.stderr, flush=True)
            def heartbeat():
                while not stopped.wait(interval):
                    emit('working')
            token = _current.set(state)
            emit('started')
            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            status = 'failed'
            try:
                result = function(*args, **kwargs)
                status = 'finished'
                return result
            finally:
                stopped.set()
                thread.join()
                emit(status)
                _current.reset(token)
        return wrapped
    return decorate
