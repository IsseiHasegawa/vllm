# SPDX-License-Identifier: Apache-2.0
"""Minimal JSONL logger for phase-level benchmarking instrumentation.

Activation:
    Set the environment variable ``VLLM_PHASE_LOG_DIR`` to a writable
    directory before starting ``vllm serve``. When unset, all hooks are
    no-ops (ENABLED is False and call sites skip the record construction).

Output:
    One JSONL file per (kind, process): ``<kind>-<pid>.jsonl``.
    The frontend (API server) process writes ``requests-*.jsonl`` and the
    EngineCore process writes ``steps-*.jsonl``. The first line of every
    file is a ``{"record": "meta", ...}`` header. Every record carries a
    per-process monotonically increasing ``seq`` and the writer ``pid``.

Design constraints:
    * stdlib only, no imports from vllm (no circular-import risk).
    * Never raises into the caller: instrumentation must not break serving.
    * Buffered (flush every _FLUSH_EVERY records) to keep per-call cost
      in the microsecond range; explicit flush on interpreter exit.
    * Fork-aware: if the pid changes, a fresh file is opened.

Run attribution is NOT handled here by design: the server outlives
individual benchmark runs, so runs are reconstructed offline by slicing
records on ``ts`` (wall clock) against the runner's manifest.csv.
"""

import atexit
import json
import os
import socket
import threading
import time

_LOG_DIR: str = os.environ.get("VLLM_PHASE_LOG_DIR", "")

# Read once at import; call sites use this to skip work when disabled.
ENABLED: bool = bool(_LOG_DIR)

_FLUSH_EVERY = 200

_lock = threading.Lock()
# kind -> {"fh": file, "pid": int, "seq": int, "buf": list[str]}
_state: dict = {}


def _open(kind: str) -> dict:
    os.makedirs(_LOG_DIR, exist_ok=True)
    pid = os.getpid()
    path = os.path.join(_LOG_DIR, f"{kind}-{pid}.jsonl")
    fh = open(path, "a")
    st = {"fh": fh, "pid": pid, "seq": 0, "buf": []}
    _state[kind] = st
    meta = {
        "record": "meta",
        "ts": time.time(),
        "pid": pid,
        "host": socket.gethostname(),
        "kind": kind,
    }
    fh.write(json.dumps(meta) + "\n")
    fh.flush()
    return st


def _flush_locked(st: dict) -> None:
    if st["buf"]:
        st["fh"].write("".join(st["buf"]))
        st["buf"].clear()
    st["fh"].flush()


def _flush_all() -> None:
    with _lock:
        for st in _state.values():
            try:
                _flush_locked(st)
            except Exception:
                pass


atexit.register(_flush_all)


def log(kind: str, record: dict) -> None:
    """Append one record to <kind>-<pid>.jsonl. Never raises."""
    if not ENABLED:
        return
    try:
        with _lock:
            st = _state.get(kind)
            if st is None or st["pid"] != os.getpid():
                st = _open(kind)
            st["seq"] += 1
            record["seq"] = st["seq"]
            record["pid"] = st["pid"]
            st["buf"].append(json.dumps(record) + "\n")
            if len(st["buf"]) >= _FLUSH_EVERY:
                _flush_locked(st)
    except Exception:
        # Instrumentation must never break serving.
        pass


def flush() -> None:
    """Force pending records to disk (e.g., from tests)."""
    _flush_all()
