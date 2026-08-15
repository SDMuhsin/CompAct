"""Lock-protected atomic upsert of one row into a results CSV.

WHY THIS IS ITS OWN MODULE
--------------------------
These three functions used to live in `train_glue.py`, and every writer therefore paid for
`train_glue`'s module scope to write a CSV row. That is not a style complaint, it cost this project
three round trips on the fir cluster:

  * `run_production.py:write_row` did `from train_glue import write_result_row`, and `train_glue`
    imports galore_torch (:82), lion_pytorch (:98) and adapters (:101) at MODULE scope. Preflight
    jobs 54306984 / 54607464 / 54609748 trained all four arms successfully and then died writing
    the CSV, on an optimizer package none of them used.
  * The import costs **7.7 s and 1.0 GiB RSS** (measured 2026-08-14) because it pulls torch,
    transformers, peft and adapters. At the 100s-of-parallel-writers scale a camera-ready sweep
    runs at, that is ~200 GiB of RSS spent on `DataFrame.to_csv`.

Nothing here imports torch. `train_glue` re-exports these names, so existing callers are unchanged.

CONCURRENCY CONTRACT -- READ BEFORE RUNNING A LARGE SWEEP
--------------------------------------------------------
The protocol is: take an exclusive `FileLock`, read the whole CSV, upsert one row, write a temp file,
`os.replace` it over the original. `os.replace` is atomic within a filesystem, so a reader never sees
a partial file and a killed writer cannot corrupt the CSV.

⚠ **THE LOCK IS `fcntl.flock`, AND THAT IS ONLY AS GOOD AS THE FILESYSTEM.** `filelock` uses
`fcntl.flock` on Unix. On a shared parallel filesystem this is the one assumption that can silently
fail:
  * Lustre mounted with `-o localflock` gives NODE-LOCAL locks. Two array tasks on DIFFERENT NODES
    would then BOTH hold "the" lock, interleave their read-modify-write, and the second `os.replace`
    would silently discard the first's row. Nothing raises; the sweep just comes up short.
  * Lustre mounted with `-o flock` (cluster-wide) is correct.
  * NFS without `lockd` is likewise unsafe.
**Verify before trusting a multi-node sweep** -- `verify_flock_scope()` below, or check the mount
options. If cluster-wide locking is not available, do not fan 100s of writers at one CSV: give each
task its own file and aggregate afterwards (`shard_path()`), which is the pattern `train_glue`
already uses for its per-seed partials.
"""

from __future__ import annotations

import logging
import os
import random
import socket
import time
from functools import reduce
from typing import Dict, List
import operator

import numpy as np
import pandas as pd
from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)


def _load_results_df(results_file: str, columns: List[str]) -> pd.DataFrame:
    if os.path.isfile(results_file):
        df = pd.read_csv(results_file)
        for c in columns:
            if c not in df.columns:
                df[c] = np.nan
        return df[columns]
    return pd.DataFrame(columns=columns)


def _upsert_result(df: pd.DataFrame, comb_cols: List[str], row_dict: Dict) -> pd.DataFrame:
    mask = reduce(
        operator.and_, [(df[col] == row_dict[col]) for col in comb_cols], pd.Series(True, index=df.index)
    )
    df = df[~mask]
    df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    return df


def write_result_row(results_file: str, all_columns: List[str], comb_cols: List[str],
                     result_row: Dict, timeout: int = 300, max_retries: int = 5) -> bool:
    """Hardened, atomic, lock-protected upsert of one row into the results CSV."""
    lock_file = results_file + ".lock"
    os.makedirs(os.path.dirname(os.path.abspath(results_file)), exist_ok=True)
    lock = FileLock(lock_file, timeout=timeout)
    for attempt in range(max_retries):
        try:
            with lock:
                logger.info(f"Acquired lock on {lock_file} (attempt {attempt + 1}).")
                df_results = _load_results_df(results_file, all_columns)
                df_results = _upsert_result(df_results, comb_cols, result_row)
                # ⚠ HOSTNAME AS WELL AS PID. The temp name used to be `.tmp.<pid>` alone, and PIDs
                # are NOT unique across nodes -- two array tasks on different hosts can share one.
                # Inside a working cluster-wide lock that is harmless, but if locking degrades to
                # node-local (see the module docstring) it turns a lost row into two writers
                # clobbering one temp file, which is far harder to diagnose.
                tmp_file = f"{results_file}.tmp.{socket.gethostname()}.{os.getpid()}"
                df_results.to_csv(tmp_file, index=False)
                os.replace(tmp_file, results_file)
                logger.info(f"Released lock. Logged results to {results_file}")
            return True
        except Timeout:
            wait = 2 ** attempt + random.uniform(0, 1)
            logger.warning(f"Lock timeout on attempt {attempt + 1}/{max_retries}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    logger.error(f"Failed to acquire lock on {lock_file} after {max_retries} attempts. Results NOT saved.")
    return False


def shard_path(results_file: str, task_id: str) -> str:
    """Per-task shard path, for sweeps too wide (or filesystems too weak) for one shared CSV.

    Writing one file per task needs no lock at all, so it is correct under ANY filesystem, and the
    shards concatenate afterwards. Use it when the writer count is in the hundreds or when
    `verify_flock_scope()` cannot confirm cluster-wide locking.
    """
    base, ext = os.path.splitext(results_file)
    return f"{base}.shard-{task_id}{ext or '.csv'}"


def merge_shards(results_file: str, all_columns: List[str], comb_cols: List[str],
                 remove: bool = False) -> pd.DataFrame:
    """Concatenate every `shard-*` beside `results_file` into it, last write winning per key."""
    import glob
    base, ext = os.path.splitext(results_file)
    shards = sorted(glob.glob(f"{base}.shard-*{ext or '.csv'}"))
    frames = [_load_results_df(results_file, all_columns)] if os.path.isfile(results_file) else []
    for s in shards:
        frames.append(_load_results_df(s, all_columns))
    if not frames:
        return pd.DataFrame(columns=all_columns)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=comb_cols, keep="last").reset_index(drop=True)
    tmp = f"{results_file}.tmp.merge.{os.getpid()}"
    df.to_csv(tmp, index=False)
    os.replace(tmp, results_file)
    if remove:
        for s in shards:
            os.remove(s)
    return df


def verify_flock_scope(probe_dir: str) -> Dict:
    """Report what `fcntl.flock` actually does on the filesystem holding `probe_dir`.

    Same-process re-entry cannot detect a node-local lock, so this only proves the lock works at
    all; the cross-NODE question can only be answered by two processes on two nodes (run this from
    two jobs against a shared path and compare) or from the mount options. It reports the
    filesystem type so the answer is at least attributable.
    """
    import subprocess
    os.makedirs(probe_dir, exist_ok=True)
    p = os.path.join(probe_dir, ".flock_probe")
    out = {"path": os.path.abspath(p), "host": socket.gethostname()}
    try:
        out["fstype"] = subprocess.run(["stat", "-f", "-c", "%T", probe_dir],
                                       capture_output=True, text=True).stdout.strip()
    except Exception as e:                                    # noqa: BLE001
        out["fstype"] = f"unknown ({e})"
    try:
        lock = FileLock(p + ".lock", timeout=5)
        with lock:
            out["acquired"] = True
            inner = FileLock(p + ".lock", timeout=1)
            try:
                with inner:
                    out["reentrant_blocked"] = False
            except Timeout:
                out["reentrant_blocked"] = True
    except Exception as e:                                    # noqa: BLE001
        out["acquired"] = False
        out["error"] = str(e)
    out["safe_for_multinode"] = out.get("fstype") in ("ext2/ext3", "xfs", "tmpfs", "overlayfs")
    out["note"] = ("fstype alone does not settle Lustre/NFS: check the mount options for `flock` "
                   "vs `localflock`, or use shard_path() and skip locking entirely.")
    return out
