"""Stress the results-CSV writer at camera-ready sweep concurrency.

The claim under test is the one a 100s-of-tasks sweep depends on: N processes writing to ONE csv at
once lose no rows and corrupt nothing. "It uses a FileLock" is not evidence -- a lock that is taken
per-node, a temp file that collides, or an upsert that drops a key all fail silently and produce a
SHORT table rather than an error.

    python src/test_results_csv_concurrency.py --n 300
    python src/test_results_csv_concurrency.py --n 300 --mode shard   # the lock-free path

Three phases:
  INSERT   N processes, N distinct keys, launched together -> expect exactly N rows.
  UPSERT   the same N processes re-run with the same keys   -> expect exactly N rows, values updated.
  CONTEND  N processes onto only N/10 distinct keys         -> expect exactly N/10 rows, no tearing.

⚠ WHAT THIS CANNOT TEST. Every process here is on ONE HOST, so it exercises `fcntl.flock` within a
node. A Lustre filesystem mounted `-o localflock` gives node-local locks and would pass this test
while losing rows across a multi-node SLURM array. See `results_csv`'s docstring; the `shard` mode
is the answer that needs no lock at all.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))

COLUMNS = ["method", "seed", "task", "value", "host_pid", "phase"]
COMB = ["method", "seed", "task"]

WORKER = r'''
import os, sys, time
sys.path.insert(0, {here!r})
from results_csv import write_result_row, shard_path
idx, start_at, csv_path, phase, mode, nkeys = sys.argv[1:7]
idx, start_at, nkeys = int(idx), float(start_at), int(nkeys)
row = {{"method": f"m{{idx % nkeys}}", "seed": 41, "task": f"t{{idx % nkeys}}",
        "value": idx, "host_pid": os.getpid(), "phase": phase}}
# Spin to a common wall-clock instant so the writers actually collide.
while time.time() < start_at:
    time.sleep(0.002)
if mode == "shard":
    target = shard_path(csv_path, str(idx))
else:
    target = csv_path
ok = write_result_row(target, {cols!r}, {comb!r}, row)
sys.exit(0 if ok else 1)
'''.format(here=HERE, cols=COLUMNS, comb=COMB)


def launch(n, csv_path, phase, mode, nkeys, lead=6.0):
    src = os.path.join(HERE, "_ccworker.py")
    with open(src, "w") as f:
        f.write(WORKER)
    start_at = time.time() + lead
    procs = [subprocess.Popen([sys.executable, src, str(i), str(start_at), csv_path, phase,
                               mode, str(nkeys)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
             for i in range(n)]
    t0 = time.time()
    rcs, errs = [], []
    for p in procs:
        _, err = p.communicate()
        rcs.append(p.returncode)
        if p.returncode != 0 and err:
            errs.append(err.decode()[-300:])
    os.remove(src)
    return rcs, errs, time.time() - t0 - lead


def check(csv_path, expect_rows, phase, mode, columns, comb, expect_phase_rows=None):
    import pandas as pd
    from results_csv import merge_shards
    if mode == "shard":
        merge_shards(csv_path, columns, comb, remove=True)
    problems = []
    if not os.path.exists(csv_path):
        return [f"no CSV at {csv_path}"], None
    df = pd.read_csv(csv_path)
    if len(df) != expect_rows:
        problems.append(f"row count {len(df)} != expected {expect_rows}  <-- LOST OR DUPLICATED ROWS")
    # ⚠ THE PHASE-SCOPED COUNT IS THE REAL ASSERTION. The CSV is deliberately NOT cleared between
    # phases, so CONTEND's N writers hitting K keys must leave exactly K rows STAMPED CONTEND while
    # the untouched INSERT rows survive. Asserting on the total instead just re-measures the
    # earlier phases (and my first version of this test did exactly that and cried wolf).
    if expect_phase_rows is not None:
        got = int((df["phase"] == phase).sum())
        if got != expect_phase_rows:
            problems.append(f"{got} rows stamped {phase} != expected {expect_phase_rows} "
                            f"<-- writers onto shared keys lost or duplicated")
    dups = df.duplicated(subset=comb).sum()
    if dups:
        problems.append(f"{dups} duplicate key(s) -- the upsert did not dedupe")
    if list(df.columns) != columns:
        problems.append(f"columns drifted: {list(df.columns)}")
    if df.isna().all(axis=1).any():
        problems.append("all-NaN row present -- a torn write")
    leftovers = [f for f in os.listdir(os.path.dirname(csv_path) or ".") if ".tmp." in f]
    if leftovers:
        problems.append(f"leftover temp files: {leftovers[:5]}")
    return problems, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--mode", default="lock", choices=["lock", "shard"])
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()

    d = args.dir or os.path.join(os.path.dirname(HERE), "results", "_cctest")
    os.makedirs(d, exist_ok=True)
    csv_path = os.path.join(d, "cc.csv")
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))

    from results_csv import verify_flock_scope
    print("flock probe:", verify_flock_scope(d))
    print()

    failed = 0
    K = max(args.n // 10, 1)
    for phase, n, nkeys, expect, expect_phase in [
            ("INSERT",  args.n, args.n, args.n, args.n),
            ("UPSERT",  args.n, args.n, args.n, args.n),
            # N writers onto K keys: K rows carry this phase, the other N-K INSERT rows survive.
            ("CONTEND", args.n, K,      args.n, K)]:
        rcs, errs, secs = launch(n, csv_path, phase, args.mode, nkeys)
        bad_rc = sum(1 for r in rcs if r != 0)
        problems, df = check(csv_path, expect, phase, args.mode, COLUMNS, COMB,
                             expect_phase_rows=expect_phase)
        ok = not problems and not bad_rc
        stamped = 0 if df is None else int((df["phase"] == phase).sum())
        print(f"{'PASS' if ok else 'FAIL'}  {phase:<8} n={n:<4} keys={nkeys:<4} "
              f"rows={0 if df is None else len(df):<5} stamped={stamped:<5} "
              f"nonzero_exit={bad_rc:<3} {secs:6.1f}s")
        for p in problems:
            print(f"        - {p}")
        for e in errs[:2]:
            print(f"        - worker stderr: {e.strip()[-200:]}")
        failed += 0 if ok else 1
    print()
    print("ALL PASS" if not failed else f"{failed} PHASE(S) FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
