"""Aggregate every arm x sequence-length row under `results/hyclora/frontier/` into ONE table.

Why this exists: the CONTEXT.md section 33.4 headline table was transcribed by hand, and an arm
that had been measured -- `liger_gc_sdpa`, lighter than every competitor in the must-beat set --
was simply absent from it for a day. A table built from disk cannot lose a row that way.

Reads every JSON in the frontier directory, keys rows by (arm, seq, regime), keeps the best row per
key (lowest peak; among equal peaks the fastest), and prints:

  * peak allocated MiB per arm per sequence length, per CE regime;
  * stored state per layer, so arms with the SAME checkpointing granularity can be compared
    directly -- that pairing is what separates a stored-state win from a transient-window win;
  * GPU kernel time per step (protocol A.3.6), which is the throughput figure that survives a
    co-tenant, alongside wall clock, which on this box usually does not;
  * every row's co-tenancy state at measurement time, because an OOM recorded while another tenant
    held 21 GB is NOT a card-level OOM and must never be reported as one.

Usage:
  PYTHONPATH=src python src/agg_baselines.py [--dir results/hyclora/frontier] [--regime B]
"""
import argparse
import glob
import json
import os

FRONTIER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "hyclora", "frontier")
SEQS = [1024, 2048, 4096, 8192, 16384]


def regime_of(row):
    """B = matched fused CE. Every non-unsloth arm needs `--flce` to be in it; unsloth and the
    Liger arms ship their own fused CE, so for them the flag is absent and the regime is still B.
    """
    arm = row.get("arm", "")
    if arm.startswith("unsloth") or arm.startswith("liger"):
        return "B"
    return "B" if row.get("flce") else "A"


def collect(dirs, include_diag=False):
    rows = {}
    paths = []
    for d in dirs:
        paths += sorted(glob.glob(os.path.join(d, "*.json")))
    for path in paths:
        try:
            blob = json.load(open(path))
        except Exception:
            continue
        if not isinstance(blob, dict):
            continue
        rr = blob.get("rows")
        items = list(rr.values()) if isinstance(rr, dict) else (rr if isinstance(rr, list) else [])
        for r in items:
            if not isinstance(r, dict) or not r.get("arm"):
                continue
            seq = r.get("seq") or (blob.get("cfg") or {}).get("seq")
            # `--force_logits` and `--no_head` produce rows that are NOT comparable with the
            # others and must never be pooled with them.  This aggregator did pool them on its
            # first run: `unsloth_seq1024_logits_bf16ad.json` is a `--force_logits` DIAGNOSTIC
            # (it makes unsloth materialise logits instead of taking its fused CE) and it came
            # out at 2630.20 MiB, LOWER than the 2655.20 comparable row, so "keep the lowest"
            # silently promoted a diagnostic into the headline table.  That the forced-logits
            # path is the lighter one is itself the G1 finding: unsloth's peak at this shape is
            # set inside its fused-CE kernel, not by the logits tensor.
            if not include_diag and (r.get("force_logits") or r.get("headless")):
                continue
            key = (r["arm"], seq, regime_of(r))
            r = dict(r, _src=os.path.basename(path))
            if "error" in r:
                rows.setdefault(key, r)          # keep an error row only if nothing better exists
                continue
            old = rows.get(key)
            if (old is None or "error" in old
                    or r["train_step_peak_alloc_mib"] < old["train_step_peak_alloc_mib"]):
                rows[key] = r
    return rows


def fmt(v, w=9, p=2):
    return " " * w if v is None else f"{v:{w}.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=[FRONTIER], nargs="+",
                    help="one or more result directories; competitor rows live in "
                         "results/hyclora/holes and results/hyclora/profile too")
    ap.add_argument("--regime", default="B", choices=["A", "B", "both"])
    ap.add_argument("--include_diag", action="store_true",
                    help="also pool --force_logits / --no_head rows. They are DIAGNOSTICS, "
                         "not comparable arms; off by default for that reason.")
    ap.add_argument("--ref", default="fb_min_fnorm_sdpa",
                    help="arm the delta columns are taken against")
    args = ap.parse_args()

    rows = collect(args.dir, include_diag=args.include_diag)
    regimes = ["A", "B"] if args.regime == "both" else [args.regime]

    for reg in regimes:
        arms = sorted({a for (a, _s, g) in rows if g == reg})
        if not arms:
            continue
        print(f"\n{'=' * 118}\nREGIME {reg}  --  peak allocated MiB (batch 2)\n{'=' * 118}")
        print(f"{'arm':<26}" + "".join(f"{s:>11}" for s in SEQS) + "   stored MiB/layer")
        for a in arms:
            line = f"{a:<26}"
            stored = []
            for s in SEQS:
                r = rows.get((a, s, reg))
                if r is None:
                    line += f"{'-':>11}"
                elif "error" in r:
                    line += f"{'OOM/err':>11}"
                else:
                    line += fmt(r["train_step_peak_alloc_mib"], 11)
                    st = (r.get("retained") or {}).get("retained_MiB_per_layer_median")
                    if st is not None:
                        stored.append(f"{st:g}")
            print(line + "   " + "/".join(stored))

        ref = args.ref
        if any((ref, s, reg) in rows for s in SEQS):
            print(f"\n  delta vs {ref}  (positive = the other arm is HEAVIER; % of the other arm)")
            for a in arms:
                if a == ref:
                    continue
                cells = []
                for s in SEQS:
                    r, b = rows.get((a, s, reg)), rows.get((ref, s, reg))
                    if not r or not b or "error" in r or "error" in b:
                        cells.append(f"{'-':>13}")
                        continue
                    d = r["train_step_peak_alloc_mib"] - b["train_step_peak_alloc_mib"]
                    cells.append(f"{d:+8.2f}/{100 * d / r['train_step_peak_alloc_mib']:+5.2f}%")
                print(f"  {a:<24}" + "".join(cells))

        print(f"\n  throughput -- GPU kernel time ms/step (protocol A.3.6 primary on a loud box);"
              f"\n                wall clock in brackets, VOID whenever a co-tenant was present")
        for a in arms:
            line = f"  {a:<24}"
            for s in SEQS:
                r = rows.get((a, s, reg))
                if not r or "error" in r:
                    line += f"{'-':>19}"
                    continue
                kt = (r.get("kernel_time") or {}).get("kernel_us_per_step")
                wc = r.get("ms_per_step_median")
                line += f"{(kt / 1e3 if kt else float('nan')):8.2f}[{wc:8.1f}]"
            print(line)

        # `_gpu_state()` returns one CSV line per device:
        # "index, clocks.sm, temperature, power.draw, memory.used".  The last field is what says
        # whether an OOM row is a card-level limit or a co-tenant holding 21 GB.
        print("\n  device memory in use by ALL processes at measurement time, MiB "
              "(an OOM under a heavy co-tenant is NOT a card-level OOM)")
        for a in arms:
            st = []
            for s in SEQS:
                r = rows.get((a, s, reg))
                g = (r or {}).get("gpu_state_before") or []
                used = "?"
                if isinstance(g, list) and g:
                    try:
                        used = str(int(float(str(g[0]).split(",")[-1])))
                    except (ValueError, IndexError):
                        used = "?"
                st.append(f"{s}:{used}")
            print(f"  {a:<24}" + "  ".join(st))


if __name__ == "__main__":
    main()
