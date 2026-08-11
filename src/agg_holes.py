"""Aggregate the HOLE-1 / HOLE-2 sweep JSONs into the tables that go into
`llmdocs/trackers/holes_closed.md`.

Reads every `results/hyclora/holes/*.json`, keys rows by (regime, seq, arm), and prints:
  * the peak-allocated / peak-reserved / ms-per-step table per CE regime;
  * the rep-to-rep spread of every row (the contamination canary);
  * the fb_min_fnorm-vs-competitor deltas and the interpolated crossover.
"""
import glob
import json
import math
import os
import sys

HOLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "results", "hyclora", "holes")

REGIME = {"sweepA_stockce": "A(stockCE)", "sweepB_flce": "B(FLCE)",
          "sweepC_unsloth": "B(FLCE)", "hole2_matchedCE": "B(FLCE)",
          "control": "ctrl", "compile_gc": "ctrl"}


def regime_of(path):
    b = os.path.basename(path)
    for k, v in REGIME.items():
        if b.startswith(k):
            return v
    return "?"


def load():
    rows = {}          # (regime, seq, arm) -> dict
    reps = {}          # (regime, seq, arm) -> [ms,...]
    for p in sorted(glob.glob(os.path.join(HOLES, "*.json"))):
        try:
            blob = json.load(open(p))
        except Exception:
            continue
        if "rows" not in blob:
            continue
        reg = regime_of(p)
        for arm, r in blob["rows"].items():
            if "error" in r:
                rows.setdefault((reg, blob["cfg"]["seq"], arm), r)
                continue
            rows[(reg, r["seq"], arm)] = r
        for r in blob.get("all_reps", []):
            if "error" in r:
                continue
            reps.setdefault((reg, r["seq"], r["arm"]), []).append(
                (r["ms_per_step_median"], r["train_step_peak_alloc_mib"]))
    return rows, reps


def main():
    rows, reps = load()
    seqs = sorted({k[1] for k in rows})
    for reg in ("A(stockCE)", "B(FLCE)", "ctrl"):
        keys = [k for k in rows if k[0] == reg]
        if not keys:
            continue
        arms = sorted({k[2] for k in keys})
        print(f"\n{'=' * 128}\nREGIME {reg}\n{'=' * 128}")
        hdr = f"{'arm':<26}{'seq':>7}{'peak alloc':>12}{'peak resv':>11}{'floor':>10}{'pk-floor':>11}{'ms/step':>10}{'IQR':>8}{'reps(ms)':>26}"
        print(hdr)
        for arm in arms:
            for s in seqs:
                r = rows.get((reg, s, arm))
                if r is None:
                    continue
                if "error" in r:
                    print(f"{arm:<26}{s:>7}   {r['error'][:70]}")
                    continue
                rp = reps.get((reg, s, arm), [])
                spread = ""
                if len(rp) > 1:
                    ms = [x[0] for x in rp]
                    pk = [x[1] for x in rp]
                    spread = "/".join(f"{m:.1f}" for m in ms)
                    dm = (max(ms) - min(ms)) / min(ms) * 100
                    dp = max(pk) - min(pk)
                    spread += f" d{dm:.1f}% p{dp:.1f}"
                print(f"{arm:<26}{s:>7}{r['train_step_peak_alloc_mib']:>12.2f}"
                      f"{r['train_step_peak_reserved_mib']:>11.2f}{r['resident_floor_mib']:>10.2f}"
                      f"{r['peak_minus_floor_mib']:>11.2f}{r['ms_per_step_median']:>10.2f}"
                      f"{r['ms_per_step_iqr']:>8.2f}{spread:>26}")

    # ------------------------------------------------------------------ deltas + crossover
    for reg in ("A(stockCE)", "B(FLCE)"):
        base = "fb_min_fnorm_sdpa"
        comps = sorted({k[2] for k in rows if k[0] == reg and k[2] != base})
        if not comps:
            continue
        print(f"\n{'=' * 128}\nDELTAS vs {base}  --  {reg}   (negative = we are lighter / faster)\n{'=' * 128}")
        print(f"{'competitor':<26}{'seq':>7}{'d mem %':>10}{'d time %':>10}{'our MiB':>10}{'their MiB':>11}{'our ms':>9}{'their ms':>10}")
        for c in comps:
            pts = []
            for s in seqs:
                a = rows.get((reg, s, base))
                b = rows.get((reg, s, c))
                if not a or not b or "error" in a or "error" in b:
                    continue
                dm = (a["train_step_peak_alloc_mib"] / b["train_step_peak_alloc_mib"] - 1) * 100
                dt = (a["ms_per_step_median"] / b["ms_per_step_median"] - 1) * 100
                pts.append((s, dt))
                print(f"{c:<26}{s:>7}{dm:>10.2f}{dt:>10.2f}"
                      f"{a['train_step_peak_alloc_mib']:>10.1f}{b['train_step_peak_alloc_mib']:>11.1f}"
                      f"{a['ms_per_step_median']:>9.1f}{b['ms_per_step_median']:>10.1f}")
            # log-linear interpolation of the throughput crossover (d time % == 0)
            for (s0, d0), (s1, d1) in zip(pts, pts[1:]):
                if d0 * d1 < 0:
                    t = -d0 / (d1 - d0)
                    xs = math.exp(math.log(s0) + t * (math.log(s1) - math.log(s0)))
                    print(f"    -> throughput crossover vs {c} at seq ~= {xs:.0f}")
            if pts and all(d > 0 for _s, d in pts):
                print(f"    -> {c} is faster at EVERY measured seq (no crossover in range)")
            if pts and all(d < 0 for _s, d in pts):
                print(f"    -> we are faster at EVERY measured seq (no crossover in range)")


if __name__ == "__main__":
    sys.exit(main())
