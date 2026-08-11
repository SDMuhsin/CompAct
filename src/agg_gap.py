"""Aggregate the 2026-08-03 throughput-gap sweep into the tables for `holes_closed.md`.

Reads `results/hyclora/gap/*.json` and prints:
  * the per-shape table (peak allocated / peak reserved / ms per step) for every arm, with the
    reserved column taken from the DEDICATED SINGLE-ARM process wherever one exists (protocol:
    `max_memory_reserved` from a multi-arm process is a caching-allocator artifact -- see
    `holes_closed.md` section 3.5);
  * the rep-to-rep spread of every row (the contamination canary);
  * the deltas of every fused arm against every competitor, both halves at every shape;
  * whether the 5% throughput budget is met at every shape, and what memory lead remains.
"""
import glob
import json
import os
import sys

GAP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "hyclora", "gap")

ORDER = ["gc_manual_sdpa", "fb_min_fnorm_sdpa", "fb_attn_fnorm_sdpa", "fb_auto_fnorm_sdpa",
         "hyclora_flash_nc", "unsloth_gc", "unsloth_offload", "baseline_sdpa"]
SEQS = [1024, 2048, 4096, 8192, 16384]


def load():
    rows, reps, solo, ctrl = {}, {}, {}, {}
    for p in sorted(glob.glob(os.path.join(GAP, "*.json"))):
        try:
            blob = json.load(open(p))
        except Exception:
            continue
        if "rows" not in blob:
            continue
        base = os.path.basename(p)
        is_solo = base.startswith("solo_")
        is_ctrl = base.startswith("control")
        for arm, r in blob["rows"].items():
            if "error" in r:
                continue
            key = (r["seq"], arm)
            if is_ctrl:
                ctrl.setdefault(arm, []).append((base, r))
            elif is_solo:
                solo[key] = r
            else:
                rows[key] = r
        if not (is_solo or is_ctrl):
            for r in blob.get("all_reps", []):
                if "error" not in r:
                    reps.setdefault((r["seq"], r["arm"]), []).append(
                        (r["ms_per_step_median"], r["train_step_peak_alloc_mib"]))
    return rows, reps, solo, ctrl


def main():
    rows, reps, solo, ctrl = load()

    print("=" * 132)
    print("CONTROLS -- arms that touch none of the code under test (seq 1024, regime B)")
    print("=" * 132)
    print(f"{'arm':<22}{'file':<22}{'peak alloc':>12}{'ms/step':>10}"
          f"   established: baseline 6156.39/231.27, gc_manual 2593.80/318.68")
    for arm, lst in sorted(ctrl.items()):
        for base, r in lst:
            print(f"{arm:<22}{base:<22}{r['train_step_peak_alloc_mib']:>12.2f}"
                  f"{r['ms_per_step_median']:>10.2f}")

    print("\n" + "=" * 132)
    print("THE SWEEP -- regime B (matched CE), batch 2.  reserved from a DEDICATED single-arm "
          "process where available (*)")
    print("=" * 132)
    print(f"{'arm':<22}{'seq':>7}{'alloc':>11}{'resv':>9}{'floor':>9}{'pk-floor':>10}"
          f"{'ms/step':>10}{'IQR':>7}{'reps ms / dpeak':>24}{'keep':>7}")
    for arm in ORDER:
        for s in SEQS:
            r = rows.get((s, arm)) or solo.get((s, arm))
            if r is None:
                continue
            sr = solo.get((s, arm))
            resv = f"{sr['train_step_peak_reserved_mib']:.0f}*" if sr else \
                f"{r['train_step_peak_reserved_mib']:.0f}"
            rp = reps.get((s, arm), [])
            spread = ""
            if len(rp) > 1:
                ms = [x[0] for x in rp]
                pk = [x[1] for x in rp]
                spread = ("/".join(f"{m:.1f}" for m in ms)
                          + f" d{(max(ms) - min(ms)) / min(ms) * 100:.1f}% p{max(pk) - min(pk):.1f}")
            pol = (r.get("fb_policy") or {}).get("resolved_by_seq") or {}
            keep = list(pol.values())[0] if pol else ""
            print(f"{arm:<22}{s:>7}{r['train_step_peak_alloc_mib']:>11.2f}{resv:>9}"
                  f"{r['resident_floor_mib']:>9.2f}{r['peak_minus_floor_mib']:>10.2f}"
                  f"{r['ms_per_step_median']:>10.2f}{r['ms_per_step_iqr']:>7.2f}"
                  f"{spread:>24}{keep:>7}")
            if sr and abs(sr["train_step_peak_alloc_mib"] - r["train_step_peak_alloc_mib"]) > 0.01:
                print(f"{'  ^ solo alloc differs':<22}{s:>7}"
                      f"{sr['train_step_peak_alloc_mib']:>11.2f}")

    for base in ("fb_attn_fnorm_sdpa", "fb_min_fnorm_sdpa"):
        print("\n" + "=" * 132)
        print(f"DELTAS: {base} vs each competitor   (negative = we are lighter / faster)")
        print("=" * 132)
        print(f"{'competitor':<22}{'seq':>7}{'d mem %':>10}{'d resv %':>10}{'d time %':>10}"
              f"{'our MiB':>10}{'their MiB':>11}{'our ms':>9}{'their ms':>10}{'budget':>9}")
        for c in ORDER:
            if c == base or c.startswith("fb_"):
                continue
            for s in SEQS:
                a = rows.get((s, base)) or solo.get((s, base))
                b = rows.get((s, c)) or solo.get((s, c))
                if not a or not b:
                    continue
                asolo, bsolo = solo.get((s, base)), solo.get((s, c))
                dm = (a["train_step_peak_alloc_mib"] / b["train_step_peak_alloc_mib"] - 1) * 100
                dt = (a["ms_per_step_median"] / b["ms_per_step_median"] - 1) * 100
                dr = float("nan")
                if asolo and bsolo:
                    dr = (asolo["train_step_peak_reserved_mib"]
                          / bsolo["train_step_peak_reserved_mib"] - 1) * 100
                ok = "OK" if dt <= 5.0 else "OVER"
                print(f"{c:<22}{s:>7}{dm:>10.2f}{dr:>10.2f}{dt:>10.2f}"
                      f"{a['train_step_peak_alloc_mib']:>10.1f}"
                      f"{b['train_step_peak_alloc_mib']:>11.1f}"
                      f"{a['ms_per_step_median']:>9.1f}{b['ms_per_step_median']:>10.1f}{ok:>9}")

    # ---- the policy check: fb_auto must equal the pinned arm it resolves to, to the byte ----
    print("\n" + "=" * 132)
    print("POLICY CHECK -- fb_auto_fnorm_sdpa vs the pinned arm it resolved to")
    print("=" * 132)
    for s in SEQS:
        a = rows.get((s, "fb_auto_fnorm_sdpa"))
        if not a:
            continue
        pol = (a.get("fb_policy") or {}).get("resolved_by_seq") or {}
        lvl = list(pol.values())[0] if pol else "?"
        pin = rows.get((s, f"fb_{lvl}_fnorm_sdpa")) or solo.get((s, f"fb_{lvl}_fnorm_sdpa"))
        d = (a["train_step_peak_alloc_mib"] - pin["train_step_peak_alloc_mib"]) if pin else None
        print(f"  seq {s:>6}: auto -> {lvl:<5} alloc {a['train_step_peak_alloc_mib']:.2f} "
              f"vs pinned {pin['train_step_peak_alloc_mib']:.2f} "
              f"(delta {d:+.2f} MiB)" if pin else f"  seq {s}: auto -> {lvl}, no pinned row")
        fbc = (a.get("fb_policy") or {}).get("counters") or {}
        print(f"           flash_recompute={fbc.get('flash_recompute')} "
              f"levels={(a.get('fb_policy') or {}).get('forwards_by_level')}")

    # ---- activation-part ratio: both arms O(S)? ----
    print("\n" + "=" * 132)
    print("ACTIVATION PART (peak - floor) and the ratio against hyclora_flash_nc")
    print("=" * 132)
    print(f"{'seq':>7}{'fb_attn':>12}{'fb_min':>12}{'hy_nc':>12}{'attn ratio':>12}{'min ratio':>12}")
    for s in SEQS:
        g = {}
        for arm in ("fb_attn_fnorm_sdpa", "fb_min_fnorm_sdpa", "hyclora_flash_nc"):
            r = rows.get((s, arm)) or solo.get((s, arm))
            g[arm] = r["peak_minus_floor_mib"] if r else None
        if not all(g.values()):
            continue
        print(f"{s:>7}{g['fb_attn_fnorm_sdpa']:>12.2f}{g['fb_min_fnorm_sdpa']:>12.2f}"
              f"{g['hyclora_flash_nc']:>12.2f}"
              f"{g['hyclora_flash_nc'] / g['fb_attn_fnorm_sdpa']:>12.3f}"
              f"{g['hyclora_flash_nc'] / g['fb_min_fnorm_sdpa']:>12.3f}")


if __name__ == "__main__":
    sys.exit(main())
