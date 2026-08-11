#!/usr/bin/env python
"""Combine a paired cell across INDEPENDENT timing windows -- CONTEXT.md 37.5 item 6.

WHY THIS EXISTS.  `agg_timing_window.py` adjudicates ONE window: it asks whether the reps inside
that window resolve a comparison.  That is the wrong question to ask of a CLAIM, because the reps
inside a window are serially correlated -- one process, one clock ramp, one allocator state -- so
the within-window interval is systematically narrower than the windows actually reproduce.

This project's own disk proves it.  seq 4096, `fb_min` vs `hyclora_flash_nc`, four surviving
non-void windows: +0.31 / +0.45 / +0.49 / +0.60.  Between-window scatter ~0.3pp.  The tightest
within-window interval on that cell (8 reps, 2026-08-09) is +-0.06pp -- five times narrower than
the spread the windows themselves show.  seq 1024 is the same story at ~0.5pp (-2.03 against -1.53,
intervals overlapping but point estimates far apart), and seq 2048 is why the cell has been argued
about for three sessions (-0.56 against +0.29).

So the unit of evidence for a claim is the WINDOW, not the rep.  This script treats each window as
one observation and puts a Student-t interval on the mean of the per-window estimates.  That
interval absorbs the between-window systematic by construction, which is exactly the term the
within-window interval omits.

Operative rule.  CONTEXT.md 37.1 clause 1 requires a STRICT improvement at every sequence length,
with no magnitude floor (46.10: "do not discard a mechanism for being small").  A mechanism worth
less than the between-window scatter therefore CANNOT be certified from a single window however
many reps it carries.  Run independent windows -- different processes, ideally different hours --
and certify here.

Usage:
  PYTHONPATH=src python src/agg_paired_windows.py
  PYTHONPATH=src python src/agg_paired_windows.py --ours fb_min_fnorm_sdpa --comp hyclora_flash_nc
"""
import argparse
import glob
import json
import os
import statistics

from agg_timing_window import paired_ci, _T95, CONTROLS, drift, HARD_VOID

BASE_DIR = "results/hyclora/timing"


def default_dirs():
    """The live window directory plus every archive under it.

    Discovered rather than listed, because the runner overwrites `window_main_seq<S>.json` in place
    on every firing: an independent window survives only if somebody copies it into an archive
    first, and a hard-coded list silently drops any archive a later session adds.  Duplicates are
    filtered downstream on the ratio vector, so over-collecting here is free.
    """
    out = [BASE_DIR]
    out += sorted(p for p in glob.glob(os.path.join(BASE_DIR, "*")) if os.path.isdir(p))
    return out


def ratios(path, ours, comp):
    """Per-rep paired ratios (%) for one window file, paired by rep index within the file.

    Also returns the window's worst control drift, because a window whose control failed must not
    enter a cross-window combination.  A paired ratio cancels a drift COMMON to the window; it does
    NOT cancel a STEP change -- a tenant arriving mid-window moves the two arms by different amounts
    at different reps, which is exactly what the control exists to catch (see the note in
    `agg_timing_window.main`, and CONTEXT.md 43.11).
    """
    with open(path) as fh:
        d = json.load(fh)
    reps = d.get("all_reps", [])
    o = {r["rep"]: r["ms_per_step_median"] for r in reps
         if r["arm"] == ours and "error" not in r}
    c = {r["rep"]: r["ms_per_step_median"] for r in reps
         if r["arm"] == comp and "error" not in r}
    worst = 0.0
    for ctrl in CONTROLS:
        t = [r["ms_per_step_median"] for r in reps if r["arm"] == ctrl and "error" not in r]
        if len(t) >= 2:
            worst = max(worst, drift(t))
    shared = sorted(set(o) & set(c))
    return int(d["cfg"]["seq"]), [100.0 * (o[k] / c[k] - 1.0) for k in shared], worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=None)
    ap.add_argument("--ours", default="fb_min_fnorm_sdpa")
    ap.add_argument("--comp", default="hyclora_flash_nc")
    ap.add_argument("--min_reps", type=int, default=3)
    args = ap.parse_args()
    dirs = args.dirs if args.dirs else default_dirs()

    # seq -> list of (label, n, median, ci)
    by_seq, skipped, seen = {}, [], set()
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "window_main_seq*.json"))):
            if "VOID" in os.path.basename(path):
                skipped.append((os.path.relpath(path), "filename marks it VOID"))
                continue
            try:
                seq, pr, ctrl_drift = ratios(path, args.ours, args.comp)
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                skipped.append((os.path.relpath(path), f"unreadable: {exc}"))
                continue
            if ctrl_drift > HARD_VOID:
                skipped.append((os.path.relpath(path),
                                f"VOID: control drift {ctrl_drift:.2f}% > {HARD_VOID:.0f}%"))
                continue
            if len(pr) < args.min_reps:
                skipped.append((os.path.relpath(path),
                                f"only {len(pr)} shared reps for this pair"))
                continue
            # The same window is present under more than one directory (archived copies keep
            # their mtime).  Dedupe on the ratio vector itself: two files that produce identical
            # per-rep ratios are one measurement and must not be counted twice.
            key = (seq, tuple(round(x, 9) for x in pr))
            if key in seen:
                skipped.append((os.path.relpath(path), "duplicate of an earlier window"))
                continue
            seen.add(key)
            by_seq.setdefault(seq, []).append(
                (os.path.relpath(path), len(pr), statistics.median(pr), paired_ci(pr)))

    print(f"paired cell: {args.ours}  vs  {args.comp}      (negative = ours is faster)")
    for seq in sorted(by_seq):
        wins = by_seq[seq]
        print("=" * 100)
        print(f"seq {seq}  --  {len(wins)} independent window(s)")
        for label, n, med, ci in sorted(wins, key=lambda w: os.path.getmtime(w[0])):
            span = f"[{ci['lo']:+.2f},{ci['hi']:+.2f}]" if ci else "n/a"
            stamp = os.popen(f"date -u -r '{label}' +%Y-%m-%dT%H:%MZ").read().strip()
            print(f"    {med:+7.2f}%  within-window 95% {span:>16}  n={n:<3} {stamp}  {label}")

        est = [w[2] for w in wins]
        if len(est) < 2:
            print("    -> ONE window only.  No between-window term can be formed, so the "
                  "within-window interval above is the ONLY evidence and it is a lower bound "
                  "on the uncertainty.  Run a second, independent window before quoting this.")
            continue
        mean = statistics.fmean(est)
        sd = statistics.stdev(est)
        half = _T95.get(len(est) - 1, 1.960) * sd / (len(est) ** 0.5)
        lo, hi = mean - half, mean + half
        resolved = lo * hi > 0
        print(f"    -> ACROSS WINDOWS  {mean:+.2f}%  95% CI [{lo:+.2f},{hi:+.2f}]pp   "
              f"(W={len(est)}, between-window sd {sd:.2f}pp)")
        print(f"       verdict: {'RESOLVED' if resolved else 'NOT RESOLVED -- the interval spans zero'}"
              f"; smallest effect this many windows could certify is ~{half:.2f}pp")
        if len(est) == 2:
            # t(1) = 12.706.  Two windows can estimate a between-window sd but cannot bound it,
            # so the interval is structurally uninformative however tight the two windows agree.
            # Say so rather than let the width read as evidence of a noisy measurement.
            print("       (W=2 is structurally uninformative -- t(1)=12.7 -- regardless of how "
                  "well the two agree.  W>=3 is the minimum that can certify anything.)")
        # Windows are combined UNWEIGHTED even when they carry different rep counts.  A
        # precision-weighted combiner would weight by the within-window variance, which is exactly
        # the term shown above to understate the truth; unweighted keeps each window one vote.

    if skipped:
        print("=" * 100)
        print("windows not counted:")
        for path, why in skipped:
            print(f"    {path}  --  {why}")


if __name__ == "__main__":
    main()
