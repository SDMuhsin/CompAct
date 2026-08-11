#!/usr/bin/env python
"""Adjudicate a timing window (G12) -- CONTEXT.md 34.9 item 1, protocol A.3.4.

The protocol says a timing sweep carries a control arm at start / middle / end and the window is
DISCARDED if the control moves more than ~1%.  That rule has never been mechanised, and the 2026-08-04
first attempt shows why it must be: its control read 1000.56 / 449.08 / 1005.15 ms across three reps
-- a 123% swing -- while every arm still produced a plausible-looking number, and the harness's own
`rows[arm]` (fastest rep wins, per "never detune") would have published `fb_attn` at 377.16 ms and
`hyclora_flash_nc` at 387.34 ms as if that were a result.

So this script refuses to print a comparison at all until the control passes.  It reads the raw
`all_reps` -- never `rows` -- because the fastest-rep rule is a reporting convention for a VALID
window and is meaningless in a contaminated one.

Usage:  PYTHONPATH=src python src/agg_timing_window.py [--dir results/hyclora/timing] [--tol 1.0]
"""
import argparse
import glob
import json
import os
import statistics

# Arms that ride along solely to prove the window was clean.  They touch none of the code under
# test: `baseline_sdpa` is stock HF + PEFT with nothing patched, and above 4096 it does not fit, so
# the window falls back to `gc_manual_sdpa` (stock HF checkpointing -- still none of our block, and
# none of HyC-LoRA's).
CONTROLS = ("baseline_sdpa", "gc_manual_sdpa")

# Above this, the control did not "drift" -- something else was running on the card, and nothing
# from the window is usable, paired or not.  Between `--tol` and this, the window is reported as
# DRIFTING: raw medians are not quotable but the paired per-rep ratios below may still resolve a
# comparison, and they carry their own scatter so the reader can see whether they do.
HARD_VOID = 5.0

# The mandate (33.2, amended by 34.3): one configuration beating all three simultaneously.
MUST_BEAT = ("hyclora_flash_nc", "unsloth_gc", "liger_gc_sdpa")
OURS = ("fb_min_fnorm_sdpa", "fb_attn_fnorm_sdpa", "fb_auto_fnorm_sdpa")


def load(directory):
    """seq -> arm -> list of per-rep rows, pooled across the main/unsloth/liger processes."""
    by_seq = {}
    for path in sorted(glob.glob(os.path.join(directory, "window_*_seq*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        seq = int(d["cfg"]["seq"])
        for r in d.get("all_reps", []):
            r["_file"] = os.path.basename(path)
            by_seq.setdefault(seq, {}).setdefault(r["arm"], []).append(r)
    return by_seq


def drift(times):
    """Peak-to-peak spread as a fraction of the fastest rep.  The protocol's ~1% test."""
    return 100.0 * (max(times) - min(times)) / min(times) if times else float("nan")


# Student-t, two-sided 95%, indexed by degrees of freedom.  A table rather than scipy because this
# script must run in the bare venv; df>30 falls back to the normal quantile, which is within 4%.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
        23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
        30: 2.042}


def burn_in_reps(ctrl_times, cap):
    """How many leading reps of a window are still on the clock ramp, judged from the CONTROL.

    The ~1.4% monotone ramp documented in CONTEXT.md 45.7a/46.13 is a real transient: the card's
    clocks are still rising for the first few reps of a window.  The 12-rep seq-2048 window of
    2026-08-10 shows what it does to a paired ratio -- `fb_min` vs `hyclora_flash_nc` reads
    -0.84 / -0.14 / -0.06 / -0.17 over reps 1-4 and then settles at -0.51 / -0.60 / -0.51 / -0.55 /
    -0.62 / -0.60 / -0.64 / -0.64.  A 3-rep window samples ONLY the transient, which is why the
    three archived 3-rep estimates of that cell (-0.56, -0.06, +0.29) scatter over 0.85pp around a
    settled value of -0.58%.

    The cut is taken from the CONTROL arm, never from the arms under test.  The control touches
    none of the code under comparison, so a cut chosen from it cannot bias the comparison -- it can
    only discard reps.  Returns the index of the first settled rep.
    """
    for b in range(0, min(cap, max(0, len(ctrl_times) - 3)) + 1):
        tail = ctrl_times[b:]
        if len(tail) < 3:
            break
        # Settled == every remaining rep-to-rep step is under 0.1%.  The ramp moves ~0.3-1% per rep
        # while it is live and ~0.02-0.05% once it is not, so this threshold sits in a wide gap.
        if all(abs(tail[i + 1] / tail[i] - 1.0) < 0.001 for i in range(len(tail) - 1)):
            return b
    return 0


def paired_ci(paired):
    """Uncertainty on a set of per-rep paired ratios.  Returns None if it cannot be formed.

    WHY THIS EXISTS -- and it is a correction to this script, not an addition.

    The original resolution test was `abs(median) > (max - min)`: the effect had to exceed the
    PEAK-TO-PEAK RANGE of the per-rep ratios.  That test is backwards, because the expected range
    of n samples GROWS with n (~1.69 sigma at n=3, ~2.85 at n=8, ~3.26 at n=12) while the
    uncertainty on the central estimate SHRINKS as 1/sqrt(n).  So collecting more reps made a cell
    strictly LESS likely to be declared resolved.

    That is not hypothetical.  seq 4096 was measured twice on 2026-08-09 on the same idle box:
    the 3-rep window resolved at spread 0.23pp, and the 8-rep window -- more data, same answer,
    +0.49% against +0.45% -- was rejected as "NOT RESOLVED" at spread 1.60pp, its range inflated
    by two clock-ramp reps that the median had already absorbed.  CONTEXT.md 45.7a prescribed
    "more reps, not a quieter box" as the fix for unresolved cells; this criterion punished exactly
    that remedy, which is why the prescription never once produced a resolved cell.

    The replacement is the standard paired-difference interval: a Student-t CI on the mean of the
    per-rep ratios.  For n >= 5 the mean is symmetrically trimmed (drop one min, one max) first,
    because the clock ramp puts its transient in the first rep or two and a trimmed mean is the
    cheapest estimator that survives it without discarding the rep count.  RESOLVED means the
    interval excludes zero.  The median is still reported as the point estimate -- it is what every
    archived number in CONTEXT.md quotes, and a mean/median disagreement is itself a signal that
    the window has an outlier, so both are printed.

    ⚠ WHAT THIS INTERVAL IS NOT.  It is a WITHIN-window interval, and the per-rep ratios under it
    are serially correlated (they share one clock ramp, one process, one allocator state), so it is
    a LOWER BOUND on the uncertainty of a claim, not the uncertainty of a claim.  Measured on this
    project's own disk: at seq 4096 five independent windows put the cell at +0.31 / +0.45 / +0.49 /
    +0.60, a between-window scatter of ~0.3pp, while the 8-rep within-window interval is +-0.06pp --
    five times narrower than the windows actually reproduce.  seq 1024 shows the same at ~0.5pp
    (-2.03 on 08-06 against -1.53 on 08-09, CIs overlapping but point estimates far apart).
    So: use this interval to decide whether ONE window resolved, and `agg_paired_windows.py` to
    decide whether a CLAIM holds.  A mechanism worth less than the between-window scatter must be
    certified across independent windows or not at all.
    """
    n = len(paired)
    if n < 3:
        return None
    trimmed = sorted(paired)[1:-1] if n >= 5 else list(paired)
    k = len(trimmed)
    mean = statistics.fmean(trimmed)
    try:
        sd = statistics.stdev(trimmed)
    except statistics.StatisticsError:
        return None
    se = sd / (k ** 0.5)
    half = _T95.get(k - 1, 1.960) * se
    return {"n": n, "k": k, "trimmed": k != n, "mean": mean, "sd": sd,
            "lo": mean - half, "hi": mean + half,
            "resolved": (mean - half) * (mean + half) > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/hyclora/timing")
    ap.add_argument("--tol", type=float, default=1.0, help="control drift %% that voids a window")
    args = ap.parse_args()

    by_seq = load(args.dir)
    if not by_seq:
        raise SystemExit(f"no window_*_seq*.json under {args.dir}")

    verdicts = {}
    for seq in sorted(by_seq):
        arms = by_seq[seq]
        print("=" * 100)
        # Each of the three processes in a window carries its own control, and each is judged on
        # its own: the unsloth and liger arms run in separate processes, so a clean main window
        # says nothing about the window their number came from.
        ctrl_lines, worst = [], 0.0
        # file -> the control series in that file, in rep order, used further down to locate the
        # clock-ramp transient.  Whichever control has the most reps in a file wins; both touch
        # none of the code under test, so either is a valid ruler.
        ctrl_series, ctrl_reps = {}, {}
        for c in CONTROLS:
            for fname in sorted({r["_file"] for r in arms.get(c, [])}):
                rr = sorted(((r["rep"], r["ms_per_step_median"]) for r in arms.get(c, [])
                             if r["_file"] == fname and "error" not in r))
                t = [x for _, x in rr]
                if len(t) > len(ctrl_series.get(fname, [])):
                    ctrl_series[fname] = t
                    ctrl_reps[fname] = [(x, rep) for rep, x in rr]
                if len(t) < 2:
                    continue
                d = drift(t)
                worst = max(worst, d)
                ctrl_lines.append(f"    {c:<18} in {fname:<28} "
                                  f"{'/'.join(f'{x:.1f}' for x in t)} ms  drift {d:6.2f}%")
        ok = bool(ctrl_lines) and worst <= args.tol
        verdicts[seq] = ("VALID" if ok else "VOID" if (worst > HARD_VOID or not ctrl_lines)
                         else "DRIFTING")
        verdict = {"VALID": "VALID",
                   "VOID": "VOID (protocol A.3.4: discard the window)",
                   "DRIFTING": "DRIFTING (raw medians not quotable; see paired ratios)"}[
            verdicts[seq]]
        print(f"seq {seq}  --  control drift {worst:.2f}%  =>  {verdict}")
        for line in ctrl_lines:
            print(line)
        if not ctrl_lines:
            print("    !! no control arm has >=2 reps in this window; it cannot be adjudicated")

        # Print every arm's spread even on a void window -- the spread IS the evidence that it is
        # void, and hiding it would make a contaminated run look merely absent.
        print(f"    {'arm':<22} {'reps (ms)':<34} {'median':>9} {'min':>9} {'spread':>8} "
              f"{'peak MiB':>10}")
        for arm in sorted(arms):
            rows = [r for r in arms[arm] if "error" not in r]
            if not rows:
                err = arms[arm][0].get("error", "?")
                print(f"    {arm:<22} ERROR {err[:70]}")
                continue
            t = [r["ms_per_step_median"] for r in rows]
            peak = rows[0]["train_step_peak_alloc_mib"]
            print(f"    {arm:<22} {'/'.join(f'{x:.1f}' for x in t):<34} "
                  f"{statistics.median(t):>9.2f} {min(t):>9.2f} {drift(t):>7.2f}% {peak:>10.2f}")

        if worst > HARD_VOID:
            print(f"    -> control moved more than {HARD_VOID:.0f}%: contaminated, not merely "
                  f"drifting. No comparison printed; the window must be re-run.")
            continue

        # ---------------------------------------------------------------------------------------
        # Two estimators, because they answer different questions.
        #
        # RAW is the median over reps.  It is the right number to quote when the window is clean
        # (control within tolerance), and it is what every archived table in this project holds.
        #
        # PAIRED is the median over reps of `t_ours[r] / t_comp[r]`.  The arms are INTERLEAVED --
        # rep r of every arm runs at the same point in the window -- so a drift common to the whole
        # window cancels in the ratio.  seq 2048 is exactly why this exists: its control drifted
        # 1.36% (every arm sliding monotonically downward as the card's clocks ramped) while the
        # `fb_min` vs `hyclora_flash_nc` difference under test is 0.21%.  The raw medians cannot
        # resolve that; the paired ratios can, PROVIDED their own scatter is smaller than the
        # effect -- which is why the per-rep ratios are printed rather than summarised away.
        #
        # This is not a loosening of protocol A.3.4.  A.3.4 governs what may be QUOTED as a clean
        # measurement, and that is still the RAW column with a passing control.  The paired column
        # is reported with its scatter and is only ever as strong as that scatter allows.  Note
        # also that a control arm cancels out of a pairwise ratio algebraically, so normalising by
        # it would add nothing -- its job here is to detect a STEP change (a tenant arriving
        # mid-window), which the ratio would not reveal.
        print()
        for ours in OURS:
            o_rows = [r for r in arms.get(ours, []) if "error" not in r]
            if not o_rows:
                continue
            o = statistics.median([r["ms_per_step_median"] for r in o_rows])
            om = o_rows[0]["train_step_peak_alloc_mib"]
            for comp in MUST_BEAT:
                rows = [r for r in arms.get(comp, []) if "error" not in r]
                if not rows:
                    print(f"    {ours} vs {comp:<20} -- MISSING (row not measured in this window)")
                    continue
                c = statistics.median([r["ms_per_step_median"] for r in rows])
                cm = rows[0]["train_step_peak_alloc_mib"]
                dt, dm = 100.0 * (o - c) / c, 100.0 * (om - cm) / cm

                # Pair by rep index, and only within one process -- rows measured in different
                # processes did not share a window and pairing them would be meaningless.
                paired, tagged = [], []
                for r_o in o_rows:
                    for r_c in rows:
                        if r_o["rep"] == r_c["rep"] and r_o["_file"] == r_c["_file"]:
                            ratio = 100.0 * (r_o["ms_per_step_median"]
                                             / r_c["ms_per_step_median"] - 1.0)
                            paired.append(ratio)
                            tagged.append((r_o["_file"], r_o["rep"], ratio))
                ci = None
                if paired:
                    pm = statistics.median(paired)
                    spread = max(paired) - min(paired)
                    resolved = abs(pm) > spread
                    ptxt = (f"paired {pm:+6.2f}% (per-rep {'/'.join(f'{p:+.2f}' for p in paired)}"
                            f", spread {spread:.2f}pp"
                            f"{'' if resolved else ' -- NOT RESOLVED: spread exceeds effect'})")
                    ci = paired_ci(paired)
                else:
                    ptxt = "paired n/a (separate process -- no shared window)"
                mark = "WIN " if (dt < 0 and dm < 0) else "----"
                print(f"    {mark} {ours} vs {comp:<20} time {dt:+7.2f}%  memory {dm:+7.2f}%")
                print(f"         {ptxt}")
                # The line above is the LEGACY criterion, printed verbatim so that every number
                # already quoted in CONTEXT.md can still be reproduced from this script.  The line
                # below is the one to quote from now on -- see paired_ci().
                if ci:
                    tag = "RESOLVED  " if ci["resolved"] else "unresolved"
                    how = f"n={ci['n']}" + (f", trimmed to {ci['k']}" if ci["trimmed"] else "")
                    print(f"         [t-CI] {tag} {ci['mean']:+6.2f}% "
                          f"95%CI [{ci['lo']:+.2f},{ci['hi']:+.2f}]pp  ({how})")
                # Drop the clock-ramp transient, with the cut taken from the control, and report
                # the settled estimate separately.  Only for windows long enough to afford it: at
                # n<8 discarding reps costs more in degrees of freedom than the ramp costs in bias.
                if ci and ci["n"] >= 8 and ctrl_series:
                    kept = []
                    for fname, rep, ratio in tagged:
                        series = ctrl_series.get(fname)
                        if not series:
                            # No control in this arm's own file: nothing legitimate to cut with,
                            # so keep the rep rather than guess a burn-in from the arms themselves.
                            kept.append(ratio)
                            continue
                        b = burn_in_reps(series, cap=len(series) // 3)
                        if rep >= min(r for _, r in ctrl_reps[fname]) + b:
                            kept.append(ratio)
                    sci = paired_ci(kept) if len(kept) < ci["n"] else None
                    if sci:
                        stag = "RESOLVED  " if sci["resolved"] else "unresolved"
                        print(f"         [settled] {stag} {sci['mean']:+6.2f}% "
                              f"95%CI [{sci['lo']:+.2f},{sci['hi']:+.2f}]pp  "
                              f"(control-chosen burn-in dropped {ci['n'] - len(kept)} rep(s), "
                              f"then n={sci['n']} trimmed to {sci['k']})")

    print("=" * 100)
    print(f"VALID lengths (raw medians quotable): "
          f"{sorted(s for s, v in verdicts.items() if v == 'VALID') or 'NONE'}")
    drifting = sorted(s for s, v in verdicts.items() if v == 'DRIFTING')
    void = sorted(s for s, v in verdicts.items() if v == 'VOID')
    if drifting:
        print(f"DRIFTING lengths (quote the paired ratio, and only if it resolved): {drifting}")
    if void:
        print(f"VOID lengths (re-run required): {void}")


if __name__ == "__main__":
    main()
