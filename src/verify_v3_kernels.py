"""FlashFFN v3 fused Triton kernels (K1/K2) — correctness gates + benchmark.

G1  K1 (_v3_silu_mul_quant_fwd_kernel) vs the eager PyTorch codec on random
    AND adversarial inputs: packed bytes / scales bitwise-identical, h_mid
    bitwise-identical to triton_silu_mul.
G2  K2 (_v3_dequant_swiglu_bwd_kernel) vs eager dequant +
    triton_swiglu_backward (+ triton_silu_mul for h_mid^): bitwise + max diff.
G3  Existing V2 gradient gate re-run with use_triton=True (now the default),
    plus an explicit triton-vs-eager autograd cross-check with honesty
    counters.
G4  Shape robustness: N x D_int sweep, both bit-widths; ragged last group
    must assert cleanly (matching the eager codec's contract).

Benchmark (only after all gates pass): full fwd+bwd step at TinyLlama dims
(2048/5632), PEFT-LoRA r=8 on all 3 projections, N in {2048, 8192, 16384};
arms = standard PEFT autograd, v2 effective-weight FlashFFN, v3-recompute,
v3-int4-eager, v3-int4-triton, v3-int8-triton. CUDA events, >=10 warmup,
>=50 iters, all arms interleaved per-iteration (equalizes co-tenant
contention), p10 + median. Plus kernel-level isolation: K1 vs
(triton_silu_mul + eager quant x2), K2 vs (eager dequant x2 +
triton_swiglu_backward + triton_silu_mul).

Run from repo root:
  source env/bin/activate
  HF_HOME=./data TORCH_HOME=./data python src/verify_v3_kernels.py \
      [--gpu auto|0|1] [--gates g1 g2 g3 g4 bench] [--quick]
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time


def _pick_gpu(arg):
    if arg != "auto":
        return arg
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
        rows = [tuple(int(v) for v in line.split(",")) for line in
                out.strip().splitlines()]
        rows.sort(key=lambda r: (r[1], r[2]))
        return str(rows[0][0])
    except Exception:
        return "0"


_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", default="auto")
_pre_args, _ = _pre.parse_known_args()
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pick_gpu(_pre_args.gpu)
os.environ.setdefault("HF_HOME", "./data")
os.environ.setdefault("TORCH_HOME", "./data")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flashffn  # noqa: E402
from flashffn import (  # noqa: E402
    FlashFFNFunction,
    FlashFFNv3Function,
    make_v3_forward,
    triton_silu_mul,
    triton_swiglu_backward,
    triton_silu_mul_quant,
    triton_dequant_swiglu_backward,
    v3_quantize_group_absmax,
    v3_dequantize_group_absmax,
    v3_reset_counters,
    _V3_COUNTERS,
    _V3_TRITON_CFG,
)
from verify_flashffn_v3 import (  # noqa: E402
    build_mlp, lora_eff_weight, synthetic_param_set, run_ref_grads_fp32,
    tensor_cos, D_HID, D_INT, DTYPE,
)

OUT_DIR = "results/v3_prototype"
GROUP = 64


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Input generators (G1/G2 adversarial cases)
# ---------------------------------------------------------------------------

def make_inputs(kind, N, D, device, seed=41):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def randn(scale=1.0):
        return (torch.randn(N, D, generator=g) * scale).to(device, DTYPE)

    if kind == "random":
        return randn(3.0), randn(2.0)
    if kind == "zeros":
        z = torch.zeros(N, D, device=device, dtype=DTYPE)
        return z, z.clone()
    if kind == "outlier":
        # one large outlier per group, rest tiny -> stresses scale rounding
        base = randn(1e-3)
        idx = torch.randint(0, GROUP, (N, D // GROUP), generator=g)
        big = base.float().view(N, D // GROUP, GROUP)
        sign = torch.where(torch.rand(N, D // GROUP, generator=g) > 0.5,
                           1.0, -1.0).to(device)
        big.scatter_(2, idx.to(device).unsqueeze(-1), (1e4 * sign).unsqueeze(-1))
        a = big.view(N, D).to(DTYPE)
        return a, randn(2.0)
    if kind == "negatives":
        return (-randn(3.0).abs()), (-randn(2.0).abs())
    if kind == "denormal":
        # bf16 subnormal/min-normal territory (~1e-38..1e-41)
        a = (torch.randn(N, D, generator=g) * 1e-38).to(device, DTYPE)
        b = (torch.randn(N, D, generator=g) * 1e-40).to(device, DTYPE)
        return a, b
    if kind == "mixed_magnitude":
        mag = torch.exp(torch.empty(N, D).uniform_(-60, 6, generator=g))
        a = (torch.randn(N, D, generator=g) * mag).to(device, DTYPE)
        b = (torch.randn(N, D, generator=g) * mag.flip(0)).to(device, DTYPE)
        return a, b
    raise ValueError(kind)


def unpack_codes(payload, bits):
    if bits == 8:
        return payload.int()
    lo = torch.bitwise_and(payload, 0x0F).int()
    hi = torch.bitwise_right_shift(payload, 4).int()
    return torch.stack((lo, hi), dim=-1).reshape(payload.shape[0], -1)


def compare_payload(q, q_ref, s, s_ref, bits):
    """Bitwise compare; on mismatch quantify code-level deviation."""
    res = {"payload_bitwise": bool(torch.equal(q, q_ref)),
           "scales_bitwise": bool(torch.equal(s, s_ref))}
    if not res["payload_bitwise"]:
        c, c_ref = unpack_codes(q, bits), unpack_codes(q_ref, bits)
        neq = c != c_ref
        res["code_mismatch_frac"] = float(neq.float().mean())
        res["max_abs_code_diff"] = int((c - c_ref).abs().max())
    if not res["scales_bitwise"]:
        res["scale_mismatch_frac"] = float((s != s_ref).float().mean())
        res["max_scale_rel_diff"] = float(
            ((s.float() - s_ref.float()).abs()
             / s_ref.float().abs().clamp_min(1e-30)).max())
    return res


def judge_payload(res):
    if res["payload_bitwise"] and res["scales_bitwise"]:
        return True
    # documented fallback allowance: <1e-4 mismatched codes, |diff| <= 1
    return (res.get("code_mismatch_frac", 0.0) < 1e-4
            and res.get("max_abs_code_diff", 0) <= 1
            and res["scales_bitwise"])


# ---------------------------------------------------------------------------
# G1 — K1 vs eager codec
# ---------------------------------------------------------------------------

def gate_g1(args, device):
    log("\n=== G1: K1 fused fwd kernel vs eager codec (bitwise) ===")
    kinds = ("random", "zeros", "outlier", "negatives", "denormal",
             "mixed_magnitude")
    cases = []
    ok_all = True
    for kind in kinds:
        for bits in (4, 8):
            N, D = (512, D_INT) if args.quick else (2048, D_INT)
            hg, hu = make_inputs(kind, N, D, device)
            mid_ref = triton_silu_mul(hg, hu)
            qg_ref, sg_ref = v3_quantize_group_absmax(hg, bits, GROUP)
            qu_ref, su_ref = v3_quantize_group_absmax(hu, bits, GROUP)
            mid, qg, sg, qu, su = triton_silu_mul_quant(hg, hu, bits, GROUP)
            rg = compare_payload(qg, qg_ref, sg, sg_ref, bits)
            ru = compare_payload(qu, qu_ref, su, su_ref, bits)
            mid_bw = bool(torch.equal(mid, mid_ref))
            ok = mid_bw and judge_payload(rg) and judge_payload(ru)
            ok_all &= ok
            cases.append({"kind": kind, "bits": bits, "N": N, "D": D,
                          "h_mid_bitwise": mid_bw, "gate": rg, "up": ru,
                          "pass": ok})
            log(f"  {kind:16s} int{bits}: h_mid bitwise={mid_bw} "
                f"gate={'bitwise' if rg['payload_bitwise'] else rg} "
                f"up={'bitwise' if ru['payload_bitwise'] else ru} "
                f"-> {'PASS' if ok else 'FAIL'}")
    log(f"G1 OVERALL: {'PASS' if ok_all else 'FAIL'}")
    return ok_all, cases


# ---------------------------------------------------------------------------
# G2 — K2 vs eager dequant + triton_swiglu_backward (+ triton_silu_mul)
# ---------------------------------------------------------------------------

def gate_g2(args, device):
    log("\n=== G2: K2 fused bwd kernel vs eager dequant+swiglu_bwd ===")
    kinds = ("random", "zeros", "outlier", "negatives", "denormal",
             "mixed_magnitude")
    cases = []
    ok_all = True
    for kind in kinds:
        for bits in (4, 8):
            N, D = (512, D_INT) if args.quick else (2048, D_INT)
            hg, hu = make_inputs(kind, N, D, device)
            torch.manual_seed(args.seed + 5)
            gmid = torch.randn(N, D, device=device).to(DTYPE)
            qg, sg = v3_quantize_group_absmax(hg, bits, GROUP)
            qu, su = v3_quantize_group_absmax(hu, bits, GROUP)
            hgd = v3_dequantize_group_absmax(qg, sg, bits, GROUP, DTYPE)
            hud = v3_dequantize_group_absmax(qu, su, bits, GROUP, DTYPE)
            gg_ref, gu_ref = triton_swiglu_backward(gmid, hgd, hud)
            hm_ref = triton_silu_mul(hgd, hud)
            gg, gu, hm = triton_dequant_swiglu_backward(
                gmid, qg, sg, qu, su, bits, GROUP, want_h_mid=True)
            gg2, gu2, hm2 = triton_dequant_swiglu_backward(
                gmid, qg, sg, qu, su, bits, GROUP, want_h_mid=False)
            d = {
                "gg_bitwise": bool(torch.equal(gg, gg_ref)),
                "gu_bitwise": bool(torch.equal(gu, gu_ref)),
                "hmid_bitwise": bool(torch.equal(hm, hm_ref)),
                "no_hmid_consistent": bool(torch.equal(gg2, gg)
                                           and torch.equal(gu2, gu)
                                           and hm2 is None),
                "max_diff_gg": float((gg.float() - gg_ref.float()).abs().max()),
                "max_diff_gu": float((gu.float() - gu_ref.float()).abs().max()),
                "max_diff_hmid": float((hm.float() - hm_ref.float()).abs().max()),
                "allclose_bf16": bool(
                    torch.allclose(gg.float(), gg_ref.float())
                    and torch.allclose(gu.float(), gu_ref.float())
                    and torch.allclose(hm.float(), hm_ref.float())),
            }
            ok = (d["gg_bitwise"] and d["gu_bitwise"] and d["hmid_bitwise"]
                  and d["no_hmid_consistent"])
            ok_all &= ok
            cases.append({"kind": kind, "bits": bits, "N": N, "D": D, **d,
                          "pass": ok})
            log(f"  {kind:16s} int{bits}: gg/gu/hmid bitwise="
                f"{d['gg_bitwise']}/{d['gu_bitwise']}/{d['hmid_bitwise']} "
                f"max|diff|={max(d['max_diff_gg'], d['max_diff_gu'], d['max_diff_hmid']):.3e} "
                f"-> {'PASS' if ok else 'FAIL'}")
    log(f"G2 OVERALL: {'PASS' if ok_all else 'FAIL'}")
    return ok_all, cases


# ---------------------------------------------------------------------------
# G3 — existing V2 gradient gate with use_triton=True + honesty cross-check
# ---------------------------------------------------------------------------

def _run_v3_grads_flag(ps, x16, gy16, mode, use_triton):
    def leaf(t, rg):
        return None if t is None else t.detach().clone().requires_grad_(rg)

    xl = x16.detach().clone().requires_grad_(True)
    t = {}
    for key in ("gate", "up", "down"):
        t[f"w_{key}"] = leaf(ps[f"w_{key}"], ps["train_w"])
        has = ps[f"a_{key}"] is not None
        t[f"a_{key}"] = leaf(ps[f"a_{key}"], has)
        t[f"b_{key}"] = leaf(ps[f"b_{key}"], has)
    v3_reset_counters()
    y = FlashFFNv3Function.apply(
        xl, t["w_gate"], t["a_gate"], t["b_gate"],
        t["w_up"], t["a_up"], t["b_up"],
        t["w_down"], t["a_down"], t["b_down"],
        ps["s_gate"], ps["s_up"], ps["s_down"], mode, GROUP, use_triton)
    y.backward(gy16)
    counters = dict(_V3_COUNTERS)
    grads = {"grad_x": xl.grad}
    for k, tt in t.items():
        if tt is not None and tt.requires_grad:
            grads[k] = tt.grad
    return grads, counters


def gate_g3(args, device):
    log("\n=== G3: V2 gradient gate with use_triton=True + honesty ===")
    # (a) explicit triton-vs-eager autograd cross-check with counters
    ps = synthetic_param_set("all", args.seed, device)
    torch.manual_seed(args.seed + 7)
    N = 2048
    x16 = torch.randn(N, D_HID).to(device, DTYPE)
    gy16 = torch.randn(N, D_HID).to(device, DTYPE)
    ref = run_ref_grads_fp32(ps, x16, gy16)
    detail = {"cross_check": {}, "honesty": {}}
    ok_all = True
    for mode in ("int4", "int8"):
        g_tri, c_tri = _run_v3_grads_flag(ps, x16, gy16, mode, True)
        g_eag, c_eag = _run_v3_grads_flag(ps, x16, gy16, mode, False)
        hon = (c_tri["triton_quant_fwd"] == 1
               and c_tri["triton_dequant_bwd"] == 1
               and c_eag["triton_quant_fwd"] == 0
               and c_eag["triton_dequant_bwd"] == 0)
        ok_all &= hon
        rows = {}
        worst_cos_gap = 0.0
        bitwise_all = True
        for k in sorted(ref.keys()):
            cos_t = tensor_cos(g_tri[k], ref[k])
            cos_e = tensor_cos(g_eag[k], ref[k])
            bw = bool(torch.equal(g_tri[k], g_eag[k]))
            bitwise_all &= bw
            gap = abs(cos_t - cos_e)
            worst_cos_gap = max(worst_cos_gap, gap)
            rows[k] = {"cos_triton": cos_t, "cos_eager": cos_e,
                       "triton_eq_eager_bitwise": bw, "cos_gap": gap}
        ok = worst_cos_gap <= 1e-3
        ok_all &= ok
        detail["cross_check"][mode] = rows
        detail["honesty"][mode] = {"triton_counters": c_tri,
                                   "eager_counters": c_eag, "pass": hon}
        log(f"  {mode}: triton grads == eager grads bitwise: {bitwise_all}; "
            f"max |cos_triton - cos_eager| = {worst_cos_gap:.2e} "
            f"(report if >1e-3) | counters {'OK' if hon else 'VIOLATED'}")
    del ref, ps

    # (b) the full existing V2 gate (synthetic + real capture), use_triton
    #     default True inside FlashFFNv3Function
    log("  re-running src/verify_flashffn_v3.py --gates v2 "
        "(use_triton=True default) ...")
    env = dict(os.environ)
    proc = subprocess.run(
        [sys.executable, "src/verify_flashffn_v3.py", "--gates", "v2",
         "--seed", str(args.seed)],
        capture_output=True, text=True, env=env)
    tail = "\n".join(proc.stdout.strip().splitlines()[-14:])
    log("  --- v2 gate output (tail) ---\n" +
        "\n".join("  | " + ln for ln in tail.splitlines()))
    v2_ok = False
    v2_json = None
    if proc.returncode == 0:
        with open(os.path.join(OUT_DIR, "v2_grads.json")) as f:
            v2_json = json.load(f)
        v2_ok = bool(v2_json["pass"])
    ok_all &= v2_ok
    detail["v2_gate"] = {"returncode": proc.returncode, "pass": v2_ok}
    log(f"  V2 gate (all prior thresholds, triton path): "
        f"{'PASS' if v2_ok else 'FAIL'}")
    log(f"G3 OVERALL: {'PASS' if ok_all else 'FAIL'}")
    return ok_all, detail


# ---------------------------------------------------------------------------
# G4 — shape robustness
# ---------------------------------------------------------------------------

def gate_g4(args, device):
    log("\n=== G4: shape robustness ===")
    n_list = (1, 7, 512, 2048, 8192)
    d_list = (5632, 11008, 14336)
    if args.quick:
        n_list, d_list = (1, 7, 512), (5632, 11008)
    cases = []
    ok_all = True
    for D in d_list:
        for N in n_list:
            for bits in (4, 8):
                hg, hu = make_inputs("random", N, D, device,
                                     seed=args.seed + N + D)
                torch.manual_seed(args.seed + 1)
                gmid = torch.randn(N, D, device=device).to(DTYPE)
                mid_ref = triton_silu_mul(hg, hu)
                qg_r, sg_r = v3_quantize_group_absmax(hg, bits, GROUP)
                qu_r, su_r = v3_quantize_group_absmax(hu, bits, GROUP)
                mid, qg, sg, qu, su = triton_silu_mul_quant(hg, hu, bits, GROUP)
                hgd = v3_dequantize_group_absmax(qg_r, sg_r, bits, GROUP, DTYPE)
                hud = v3_dequantize_group_absmax(qu_r, su_r, bits, GROUP, DTYPE)
                gg_r, gu_r = triton_swiglu_backward(gmid, hgd, hud)
                hm_r = triton_silu_mul(hgd, hud)
                gg, gu, hm = triton_dequant_swiglu_backward(
                    gmid, qg_r, sg_r, qu_r, su_r, bits, GROUP, want_h_mid=True)
                ok = (torch.equal(mid, mid_ref) and torch.equal(qg, qg_r)
                      and torch.equal(sg, sg_r) and torch.equal(qu, qu_r)
                      and torch.equal(su, su_r) and torch.equal(gg, gg_r)
                      and torch.equal(gu, gu_r) and torch.equal(hm, hm_r))
                ok_all &= ok
                cases.append({"N": N, "D": D, "bits": bits,
                              "all_bitwise": bool(ok)})
                del hg, hu, gmid, mid_ref, mid, hgd, hud
        log(f"  D={D}: N in {n_list} x int4/int8 -> "
            f"{'all bitwise' if all(c['all_bitwise'] for c in cases if c['D'] == D) else 'MISMATCH'}")
        torch.cuda.empty_cache()

    # ragged last group: both eager codec and fused kernels must assert cleanly
    ragged = {"D": 5632 + 32}
    hg, hu = make_inputs("random", 64, 5632 + 32, device)
    for name, fn in (
        ("eager", lambda: v3_quantize_group_absmax(hg, 4, GROUP)),
        ("triton_k1", lambda: triton_silu_mul_quant(hg, hu, 4, GROUP)),
        ("triton_k2", lambda: triton_dequant_swiglu_backward(
            hg, hg.view(torch.uint8)[:, :2832], None, None, None, 4, GROUP)),
    ):
        try:
            fn()
            ragged[name] = "NO ASSERT (BUG)"
            ok_all = False
        except AssertionError as e:
            ragged[name] = f"clean assert: {e}"
    log(f"  ragged last group (D=5664, 5664%64=32): eager/k1/k2 all assert "
        f"cleanly: "
        f"{all('clean assert' in str(v) for k, v in ragged.items() if k != 'D')}")
    cases.append({"ragged": ragged})
    log(f"G4 OVERALL: {'PASS' if ok_all else 'FAIL'}")
    return ok_all, cases


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _expected_counter_delta(arm):
    # (forward, backward, triton_quant_fwd, triton_dequant_bwd) per step
    if arm in ("standard_peft", "v2_eff_weight"):
        return (0, 0, 0, 0)
    if arm in ("v3_recompute", "v3_int4_eager"):
        return (1, 1, 0, 0)
    return (1, 1, 1, 1)  # v3_int4_triton / v3_int8_triton


def bench(args, device):
    log("\n=== BENCHMARK: full fwd+bwd step, interleaved arms ===")
    warmup, iters = (3, 10) if args.quick else (10, 50)
    mlp = build_mlp("all", args.seed, device)
    adapter_params = [p for p in mlp.parameters() if p.requires_grad]
    n_list = (2048, 8192) if args.quick else (2048, 8192, 16384)
    out = {"warmup": warmup, "iters": iters}
    targets = None

    for N in n_list:
        torch.manual_seed(args.seed + N)
        x_vals = torch.randn(N, D_HID).to(device, DTYPE)
        gy = torch.randn(N, D_HID).to(device, DTYPE)
        arm_x = {}

        def make_x(name):
            arm_x[name] = x_vals.detach().clone().requires_grad_(True)
            return arm_x[name]

        def step_standard(x=make_x("standard_peft")):
            mlp(x).backward(gy)

        def step_v2(x=make_x("v2_eff_weight")):
            FlashFFNFunction.apply(
                x, lora_eff_weight(mlp.gate_proj), lora_eff_weight(mlp.up_proj),
                lora_eff_weight(mlp.down_proj), 0.3).backward(gy)

        fwds = {
            "v3_recompute": make_v3_forward(mlp, "recompute"),
            "v3_int4_eager": make_v3_forward(mlp, "int4", use_triton=False),
            "v3_int4_triton": make_v3_forward(mlp, "int4", use_triton=True),
            "v3_int8_triton": make_v3_forward(mlp, "int8", use_triton=True),
        }

        def make_step(name):
            x = make_x(name)

            def step():
                fwds[name](x).backward(gy)
            return step

        arms = {"standard_peft": step_standard, "v2_eff_weight": step_v2}
        for name in fwds:
            arms[name] = make_step(name)

        # honesty pre-pass: each arm must hit exactly its expected v3/triton
        # counters (a silent eager fallback would be caught here)
        for name, step in arms.items():
            v3_reset_counters()
            step()
            got = (_V3_COUNTERS["forward"], _V3_COUNTERS["backward"],
                   _V3_COUNTERS["triton_quant_fwd"],
                   _V3_COUNTERS["triton_dequant_bwd"])
            exp = _expected_counter_delta(name)
            assert got == exp, f"{name}: counters {got} != expected {exp}"
            arm_x[name].grad = None
            for p in adapter_params:
                p.grad = None
        torch.cuda.synchronize()

        events = {a: [] for a in arms}
        v3_reset_counters()
        for it in range(warmup + iters):
            for name, step in arms.items():
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                step()
                e1.record()
                if it >= warmup:
                    events[name].append((e0, e1))
                arm_x[name].grad = None
                for p in adapter_params:
                    p.grad = None
            torch.cuda.synchronize()
        total = warmup + iters
        assert _V3_COUNTERS["triton_quant_fwd"] == 2 * total \
            and _V3_COUNTERS["triton_dequant_bwd"] == 2 * total, \
            f"triton arms did not run: {_V3_COUNTERS}"
        assert _V3_COUNTERS["forward"] == 4 * total, _V3_COUNTERS

        stats = {}
        for name, evs in events.items():
            ts = sorted(e0.elapsed_time(e1) for e0, e1 in evs)
            stats[name] = {
                "p10_ms": ts[max(0, int(0.10 * len(ts)) - 1)],
                "median_ms": statistics.median(ts),
                "n": len(ts),
            }
        out[str(N)] = {"stats": stats}

        log(f"  N={N} (median / p10 ms):")
        for name in arms:
            s = stats[name]
            log(f"    {name:16s}: {s['median_ms']:8.3f} / {s['p10_ms']:8.3f}")
        if N == 8192:
            targets = {}
            for kind in ("median_ms", "p10_ms"):
                r_v2 = stats["v3_int4_triton"][kind] / stats["v2_eff_weight"][kind]
                r_std = stats["v3_int4_triton"][kind] / stats["standard_peft"][kind]
                targets[kind] = {"int4_triton_over_v2": r_v2,
                                 "int4_triton_over_std": r_std,
                                 "pass": r_v2 <= 0.75 and r_std <= 1.08}
                log(f"    targets @8192 ({kind:9s}): int4_triton/v2="
                    f"{r_v2:.3f} (<=0.75), int4_triton/std={r_std:.3f} "
                    f"(<=1.08) -> {'PASS' if targets[kind]['pass'] else 'FAIL'}")
            out["8192"]["targets"] = targets
        del x_vals, gy, arm_x
        torch.cuda.empty_cache()

    # ---- kernel-level isolation: K1 / K2 vs their eager equivalents ----
    log("  kernel isolation (median ms, interleaved):")
    iso = {}
    for N in n_list:
        torch.manual_seed(args.seed + N)
        hg = (torch.randn(N, D_INT, device=device) * 3).to(DTYPE)
        hu = (torch.randn(N, D_INT, device=device) * 2).to(DTYPE)
        gmid = torch.randn(N, D_INT, device=device).to(DTYPE)
        row = {}
        for bits in (4, 8):
            qg, sg = v3_quantize_group_absmax(hg, bits, GROUP)
            qu, su = v3_quantize_group_absmax(hu, bits, GROUP)

            def eager_fwd():
                m = triton_silu_mul(hg, hu)
                v3_quantize_group_absmax(hg, bits, GROUP)
                v3_quantize_group_absmax(hu, bits, GROUP)
                return m

            def k1():
                return triton_silu_mul_quant(hg, hu, bits, GROUP)

            def eager_bwd():
                hgd = v3_dequantize_group_absmax(qg, sg, bits, GROUP, DTYPE)
                hud = v3_dequantize_group_absmax(qu, su, bits, GROUP, DTYPE)
                triton_swiglu_backward(gmid, hgd, hud)
                return triton_silu_mul(hgd, hud)

            def k2():
                return triton_dequant_swiglu_backward(
                    gmid, qg, sg, qu, su, bits, GROUP, want_h_mid=True)

            fns = {"k1_fused": k1, "fwd_eager": eager_fwd,
                   "k2_fused": k2, "bwd_eager": eager_bwd}
            ts = {k: [] for k in fns}
            wu_i, it_i = (3, 10) if args.quick else (5, 30)
            for it in range(wu_i + it_i):
                for kname, fn in fns.items():
                    # drain the queue first: with a deep backlog, event deltas
                    # can blame an arm for neighbouring allocator/stream work
                    # (reproduced: K2@16384 read 3.9 ms in a backlogged
                    # interleave vs 1.5 ms standalone/pair/drained)
                    torch.cuda.synchronize()
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    fn()
                    e1.record()
                    if it >= wu_i:
                        ts[kname].append((e0, e1))
                torch.cuda.synchronize()
            med = {k: statistics.median(e0.elapsed_time(e1) for e0, e1 in v)
                   for k, v in ts.items()}
            row[f"int{bits}"] = {
                **{k: round(v, 4) for k, v in med.items()},
                "k1_speedup": med["fwd_eager"] / med["k1_fused"],
                "k2_speedup": med["bwd_eager"] / med["k2_fused"],
            }
            log(f"    N={N:5d} int{bits}: K1 {med['k1_fused']:7.3f} ms vs "
                f"eager {med['fwd_eager']:7.3f} ({row[f'int{bits}']['k1_speedup']:.1f}x) | "
                f"K2 {med['k2_fused']:7.3f} ms vs eager {med['bwd_eager']:7.3f} "
                f"({row[f'int{bits}']['k2_speedup']:.1f}x)")
        iso[str(N)] = row
        del hg, hu, gmid
        torch.cuda.empty_cache()
    out["kernel_isolation"] = iso
    target_pass = bool(targets and targets["median_ms"]["pass"])
    return target_pass, out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--gates", nargs="+",
                    default=["g1", "g2", "g3", "g4", "bench"])
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="benchmark even if gates fail (numbers flagged)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = "cuda"
    import triton
    log(f"device: {torch.cuda.get_device_name(0)} | CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES')} | torch {torch.__version__}"
        f" | triton {triton.__version__} | cfg {_V3_TRITON_CFG}")

    summary, details = {}, {}
    t0 = time.time()
    gates_ok = True
    for gate in args.gates:
        if gate == "bench":
            if not gates_ok and not args.force:
                log("\nSKIPPING bench: correctness gates failed "
                    "(never benchmark a kernel that fails gates)")
                summary["bench"] = "skipped (gates failed)"
                continue
            ok, detail = bench(args, device)
            summary["bench_targets"] = "PASS" if ok else "FAIL"
            details["bench"] = detail
            continue
        fn = {"g1": gate_g1, "g2": gate_g2, "g3": gate_g3, "g4": gate_g4}[gate]
        ok, detail = fn(args, device)
        summary[gate] = "PASS" if ok else "FAIL"
        details[gate] = detail
        gates_ok &= ok

    log("\n" + "=" * 60)
    log("SUMMARY: " + " | ".join(f"{g.upper()}={s}" for g, s in summary.items()))
    log(f"total wall time: {time.time() - t0:.0f}s")
    payload = {"summary": summary, "args": vars(args),
               "triton_cfg": dict(_V3_TRITON_CFG),
               "gpu": torch.cuda.get_device_name(0),
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "wall_s": time.time() - t0, "details": details}
    path = os.path.join(OUT_DIR, "kernel_bench.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=str)
    log(f"[saved {path}]")


if __name__ == "__main__":
    main()
