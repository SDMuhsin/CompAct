"""Pinned-host residency for the FROZEN base weights: stream each layer's weights in, use, free.

WHAT THIS IS AND WHOSE IT IS
---------------------------
This is **parameter offload**, and it is not ours.  DeepSpeed ZeRO-Offload / ZeRO-Infinity
(arXiv 2101.06840, 2104.07857) and HuggingFace `accelerate`'s `device_map={"": "cpu"}` /
`offload_folder` own the mechanism outright, as do `bitsandbytes`' paged optimizers for the state
half and llama.cpp / FlexGen for inference-time layer streaming.  Nothing here is a contribution;
it is engineering applied to a pool this project had never touched.

WHY IT IS WORTH BUILDING ANYWAY
-------------------------------
`fb_min`'s peak allocated has always decomposed as a constant resident floor plus activations.
The floor is 2188.90 MiB and constant to 0.70 MiB across a 16x sequence span, and the live dtype
receipt says `base/torch.bfloat16` = 2098.18 MiB of it -- 96%.  Four activation-side mechanisms
have since cut the activation half hard, which makes the floor ~50% of peak at seq 16384 and ~93%
at seq 1024.  It is now the largest single removable pool at every length.

WHY IT IS EXACTLY BITWISE, AND EASY
-----------------------------------
Moving a tensor between devices changes no bits, and under LoRA the base weights are frozen:
read-only, no gradient, no writeback, and the optimizer never sees them (it sees only the 308
adapter tensors).  So this is a pure read-side staging problem.  The host copy is written ONCE at
install and never again, so there is no D2H at all -- unlike `fb_offload.py`, whose `o_h` goes out
and comes back.  Same bytes every step, never modified.

THE THREE HAZARDS, AND WHICH OF `fb_offload.py`'s APPLY
------------------------------------------------------
`fb_offload.py`'s module docstring lists three.  Ours is H2D-only, so:

1. THE D2H SOURCE -- does not exist here.  The host slab is permanently referenced by this module
   and by `p.data`; nothing can recycle it.

2. THE LANDING BUFFER'S ALLOCATION STREAM -- **applies, and is the one that bit route_b.md.**
   PyTorch's caching allocator is stream-ordered: a freed block returns only to an allocation on
   the same stream, with no event, because a stream is ordered with itself.  Allocating a landing
   slab with the COMPUTE stream current therefore hands back a block whose previous compute-stream
   tenant still has kernels in flight, and the H2D then writes it from the copy stream, unordered
   against them: the transfer is not late, it is overwritten after it lands.  `_stage` allocates
   every slab inside `with torch.cuda.stream(copy_stream)`, so slabs only ever come from the copy
   stream's pool.  `fb_wstream_alloc_stream('compute')` restores the defect on demand.

3. THE LANDING BUFFER'S RECYCLE -- applies, and is covered by `record_stream`.  A slab is WRITTEN
   by the copy stream and READ by the compute stream.  `_release` calls
   `slab.record_stream(compute_stream)` before dropping the reference, which makes the allocator
   insert a compute-stream event at free time and withhold the block from any later copy-stream
   allocation until it fires.  That is the cross-stream write-after-read edge, expressed once.

   `record_stream` decrements `allocated_bytes` immediately (the deferral is in `insert_events`,
   after the stat update in `CUDACachingAllocator::free`), so `memory_allocated` -- and therefore
   the peak this project measures -- sees the slab go away at once.  What it defers is `reserved`
   and reuse.  It is also why `--peak_composition` overstates a staging arm: the allocator-history
   replay counts the requested-to-completed gap as live.

WHAT IS STREAMED, AND WHAT IS NOT
---------------------------------
  * the 7 projection weights of every decoder layer, as ONE contiguous slab per layer (TinyLlama:
    84.01 MiB/layer x 22 = 1848.17 MiB, one 84 MiB DMA rather than seven);
  * `embed_tokens.weight` (125.00 MiB), which under a frozen embedding is read in the forward and
    NEVER again -- `F.embedding` with `weight.requires_grad=False` records no autograd node at
    all, so nothing saves it;
  * `lm_head.weight` (125.00 MiB), staged at the last decoder layer's exit and handed back at the
    model's forward hook.  Its storage then stays alive on the *saved-tensor* reference that Liger
    FLCE (or `nn.Linear`'s backward) holds, and autograd frees it when that node executes -- which
    is the first thing the backward does.  No hook of ours is involved in the release, which is
    what makes it correct under both the FLCE path (which never calls `lm_head.forward`) and the
    stock path (which does).

  NOT streamed: the two RMSNorm weights per layer plus the final norm (8.20 KiB total across the
  whole model), and any base weight that is TRAINABLE.  A layer with a trainable base weight is
  skipped whole, so full fine-tuning simply does not engage this mechanism.

THE ONE THING THE FUSED BLOCK HAD TO CHANGE
-------------------------------------------
`FusedLoRABlockFunction.forward` put the seven `w` in `save_for_backward`.  `SavedVariable` takes
a shallow copy that holds the STORAGE, so a staged slab saved that way stays alive from the
forward to the backward -- i.e. all 22 layers resident, i.e. no saving at all.  Under streaming
those seven slots are saved as `None` and the backward re-acquires from here.  That is sound
because a frozen weight is not an autograd input in any meaningful sense: it needs no version
counter (nothing writes it) and produces no gradient.
"""
import torch

_ALIGN = 512

_WS = {
    "on": False,             # whether `apply_flash_block` should install a streamer
    "streamer": None,        # the installed WeightStreamer, or None
    "cur": None,             # (streamer, layer_idx) set by the fused wrapper around .apply()
    "alloc_stream": "copy",  # 'compute' reproduces hazard 2 on demand
    "lookahead": 1,          # layers prefetched ahead of the one in use
    "bwd": "hold",           # 'hold' (2 stagings/layer/step) or 'split' (3)
    "embed": True,
    "head": True,
    "fault": 0,              # corrupt one element of every Nth landing slab (0 = off)
    "last_declined": None,   # why the last install() declined, or None
}


# ------------------------------------------------------------------------------------------------
# switches -- same shape as `fb_offload`'s, so the off switch looks like every other one here
# ------------------------------------------------------------------------------------------------
def fb_wstream_enable(on=True):
    prev = _WS["on"]
    _WS["on"] = bool(on)
    return prev


def fb_wstream_enabled():
    return _WS["on"]


def fb_wstream_alloc_stream(mode):
    """`'copy'` (correct) or `'compute'` (reproduces hazard 2 -- see the module docstring)."""
    if mode not in ("copy", "compute"):
        raise ValueError(f"fb_wstream: alloc_stream must be 'copy' or 'compute', got {mode!r}")
    prev = _WS["alloc_stream"]
    _WS["alloc_stream"] = mode
    return prev


def fb_wstream_lookahead(n):
    prev = _WS["lookahead"]
    _WS["lookahead"] = max(0, int(n))
    return prev


def fb_wstream_bwd(mode):
    """`'hold'` -- one staging covers the whole backward (recompute AND dgrad), 2 per layer/step.
    `'split'` -- release after the recompute and re-acquire for the dgrad, 3 per layer/step."""
    if mode not in ("hold", "split"):
        raise ValueError(f"fb_wstream: bwd must be 'hold' or 'split', got {mode!r}")
    prev = _WS["bwd"]
    _WS["bwd"] = mode
    return prev


def fb_wstream_parts(embed=None, head=None):
    prev = (_WS["embed"], _WS["head"])
    if embed is not None:
        _WS["embed"] = bool(embed)
    if head is not None:
        _WS["head"] = bool(head)
    return prev


def fb_wstream_fault(n=0):
    """TEETH.  Corrupt one element of every Nth landing slab, after it has arrived.

    An instrument that has only ever passed proves nothing.  With this on, a bitwise A/B and the
    rematerialisation certificate must both FAIL; if they do not, they are not watching the bytes
    this mechanism moves.
    """
    prev = _WS["fault"]
    _WS["fault"] = int(n)
    return prev


def streamer():
    return _WS["streamer"] if _WS["on"] else None


def set_current(pair):
    _WS["cur"] = pair


def current():
    return _WS["cur"]


def reset():
    """Drop the installed streamer and its pinned host memory.

    Called for EVERY arm by `build_model`, for the same reason `fb_offload.reset()` is: the state
    here is process-global and 2 GiB of pinned host memory left over from the previous arm is a
    real cost that must not be carried silently into the next one.
    """
    s = _WS["streamer"]
    _WS["streamer"] = None
    _WS["cur"] = None
    if s is not None:
        s.uninstall()


def fb_wstream_stats():
    s = _WS["streamer"]
    out = {"installed": s is not None, "declined": _WS["last_declined"],
           "on": _WS["on"], "alloc_stream": _WS["alloc_stream"],
           "lookahead": _WS["lookahead"], "bwd": _WS["bwd"],
           "embed": _WS["embed"], "head": _WS["head"], "fault": _WS["fault"]}
    if s is not None:
        out.update(s.report())
    return out


# ------------------------------------------------------------------------------------------------
# one streaming unit
# ------------------------------------------------------------------------------------------------
class _Group:
    """A set of frozen parameters staged together as ONE contiguous slab.

    `host` is pinned and permanent; every member's `p.data` points into it whenever the group is
    not staged, so the model stays self-describing (`state_dict()` returns real weights, `p.dtype`
    and `p.numel()` are unchanged, and any code path that reads the parameter off the module --
    `_fb_factors`, DoRA's column norm, an eval forward through the unfused layer -- just works).
    """

    __slots__ = ("name", "members", "nbytes", "host", "hostviews", "dev", "arrived", "staged_n")

    def __init__(self, name, params):
        self.name = name
        self.members = []            # (param, offset, nbytes, shape, dtype)
        off = 0
        for p in params:
            nb = p.numel() * p.element_size()
            self.members.append((p, off, nb, tuple(p.shape), p.dtype))
            off += (nb + _ALIGN - 1) // _ALIGN * _ALIGN
        self.nbytes = off
        self.host = None
        self.hostviews = None
        self.dev = None
        self.arrived = None
        self.staged_n = 0

    def pin(self):
        """Move the group's bytes to pinned host memory ONCE and re-point every `p.data` at it."""
        self.host = torch.empty(self.nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)
        self.hostviews = []
        for p, off, nb, shape, dtype in self.members:
            hv = self.host[off:off + nb].view(dtype).view(shape)
            hv.copy_(p.data)
            self.hostviews.append(hv)
            p.data = hv

    def unpin(self, device=None):
        """Put the parameters back on `device` (or leave them on the host) and drop the slab."""
        for i, (p, _off, _nb, _shape, _dtype) in enumerate(self.members):
            if device is not None:
                p.data = self.hostviews[i].to(device)
            else:
                p.data = self.hostviews[i].clone()
        self.host = None
        self.hostviews = None
        self.dev = None
        self.arrived = None

    def devviews(self):
        return [self.dev[off:off + nb].view(dtype).view(shape)
                for _p, off, nb, shape, dtype in self.members]


class WeightStreamer:
    """Owns the pinned host copies, the copy stream, and the staging schedule."""

    def __init__(self, device, layer_groups, embed=None, head=None):
        self.device = device
        self.layers = layer_groups            # list, index = decoder layer index
        self.embed = embed
        self.head = head
        self.stream = torch.cuda.Stream(device=device)
        self.n = len(layer_groups)
        self.st = {"h2d_bytes": 0, "stagings": 0, "blocked_acquires": 0,
                   "dev_bytes_live": 0, "dev_bytes_high_water": 0,
                   "fwd_acq": 0, "bwd_acq": 0, "faults": 0}
        self._acq = 0
        self._hooks = []
        self.installed = True

    # ---------------------------------------------------------------------------------- plumbing
    def all_groups(self):
        gs = list(self.layers)
        if self.embed is not None:
            gs.append(self.embed)
        if self.head is not None:
            gs.append(self.head)
        return gs

    def host_pinned_bytes(self):
        return sum(g.nbytes for g in self.all_groups())

    def report(self):
        return {
            "n_layer_groups": self.n,
            "layer_group_MiB": round(self.layers[0].nbytes / 2 ** 20, 3) if self.n else 0.0,
            "embed_group_MiB": round(self.embed.nbytes / 2 ** 20, 3) if self.embed else None,
            "head_group_MiB": round(self.head.nbytes / 2 ** 20, 3) if self.head else None,
            "host_pinned_MiB": round(self.host_pinned_bytes() / 2 ** 20, 3),
            "h2d_MiB_total": round(self.st["h2d_bytes"] / 2 ** 20, 2),
            "stagings": self.st["stagings"],
            "blocked_acquires": self.st["blocked_acquires"],
            "fwd_acquires": self.st["fwd_acq"], "bwd_acquires": self.st["bwd_acq"],
            "dev_staged_high_water_MiB": round(self.st["dev_bytes_high_water"] / 2 ** 20, 3),
            "faults_injected": self.st["faults"],
        }

    def reset_stats(self):
        for k in self.st:
            self.st[k] = 0

    # ------------------------------------------------------------------------------- the machinery
    def _stage(self, g):
        """Issue the H2D for `g` on the copy stream.  Non-blocking; no compute-stream ordering."""
        if g.dev is None:
            cs = self.stream
            if _WS["alloc_stream"] == "compute":
                # Reproduces hazard 2 exactly (see the module docstring).  Bisection only.
                dev = torch.empty(g.nbytes, dtype=torch.uint8, device=self.device)
            else:
                with torch.cuda.stream(cs):
                    dev = torch.empty(g.nbytes, dtype=torch.uint8, device=self.device)
            with torch.cuda.stream(cs):
                dev.copy_(g.host, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(cs)
            g.dev = dev
            g.arrived = ev
            g.staged_n += 1
            self.st["stagings"] += 1
            self.st["h2d_bytes"] += g.nbytes
            self.st["dev_bytes_live"] += g.nbytes
            self.st["dev_bytes_high_water"] = max(self.st["dev_bytes_high_water"],
                                                  self.st["dev_bytes_live"])

    def _acquire(self, g):
        """Make `g` readable by the compute stream and point every `p.data` at the slab."""
        if g.dev is None:
            self.st["blocked_acquires"] += 1
            self._stage(g)
        torch.cuda.current_stream(self.device).wait_event(g.arrived)
        views = g.devviews()
        for (p, _o, _nb, _s, _d), v in zip(g.members, views):
            p.data = v
        self._acq += 1
        n = _WS["fault"]
        if n and self._acq % n == 0:
            # On the COMPUTE stream, after the wait -- so every downstream read sees it.  One
            # mantissa bit of one element of the slab.
            g.dev.view(torch.int16)[0] ^= 1
            self.st["faults"] += 1
        return views

    def _release(self, g):
        """Hand the parameters back to the host slab and let the landing slab go.

        `record_stream` is the whole of hazard 3: the slab was written by the copy stream and read
        by the compute stream, and this is what stops a later copy-stream allocation from getting
        the block before those reads have run.
        """
        if g.dev is None:
            return
        for i, (p, _o, _nb, _s, _d) in enumerate(g.members):
            p.data = g.hostviews[i]
        g.dev.record_stream(torch.cuda.current_stream(self.device))
        self.st["dev_bytes_live"] -= g.nbytes
        g.dev = None
        g.arrived = None

    # ------------------------------------------------------------------------- the schedule
    def fwd_enter(self, i):
        """Layer `i` is about to run its forward."""
        self.st["fwd_acq"] += 1
        views = self._acquire(self.layers[i])
        for k in range(1, _WS["lookahead"] + 1):
            if i + k < self.n:
                self._stage(self.layers[i + k])
        return views

    def fwd_exit(self, i, for_backward=True):
        self._release(self.layers[i])
        if i == self.n - 1:
            # The fwd/bwd turn.  Re-issue the LAST layer now (only if a backward is coming) so its
            # transfer has the whole LM head and loss to land in, and stage the head while we are
            # here -- the head is read on BOTH paths, since Liger FLCE takes `lm_head.weight`
            # directly and the stock path calls `lm_head.forward`.
            if for_backward:
                self._stage(self.layers[i])
            if self.head is not None:
                self._acquire(self.head)

    def bwd_enter(self, i):
        self.st["bwd_acq"] += 1
        views = self._acquire(self.layers[i])
        for k in range(1, _WS["lookahead"] + 1):
            if i - k >= 0:
                self._stage(self.layers[i - k])
        return views

    def bwd_exit(self, i):
        self._release(self.layers[i])

    def bwd_split_release(self, i):
        """`bwd='split'`: drop the layer between the recompute and the dgrad."""
        self._release(self.layers[i])

    def bwd_split_acquire(self, i):
        return self._acquire(self.layers[i])

    # --------------------------------------------------------------- embed / head (module hooks)
    def _embed_pre(self, _m, _a):
        if self.embed is not None:
            self._acquire(self.embed)
        if self.n:
            self._stage(self.layers[0])          # so layer 0's forward never blocks
        return None

    def _embed_post(self, _m, _a, out):
        if self.embed is not None:
            self._release(self.embed)
        return None

    def _model_post(self, _m, _a, out):
        """The head's parameters go back to the host here.

        NOT a release of its storage: whatever autograd saved (Liger FLCE's `lm_head_weight`, or
        `nn.Linear`'s saved weight) holds the slab alive until that node executes, and the
        `record_stream` already recorded in `_release` is what orders its eventual free.  Under
        `no_grad` nothing saved it and the slab dies here.
        """
        if self.head is not None:
            self._release(self.head)
        return None

    def install_hooks(self, model, embed_mod):
        if embed_mod is not None:
            self._hooks.append(embed_mod.register_forward_pre_hook(self._embed_pre))
            self._hooks.append(embed_mod.register_forward_hook(self._embed_post))
        self._hooks.append(model.register_forward_hook(self._model_post))

    def uninstall(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []
        for g in self.all_groups():
            if g.host is not None:
                # Leave the parameters where the model can still use them: back on the device.
                try:
                    g.unpin(self.device)
                except Exception:
                    g.unpin(None)
        self.installed = False


# ------------------------------------------------------------------------------------------------
# installation
# ------------------------------------------------------------------------------------------------
def install(model, inner, layers, proj_names, base_of, verbose=False):
    """Build the streamer for a patched model.  Called from `apply_flash_block`.

    `base_of(proj)` returns the `nn.Linear` underneath an adapter wrapper.  A layer whose base
    weights are not ALL frozen is skipped whole rather than partly streamed: partial streaming
    would need `save_for_backward` to be selectively `None`, and a trainable base weight needs its
    own dense gradient, so full fine-tuning simply does not engage this mechanism.
    """
    # Exactly ONE streamer can be installed at a time -- the fused wrapper reaches it through a
    # module global, so a second install while a first model is still alive would point that
    # model's layers at another model's slabs.  Uninstalling first puts the previous model's
    # weights back on the device, which is what a still-live model needs, and returns its ~2 GiB of
    # pinned host memory.  The gate suite builds a fresh patched model per gate, so this fires
    # dozens of times per run.
    reset()

    device = None
    layer_groups, skipped = [], []
    for i, layer in enumerate(layers):
        ws = []
        for j, nm in enumerate(proj_names):
            # `_FB_PROJ_NAMES` order: q k v o on self_attn, then gate up down on mlp -- the same
            # `j < 4` split `_fb_make_forward` uses, so the slab's member order is exactly the
            # order the fused backward unpacks.
            owner = layer.self_attn if j < 4 else layer.mlp
            ws.append(base_of(getattr(owner, nm)).weight)
        if any(w.requires_grad for w in ws):
            skipped.append(i)
            continue
        if device is None:
            device = ws[0].device
        layer_groups.append(_Group(f"layer{i}", ws))
    if skipped or device is None or device.type != "cuda":
        # A trainable base weight needs its own dense gradient and must stay in `save_for_backward`,
        # so this mechanism cannot engage for full fine-tuning.  DECLINE rather than raise, and say
        # so loudly: the model is then exactly the resident-weight model, which is the correct
        # answer, and `_WS["streamer"]` stays None so nothing in the fused path changes.  Silence
        # here would let a full-FT row be reported under a streaming label.
        why = (f"layers {skipped} have TRAINABLE base weights" if skipped
               else f"base weights are on device={device}, not CUDA")
        _WS["last_declined"] = why
        print(f"[fb_wstream] NOT INSTALLED -- {why}. This model runs fully resident.", flush=True)
        return None
    _WS["last_declined"] = None

    emb = getattr(inner, "embed_tokens", None)
    try:
        head = model.get_output_embeddings()
    except Exception:
        head = None
    tied = (emb is not None and head is not None and getattr(head, "weight", None) is not None
            and head.weight.data_ptr() == emb.weight.data_ptr())
    embed_g = head_g = None
    if tied:
        # One tensor read at BOTH ends of the step.  Streaming it on the embedding's schedule would
        # free it before the head reads it, so it stays resident and is reported as resident.
        pass
    else:
        if _WS["embed"] and emb is not None and not emb.weight.requires_grad:
            embed_g = _Group("embed_tokens", [emb.weight])
        if _WS["head"] and head is not None and getattr(head, "weight", None) is not None \
                and not head.weight.requires_grad:
            head_g = _Group("lm_head", [head.weight])

    s = WeightStreamer(device, layer_groups, embed_g, head_g)
    for g in s.all_groups():
        g.pin()
    s.install_hooks(model, emb)
    _WS["streamer"] = s
    if verbose:
        r = s.report()
        print(f"[fb_wstream] streaming {r['n_layer_groups']} layer slabs of "
              f"{r['layer_group_MiB']:.2f} MiB"
              + (f" + embed {r['embed_group_MiB']:.2f}" if embed_g else " (embed RESIDENT)")
              + (f" + head {r['head_group_MiB']:.2f}" if head_g else " (head RESIDENT)")
              + f"; host pinned {r['host_pinned_MiB']:.2f} MiB; lookahead={_WS['lookahead']}, "
                f"bwd={_WS['bwd']}, alloc_stream={_WS['alloc_stream']}", flush=True)
    return s
