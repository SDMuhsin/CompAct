"""Pinned-host staging for the fused block's FlashAttention output (`o_h`).

WHAT THIS IS FOR
----------------
`llmdocs/trackers/route_b.md` §2.3 shows that `keep='min'`'s FlashAttention forward recompute is
algebraically irreducible: FA-2's backward needs `O` before it can form any `dS` (because
`D_i = rowsum(dO_i o O_i)` is a reduction over every j-tile), so producing `O` costs two extra GEMM
passes over the O(S^2) tile space no matter how the sweeps are arranged.  §2.4 shows the same
tensor cannot simply be kept in GPU memory -- at seq 16384 that is 128 MiB/layer and the budget
against `liger_gc_sdpa` is 1054 MiB total.

What is left is the third resource: the copy engines, which are idle for the whole step.  This
module moves `o_h` to pinned host memory during the forward and brings it back during the backward,
one layer ahead of where it is needed.

THIS MECHANISM IS NOT OURS -- see `related_work.md` SCOPING PASS 5.
MEMO (arXiv 2407.12117) offloads exactly the FlashAttention output tensor to CPU, chosen by exactly
this reasoning ("recomputing its output is very time-consuming" against 6.25% of the activation
size).  unsloth offloads its checkpoint the same way.  It must be cited in the method section, not
the related-work section.  What is new here is nothing; what it buys is the throughput half of the
§33.2 mandate.

WHY IT SHOULD BE FREE HERE
--------------------------
Measured on this box, not assumed: `unsloth_offload` moves 2688 MiB H2D + 2688 MiB D2H per step at
seq 16384 (`results/hyclora/frontier/pcie_offload_seq16384.json`, kineto `gpu_memcpy` bytes) and
costs +0.001% of step time against `unsloth_gc`, while saving 2304 MiB of peak.  Our traffic at the
same shape would be 5632 MiB -- within 5% of that.  MEMO needs seq >= 192K before their offload
overlaps, but their 8 GPUs share 32 GB/s of host bandwidth (~4 GB/s each) and they move all 16*b*s*h
skeletal activations; a single GPU owns a whole PCIe Gen4 x16 link and we move only 2*b*s*h.

THE CORRECTNESS HAZARDS -- THERE ARE THREE, AND THEY ARE NOT THE SAME ONE
------------------------------------------------------------------------
Every tensor here is touched by two streams, and each direction needs its own ordering.  Getting
two of the three right still corrupts one layer per step, silently, intermittently, and only under
the memory pressure this code exists to create.  `route_b.md` §5b is the full account.

1. THE D2H SOURCE.  `o_h` is produced on the compute stream and read asynchronously by the copy
   stream.  If the allocator hands those bytes to another op before the DMA drains, the copy lands
   garbage.  Covered by `record_stream` on every tensor handed to the copy stream, plus an explicit
   reference held on the handle until the `done` event fires.

2. THE LANDING BUFFER'S ALLOCATION STREAM.  **This is the one that bit us, and no barrier fixes
   it.**  PyTorch's caching allocator is stream-ordered: a freed block goes back only to an
   allocation on the same stream, with no event, because a stream is ordered with itself.
   Allocating a landing buffer with the COMPUTE stream current therefore returns a block whose
   previous compute-stream tenant still has kernels in flight -- the host runs a whole backward
   ahead of the device -- and the H2D then writes it from the copy stream, unordered against them.
   The tenant's kernels land on top of the arrived `o_h`.  The transfer is not late; it is
   overwritten.  Covered by allocating fresh landing buffers on the copy stream (`_Pool.take_dev`).

3. THE LANDING BUFFER'S RECYCLE.  A buffer returning through the pool was READ by the compute
   stream (the FlashAttention backward) and is about to be WRITTEN by the copy stream.  Covered by
   a compute-stream event recorded in `give_dev` and waited on in `prefetch`.

Only end-to-end gradients are ground truth for any of this.  A round-trip verifier, a `.sum()` spy,
or any extra device work at the fetch point can hide hazard 2 -- see `src/probe_offload_grads.py`.
"""
import torch

# One shared state dict, in the style of `_FB_INPLACE_GLU` / `_FB_POLICY` in `flashffn.py`, so the
# off switch and the counters look the same as every other switch in this project.
_FB_OFFLOAD = {"on": False, "stream": None, "device": None, "lookahead": True, "verify": False,
               "barrier": "event", "wait_alloc": False, "alloc_stream": "copy",
               "digest": False, "digests": None,
               "stats": {"stashed": 0, "fetched": 0, "blocked_waits": 0,
                         "bytes_d2h": 0, "bytes_h2d": 0,
                         "verify_ok": 0, "verify_mismatch": 0}}


def fb_offload_barrier(mode):
    """How the COMPUTE stream waits for a landing buffer.  Bisection handle, see `fetch`.

    `'event'`  -- wait only on this handle's own `arrived` event.  This is the shipped mode and the
                  only one that overlaps: the lookahead H2D keeps travelling while the compute
                  stream runs this layer's backward.
    `'stream'` -- `wait_stream` on the whole copy stream.  Correct, but it also waits for the NEXT
                  layer's transfer, which serialises exactly the overlap the mechanism exists for
                  (route_b.md section 4.2 measured that overlap at >= 0.989).  Kept for bisection
                  only; it is not an acceptable shipping configuration.
    """
    if mode not in ("event", "stream"):
        raise ValueError(f"fb_offload: barrier mode must be 'event' or 'stream', got {mode!r}")
    prev = _FB_OFFLOAD["barrier"]
    _FB_OFFLOAD["barrier"] = mode
    return prev


def fb_offload_alloc_stream(mode):
    """Which stream a FRESH landing buffer is allocated on -- `'copy'` (correct) or `'compute'`.

    THIS IS THE ROOT CAUSE SWITCH for route_b.md section 5b.  `'compute'` restores the defect
    exactly, so the bug can be reproduced on demand rather than described; see `_Pool.take_dev` for
    why the allocation stream, and not any barrier, is what decides correctness here.
    """
    if mode not in ("copy", "compute"):
        raise ValueError(f"fb_offload: alloc_stream must be 'copy' or 'compute', got {mode!r}")
    prev = _FB_OFFLOAD["alloc_stream"]
    _FB_OFFLOAD["alloc_stream"] = mode
    return prev


def fb_offload_wait_alloc(on=True):
    """Whether the COPY stream waits for the compute stream before writing a landing buffer.

    OFF by default and correctness does NOT depend on it.  It is a blunt alternative to the actual
    fix (allocating landing buffers on the copy stream -- see `_Pool.take_dev`): it also removes the
    corruption, by making the copy stream wait for every compute kernel issued so far, but it
    couples the two streams for no reason once the allocation stream is right.  Kept as a bisection
    handle and as a second, independent way to demonstrate the same root cause.
    """
    prev = _FB_OFFLOAD["wait_alloc"]
    _FB_OFFLOAD["wait_alloc"] = bool(on)
    return prev


def fb_offload_verify(on=True):
    """Debug: keep a GPU copy at stash time and compare it at fetch time.  Doubles `o_h` memory."""
    prev = _FB_OFFLOAD["verify"]
    _FB_OFFLOAD["verify"] = bool(on)
    return prev


# ------------------------------------------------------------------------------------------------
# A NON-SYNCHRONISING witness on the bytes the backward actually reads.
#
# route_b.md section 5b records that a `torch.equal` round-trip verifier reports 22/22 correct while
# the FlashAttention backward downstream reads a stale buffer: the verifier returns a Python bool,
# which synchronises, and the synchronisation is itself the fix.  So this one never returns a value
# mid-step.  It reduces `o_h` to a single int64 on the device, keeps the forward's value in a device
# tensor, and accumulates disagreements into a device counter.  The only host read is
# `fb_offload_digest_report()`, called once after the step.
#
# The reduction is over `int32` reinterpreted bits, so it is exact for bf16 and cannot be fooled by
# a float sum's associativity the way a `.sum()` witness could.
# ------------------------------------------------------------------------------------------------
def fb_offload_digest(on=True):
    prev = _FB_OFFLOAD["digest"]
    _FB_OFFLOAD["digest"] = bool(on)
    if on and _FB_OFFLOAD["digests"] is None:
        _FB_OFFLOAD["digests"] = {}
    return prev


def _digest(t):
    """Order-independent 64-bit witness over a tensor's raw bytes, computed on the device."""
    b = t.detach().contiguous().reshape(-1).view(torch.int16).to(torch.int64)
    idx = torch.arange(b.numel(), device=b.device, dtype=torch.int64)
    return (b * (idx * 2 + 1)).sum()


def fb_offload_digest_note(h, where):
    """Record (or check) the witness for one handle at a named point in the step."""
    if not _FB_OFFLOAD["digest"] or h is None:
        return
    d = _FB_OFFLOAD["digests"]
    if "counts" not in d:
        d["counts"] = {}
    dev = h.device
    if where == "fwd":
        h.dig = _digest(h.src if h.src is not None else h.ref)
        return
    if getattr(h, "dig", None) is None:
        return
    key = f"mismatch_{where}"
    if key not in d["counts"]:
        d["counts"][key] = torch.zeros((), device=dev, dtype=torch.long)
    got = _digest(h.dev_flat.view(h.shape))
    d["counts"][key] += (got != h.dig).to(torch.long)


def fb_offload_digest_report():
    """The ONE host read.  Call after the step, never inside it."""
    d = _FB_OFFLOAD["digests"]
    if not d or "counts" not in d:
        return {}
    out = {k: int(v) for k, v in d["counts"].items()}
    for v in d["counts"].values():
        v.zero_()
    return out


def fb_offload_lookahead(on=True):
    """Turn the one-layer prefetch off, making every fetch synchronous.

    Exists for bisection: it separates "the round trip is wrong" from "the round trip is right but
    the prefetch pipeline is wrong", which is exactly the split the first wiring attempt needed.
    """
    prev = _FB_OFFLOAD["lookahead"]
    _FB_OFFLOAD["lookahead"] = bool(on)
    return prev


def fb_offload_enable(on=True):
    """Turn `o_h` offloading on/off.  Returns the previous state, so a probe can restore it."""
    prev = _FB_OFFLOAD["on"]
    _FB_OFFLOAD["on"] = bool(on)
    return prev


def fb_offload_enabled():
    return _FB_OFFLOAD["on"]


def fb_offload_stats(reset=False):
    s = dict(_FB_OFFLOAD["stats"])
    if reset:
        for k in _FB_OFFLOAD["stats"]:
            _FB_OFFLOAD["stats"][k] = 0
    return s


def _stream(device):
    """The copy stream, one per process.  Created lazily so importing this module on a CPU-only
    box is harmless."""
    if _FB_OFFLOAD["stream"] is None or _FB_OFFLOAD["device"] != device:
        if device.type != "cuda":
            raise RuntimeError(
                f"fb_offload: the o_h offload needs a CUDA device, got {device}. It refuses "
                f"rather than silently running without offload -- see route_b.md section 2.7.")
        _FB_OFFLOAD["stream"] = torch.cuda.Stream(device=device)
        _FB_OFFLOAD["device"] = device
    return _FB_OFFLOAD["stream"]


class _Pool:
    """Free-lists of pinned host buffers and of GPU landing buffers, keyed by (nbytes, dtype).

    Allocating pinned memory is a synchronising, expensive operation (`cudaHostAlloc`), so it must
    happen once per shape and then be reused for the whole run.  Buffers are returned as flat 1-D
    tensors and viewed by the caller, so one pool entry serves every layer of a given shape.
    """

    def __init__(self):
        self.host = {}
        self.dev = {}

    def take_host(self, numel, dtype):
        key = (numel, dtype)
        free = self.host.setdefault(key, [])
        if free:
            return free.pop()
        return torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)

    def give_host(self, t):
        self.host.setdefault((t.numel(), t.dtype), []).append(t)

    def take_dev(self, numel, dtype, device):
        """Returns `(buffer, free_event_or_None)`.

        The event matters and its absence was a real bug in the first draft of this file.  A GPU
        landing buffer is WRITTEN by the copy stream (the H2D) and READ by the compute stream (the
        layer backward).  When it is recycled, the next H2D would begin writing it on the copy
        stream with nothing ordering that against the previous layer's still-running reads on the
        compute stream -- a cross-stream write-after-read race that would corrupt `o_h` for one
        layer under exactly the deep-pipeline conditions this code exists to create.  So `give_dev`
        records an event on the compute stream and `prefetch` waits on it before reusing.

        (Host buffers need no such event: both their writer (D2H) and their reader (H2D) are the
        single copy stream, which orders them by construction.)
        """
        key = (numel, dtype, device)
        free = self.dev.setdefault(key, [])
        if free:
            return free.pop()
        # A FRESH BLOCK MUST BE ALLOCATED ON THE COPY STREAM.  This is the root cause of the
        # corruption in route_b.md section 5b, and the reason a barrier could mask it but never
        # explain it.
        #
        # PyTorch's caching allocator is STREAM-ORDERED: every block records the stream it was
        # allocated on, and `get_free_block` only ever returns a block to an allocation on that same
        # stream.  That rule is what makes reuse safe without any event -- a stream is ordered with
        # itself -- and it is exactly what breaks here.  Allocating the landing buffer under the
        # compute stream lets the allocator hand back a block whose previous compute-stream tenant
        # still has kernels IN FLIGHT (the host runs a whole backward ahead of the device).  The
        # H2D then writes those bytes from the COPY stream, which nothing orders against them, and
        # the tenant's own queued kernels write their results on top of the arrived `o_h`.  The
        # transfer is not late; it is overwritten after it lands, which is why `arrived` is
        # satisfied, why a round-trip verifier sees correct bytes, and why only the layers whose
        # buffer came from a fresh `torch.empty` are hit.
        #
        # Allocating under the copy stream puts the buffer in the copy stream's pool, so the
        # allocator can only ever hand us blocks whose prior tenant was also the copy stream --
        # already ordered.  Cross-stream reuse then remains only for buffers recycled through
        # `give_dev`, and that is what `free_evt` covers.  Costs nothing and orders nothing, so the
        # overlap this mechanism exists for is untouched.
        if _FB_OFFLOAD["alloc_stream"] == "compute":
            return torch.empty(numel, dtype=dtype, device=device), None   # reproduces the defect
        with torch.cuda.stream(_stream(device)):
            return torch.empty(numel, dtype=dtype, device=device), None

    def give_dev(self, t):
        """Return a landing buffer to the free list, tagged with a compute-stream event.

        The event is the whole protection and it is exact: it is recorded at the moment the layer's
        backward has finished issuing every read of this buffer, and `prefetch` waits on it before
        the next H2D writes it.  That is the write-after-read edge, expressed once, in the only
        place it exists.

        A one-deep COOLDOWN slot used to sit here, holding one buffer back so a recycled buffer had
        always survived an extra layer of backward.  It was added while chasing the corruption of
        route_b.md section 5b, it never fixed it (the cause was the allocation stream -- see
        `take_dev`), and it cost a resident `o_h` per shape: 32 MiB at seq 4096, 128 MiB at 16384.
        Removed once the real cause was found, and the stress test in `probe_offload_grads.py
        --trials` is what says the removal is safe rather than an argument.
        """
        key = (t.numel(), t.dtype, t.device)
        evt = torch.cuda.Event()
        evt.record(torch.cuda.current_stream(t.device))
        self.dev.setdefault(key, []).append((t, evt))


_POOL = _Pool()


class OffloadHandle:
    """One layer's `o_h`, in flight or resident on the host.

    Held on `ctx` rather than passed through `save_for_backward`: it is not an autograd input or
    output, and putting it there would drag version-counter bookkeeping onto a tensor that is not
    one.  `flashffn.py` already does exactly this for `ctx.flash_meta`.
    """

    __slots__ = ("host", "shape", "dtype", "device", "done", "dev_flat", "arrived", "_freed",
                 "ref", "src", "dig")

    def __init__(self, host, shape, dtype, device, done):
        self.host = host              # flat pinned host buffer
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.done = done              # event: the D2H has drained
        self.dev_flat = None          # flat GPU landing buffer, once prefetched
        self.arrived = None           # event: the H2D has drained
        self._freed = False
        self.ref = None               # debug only, see `_FB_OFFLOAD["verify"]`
        self.src = None               # the GPU tensor the D2H reads; held until `done` fires
        self.dig = None               # debug only, see `fb_offload_digest`


# Forward order.  The backward walks it in reverse, which is what makes the one-layer lookahead
# free of any layer-index bookkeeping: when layer i is fetched, the entry BEFORE it in this list is
# the next one the backward will want.
_INFLIGHT = []


# Handles whose source tensor is still held.  Bounded by `_MAX_LIVE_SOURCES`, so the extra
# resident `o_h` is at most that many layers -- the same order as the landing double buffer, and
# already inside the memory budget the mechanism is projected against.
_LIVE_SOURCES = []
_MAX_LIVE_SOURCES = 2


def _drain_sources(device, flush=False):
    """Drop source references whose D2H has completed; block on the oldest if too many are live.

    `Event.query()` is non-blocking, so the common path costs nothing: by the time the next layer's
    forward reaches here, the previous layer's transfer has long since drained.  The bound exists so
    a pathologically slow link cannot let sources accumulate without limit.
    """
    while _LIVE_SOURCES and (flush or _LIVE_SOURCES[0].done.query()):
        _LIVE_SOURCES.pop(0).src = None
    while len(_LIVE_SOURCES) > _MAX_LIVE_SOURCES:
        oldest = _LIVE_SOURCES.pop(0)
        oldest.done.synchronize()
        oldest.src = None


def stash(o_h):
    """Forward side: start `o_h` on its way to the host and return a handle.

    The caller must drop its own reference to `o_h` afterwards; the DMA holds the bytes alive via
    `record_stream` until it has drained.
    """
    device = o_h.device
    stream = _stream(device)
    # `reshape` on a non-contiguous tensor returns a COPY, and it is that copy the DMA reads -- so
    # the copy is what must be marked below, not `o_h`.  Mark both: they are the same object in the
    # contiguous case, and marking a tensor twice is harmless.
    flat = o_h.reshape(-1)
    host = _POOL.take_host(flat.numel(), flat.dtype)

    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        host.copy_(flat, non_blocking=True)
        done = torch.cuda.Event()
        done.record(stream)
    # THE hazard (see the module docstring): without this the allocator may reuse these bytes
    # while the copy is still reading them.
    flat.record_stream(stream)
    o_h.record_stream(stream)

    h = OffloadHandle(host, tuple(o_h.shape), o_h.dtype, device, done)
    # Hold the source alive until its D2H has demonstrably completed, rather than trusting
    # `record_stream` to defer the allocator.  This is the correct pattern independently of the
    # open bug below, but be clear about what it did and did not buy:
    #
    #   IT DID NOT FIX THE CORRUPTION.  Gradients with the offload on are still rel-L2 2.35e+00
    #   against a 1.32e-02 control floor with this in place.
    #
    # It was added on the hypothesis that the D2H was reading recycled bytes, because running the
    # whole thing under `PYTORCH_NO_CUDA_MEMORY_CACHING=1` IS correct (1.456e-02 against a 1.366e-02
    # control).  That inference was wrong: the env var turns every free into a synchronising
    # `cudaFree`, so it masks *any* race anywhere, not specifically this one.  It remains the most
    # useful bisect handle available -- see route_b.md section 5b.
    h.src = o_h
    fb_offload_digest_note(h, "fwd")
    _drain_sources(device)
    if _FB_OFFLOAD["verify"]:
        # Debug only: keep a GPU-resident copy of what was handed over, so `fetch` can prove the
        # round trip byte-for-byte in situ.  Doubles GPU memory for `o_h` -- never leave it on.
        h.ref = o_h.detach().clone()
    _INFLIGHT.append(h)
    _LIVE_SOURCES.append(h)
    _FB_OFFLOAD["stats"]["stashed"] += 1
    _FB_OFFLOAD["stats"]["bytes_d2h"] += flat.numel() * flat.element_size()
    return h


def prefetch(h):
    """Issue the H2D for one handle.  Idempotent, and a no-op once already in flight."""
    if h is None or h.dev_flat is not None or h._freed:
        return
    stream = _stream(h.device)
    dev_flat, free_evt = _POOL.take_dev(h.host.numel(), h.dtype, h.device)
    # The D2H must have drained before the same buffer is read back -- normally long past, but the
    # ordering has to be expressed, not assumed.
    stream.wait_event(h.done)
    if free_evt is not None:
        # ...and the previous consumer of this recycled landing buffer must have finished reading
        # it on the compute stream.  See `_Pool.take_dev`.
        stream.wait_event(free_evt)
    if _FB_OFFLOAD["wait_alloc"]:
        # THE FIX (route_b.md section 5b).  `free_evt` covers a buffer that came back through OUR
        # pool.  It does not cover the other way a landing buffer is obtained: a fresh
        # `torch.empty`, which the caching allocator satisfies by RECYCLING A BLOCK THAT WAS JUST
        # FREED ON THE COMPUTE STREAM -- an `o_h`, a `grad_*`, any of the forward's dead
        # intermediates.  The allocator's own rule makes that safe for the compute stream (a stream
        # is ordered with itself), and it is exactly wrong here, because the next thing that touches
        # those bytes is an H2D on the COPY stream, which nothing orders against the compute-stream
        # reads still in flight.  That is a cross-stream write-after-read on the landing buffer, and
        # it corrupts the layer whose `o_h` occupies the recycled block -- not the layer being
        # prefetched.  It is invisible to any round-trip verifier, because the bytes ARE delivered
        # correctly; they are delivered on top of somebody else's live data.
        #
        # `wait_stream` here does NOT cost the overlap.  It makes the copy stream wait for compute
        # work issued UP TO NOW, and this call site is the top of layer i's backward, so it waits
        # for layer i+1's backward and then has the whole of layer i's backward (~190 ms at seq
        # 16384) to move 5.6 ms of `o_h` for layer i-1.  Measured in `--overlap` below.
        stream.wait_stream(torch.cuda.current_stream(h.device))
    with torch.cuda.stream(stream):
        dev_flat.copy_(h.host, non_blocking=True)
        arrived = torch.cuda.Event()
        arrived.record(stream)
    dev_flat.record_stream(stream)
    h.dev_flat = dev_flat
    h.arrived = arrived
    _FB_OFFLOAD["stats"]["bytes_h2d"] += h.host.numel() * h.host.element_size()


def fetch(h):
    """Backward side: return `o_h` on the GPU, and start the NEXT layer's transfer.

    The lookahead is the whole point: issuing the previous entry's H2D here means it travels during
    this layer's backward (~190 ms at seq 16384) rather than being waited on.
    """
    if h._freed:
        raise RuntimeError("fb_offload: handle fetched twice. A second backward through the same "
                           "graph (retain_graph) is not supported by the offload path.")
    # One layer of lookahead, found positionally -- the backward walks `_INFLIGHT` in reverse.
    if h.dev_flat is None:                       # nobody prefetched us: synchronous fallback
        _FB_OFFLOAD["stats"]["blocked_waits"] += 1
        prefetch(h)

    if _FB_OFFLOAD["lookahead"]:
        # Layers run in reverse, so the entry before this one is the next block the backward reaches.
        try:
            i = _INFLIGHT.index(h)
        except ValueError:
            i = -1
        if i > 0:
            prefetch(_INFLIGHT[i - 1])

    # THE BARRIER, AND IT MUST COME AFTER THE LOOKAHEAD so the lookahead's H2D is already issued.
    #
    # `'event'` waits on THIS handle's transfer only, so the layer-ahead H2D keeps travelling while
    # the compute stream runs this layer's backward.  That is the whole mechanism.  `'stream'`
    # waits for the entire copy stream, which also waits for the next layer's transfer and
    # therefore serialises it; it is a bisection handle, not a shipping configuration.
    #
    # HISTORY, because this line was blamed for the corruption and it was not the cause.  Four
    # orderings were tried here and three were recorded as CORRUPT at ~1.75e+00 against a ~1.5e-02
    # control.  The real defect was one level down, in `prefetch`: the copy stream began writing a
    # freshly-allocated landing buffer with nothing ordering it against the compute-stream reads
    # still outstanding on the block the allocator had just recycled (see `_Pool.take_dev`).
    # `'stream'` masked it, which is why it read as "the only correct ordering" -- waiting on the
    # whole copy stream at every fetch keeps compute from running far enough ahead for the window
    # to open.  With the allocation stream right, `'event'` is correct; measured, not assumed, by
    # `src/probe_offload_grads.py --trials` (0/19 at seq 1024, 4096 and 8192, against a defect that
    # fires 29/29 in the same test).
    #
    # Only end-to-end gradients are ground truth here, compared against an offload-OFF vs OFF
    # control: the FA backward is nondeterministic at ~1.5e-02 so a bitwise test is meaningless, and
    # any synchronising in-situ verifier (`torch.equal`, a `.sum()` spy) reports success because the
    # synchronisation is itself a fix.  `fb_offload_digest()` exists for that reason.
    if _FB_OFFLOAD["barrier"] == "stream":
        torch.cuda.current_stream(h.device).wait_stream(_stream(h.device))
    else:
        torch.cuda.current_stream(h.device).wait_event(h.arrived)
    fb_offload_digest_note(h, "at_fetch")
    # The landing buffer belongs to the copy stream's allocator pool (see `_Pool.take_dev`) and is
    # about to be read by the compute stream, so the compute-stream use has to be recorded or the
    # allocator could hand the block out again -- on the copy stream -- while the FlashAttention
    # backward is still reading it.  In steady state the pool never frees, so this matters at
    # `reset()` and on the paths that drop a pool; it costs one allocator bookkeeping call.
    h.dev_flat.record_stream(torch.cuda.current_stream(h.device))
    out = h.dev_flat.view(h.shape)
    if h.ref is not None:
        key = "verify_ok" if torch.equal(out, h.ref) else "verify_mismatch"
        _FB_OFFLOAD["stats"][key] += 1
        h.ref = None
    _FB_OFFLOAD["stats"]["fetched"] += 1
    return out


def release(h):
    """Return both buffers to the pools once the backward has finished with them."""
    if h is None or h._freed:
        return
    h.ref = None
    h.src = None
    if h in _LIVE_SOURCES:
        _LIVE_SOURCES.remove(h)
    if h.dev_flat is not None:
        _POOL.give_dev(h.dev_flat)          # records the compute-stream free event
        h.dev_flat = None
    _POOL.give_host(h.host)
    h.host = None
    h._freed = True
    try:
        _INFLIGHT.remove(h)
    except ValueError:
        pass


def reset():
    """Drop every in-flight handle AND the buffer pools.  Between measurement arms, and after an
    exception has unwound a backward halfway through.

    The pools must go too.  They are process-global, so a landing buffer left over from an
    offloaded arm stays resident while the NEXT arm is measured and lands in its peak: measured, at
    seq 4096, as a 64.00 MiB inflation of both `fb_attn` and `fb_min` when they followed
    `fb_attn_offload` in one process -- exactly two landing buffers, and enough to move a
    published column.  Freeing them here is why `profile_hyclora.apply_family` calls this.
    """
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    for h in list(_INFLIGHT):
        release(h)
    _INFLIGHT.clear()
    for h in _LIVE_SOURCES:
        h.src = None
    _LIVE_SOURCES.clear()
    _POOL.dev.clear()
    _POOL.host.clear()


# ---------------------------------------------------------------------------------------------
# Self-test.  `PYTHONPATH=src python src/fb_offload.py`
#
# It is deliberately adversarial about the one thing that can go wrong here (section: THE
# CORRECTNESS HAZARD).  A round trip that only ever runs on an idle allocator will pass even when
# `record_stream` and the recycle event are both missing, because nothing is competing for the
# bytes.  So the test churns large allocations between the stash and the fetch, forcing the caching
# allocator to reuse freed blocks, and it reuses the pool across several rounds so the landing
# buffers are genuinely recycled while a previous consumer is still reading them.
# ---------------------------------------------------------------------------------------------
def _self_test(n_layers=22, shape=(2, 1024, 2048), rounds=3, device="cuda:0"):
    """Reproduce the round trip WITHOUT ever synchronising inside the loop.

    The first version of this test compared each fetch with `torch.equal`, which returns a Python
    bool and therefore synchronises.  It passed 66/66 while the real model was corrupting one layer
    per step -- because the synchronisation is itself the fix.  So every comparison here accumulates
    into a DEVICE tensor and the only host read happens once, at the very end.

    It also churns large allocations between stash and fetch, so the caching allocator is forced to
    recycle blocks the way it does in a real forward.
    """
    dev = torch.device(device)
    torch.manual_seed(41)
    bad = torch.zeros((), device=dev, dtype=torch.long)
    n_elem = shape[0] * shape[1] * shape[2]
    for rnd in range(rounds):
        refs, handles = [], []
        for i in range(n_layers):                                   # "forward"
            o_h = torch.randn(*shape, device=dev, dtype=torch.bfloat16)
            refs.append(o_h.clone())
            handles.append(stash(o_h))
            del o_h
            junk = torch.empty(n_elem, device=dev, dtype=torch.bfloat16)
            junk.fill_(float(i))
            del junk
        for i in reversed(range(n_layers)):                         # "backward"
            got = fetch(handles[i])
            bad += (got != refs[i]).any().to(torch.long)            # stays on device
            work = (got.float() * 1.000001).sum()                   # a consumer, like a real layer
            release(handles[i])
            del got, work
        del refs, handles
    n_bad = int(bad)                                                # the ONE host read
    st = fb_offload_stats()
    print(f"  stashed {st['stashed']}  fetched {st['fetched']}  "
          f"blocked_waits {st['blocked_waits']}  "
          f"D2H {st['bytes_d2h'] / 2**20:.0f} MiB  H2D {st['bytes_h2d'] / 2**20:.0f} MiB")
    msg = ('PASS -- every round trip bitwise identical' if n_bad == 0
           else f'FAIL -- {n_bad} of {st["fetched"]} fetches wrong')
    print(f'  {msg}')
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    import sys
    if not torch.cuda.is_available():
        raise SystemExit("fb_offload self-test needs a CUDA device")
    # `python src/fb_offload.py compute` restores the route_b.md section 5b defect, so the test can
    # be shown to have teeth rather than asserted to.  Anything else runs the shipped path.
    mode = sys.argv[1] if len(sys.argv) > 1 else "copy"
    fb_offload_alloc_stream(mode)
    print(f"  alloc_stream={mode}")
    sys.exit(_self_test())
