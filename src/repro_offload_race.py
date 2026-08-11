#!/usr/bin/env python
"""Model-free probe for the `fb_offload` corruption -- AND the record of why its first result was
an artifact.  See `llmdocs/trackers/route_b.md` section 5b.

RETRACTION (2026-08-04).  The previous version of this file reported

    barrier=event  0/22 mismatching     barrier=stream  0/22     barrier=sync  19/22

and called that last line "the most important open clue", on the reasoning that
`torch.cuda.synchronize()` subsumes every weaker barrier and therefore cannot produce MORE
staleness.  The reasoning was sound; the measurement was not.  **The reproducer had a
write-after-read bug of its own and was measuring that.**

It compared the landing buffer against a reference and then dropped the buffer inside the same
iteration.  The comparison is only ISSUED there, not executed.  With `torch.cuda.synchronize()` in
the loop, everything the copy stream had pending was already complete, so the caching allocator had
no `record_stream` events left holding the block back and recycled it immediately -- and the NEXT
iteration's H2D, on the copy stream, overwrote the buffer while the previous iteration's comparison
kernel was still reading it.  The weaker barriers looked clean for an equally accidental reason:
they left copy-stream events pending, the allocator declined to recycle at all, and every iteration
got a fresh block.

Holding the landing buffers alive takes every barrier to **0/22** -- `sync` 20/22 -> 0/22 and
`event` 1/22 -> 0/22 -- and the contradiction disappears.  `hold` below is that experiment, and it
is the evidence for this retraction.

WHAT THE REAL DEFECT WAS.  Nothing to do with barriers.  A fresh landing buffer was allocated while
the COMPUTE stream was current, and PyTorch's caching allocator is stream-ordered: it hands a freed
block back only to allocations on the same stream, without any event, because a stream is ordered
with itself.  That returned a block whose previous compute-stream tenant still had kernels in
flight; the H2D wrote `o_h` into it from the copy stream, and the tenant's queued kernels then
wrote their own results on top.  The transfer was never late -- it was overwritten after it landed,
which is why `arrived` was satisfied and why every round-trip verifier said the bytes were correct.

**To reproduce the real defect, use `python src/fb_offload.py compute` (FAIL, 1 of 66 fetches) against
`python src/fb_offload.py copy` (PASS).**  That exercises the shipped staging code rather than a
hand-rolled imitation of it, which is the mistake this file originally made.

Usage:  python src/repro_offload_race.py {event|stream|sync} [hold]
"""
import sys
import torch

barrier = sys.argv[1]
hold = len(sys.argv) > 2 and sys.argv[2] == 'hold'
dev = torch.device('cuda:0'); shape = (2, 1024, 2048); N = 22
cp = torch.cuda.Stream(device=dev)
torch.manual_seed(41)
hosts, refs, evts = [], [], []
for i in range(N):
    o = torch.randn(*shape, device=dev, dtype=torch.bfloat16)
    refs.append(o.clone())
    h = torch.empty(o.numel(), dtype=o.dtype, device='cpu', pin_memory=True)
    cp.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(cp):
        h.copy_(o.reshape(-1), non_blocking=True)
        e = torch.cuda.Event(); e.record(cp)
    o.record_stream(cp)
    hosts.append(h); evts.append(e); del o
    j = torch.empty(shape[0] * shape[1] * shape[2], device=dev, dtype=torch.bfloat16)
    j.fill_(float(i)); del j

bad = torch.zeros((), device=dev, dtype=torch.long)
alive = []
for i in reversed(range(N)):
    d = torch.empty(hosts[i].numel(), dtype=hosts[i].dtype, device=dev)
    cp.wait_event(evts[i])
    with torch.cuda.stream(cp):
        d.copy_(hosts[i], non_blocking=True)
        a = torch.cuda.Event(); a.record(cp)
    d.record_stream(cp)
    if barrier == 'event':    torch.cuda.current_stream(dev).wait_event(a)
    elif barrier == 'stream': torch.cuda.current_stream(dev).wait_stream(cp)
    elif barrier == 'sync':   torch.cuda.synchronize()
    # The comparison below is ISSUED, not executed.  Dropping `d` here is what made this file lie:
    # `hold` keeps every landing buffer alive so no block can be recycled under a comparison that
    # has not run yet.  Compare the two columns before trusting anything else here.
    bad += (d.view(shape) != refs[i]).any().to(torch.long)
    if hold:
        alive.append(d)
    del d
print(f"  barrier={barrier:<7} hold_landing_buffers={hold!s:<5} mismatching layers: {int(bad)}/{N}")
