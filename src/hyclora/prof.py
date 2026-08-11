"""Zero-cost profiling hooks for the vendored HyC-LoRA fused layers.

HyC-LoRA implements a whole decoder layer as a SINGLE `torch.autograd.Function`. That is the
point of the method (it is what lets them control exactly which buffers survive into backward),
but it also means no module-level or autograd-level profiler can see inside it: `nvtx`/
`record_function` regions emitted by PyTorch stop at `FusedLlamaLayerIntraInterFunc`, and every
one of the ~90 tensor ops in the body collapses into one opaque entry.

This module provides `rf(name)`, a `record_function` region that is a genuine no-op unless
`enable()` has been called. The annotations added to the layer bodies therefore do not change
numerics, do not change the algorithm, and cost nothing in normal runs (one module-global bool
test and the return of a preallocated singleton).

Usage:
    from hyclora import prof
    prof.enable()
    with torch.profiler.profile(...) as p:
        ...
"""

import torch

_ENABLED = False


class _NullCtx:
    """A context manager cheaper than contextlib.nullcontext (no generator machinery)."""

    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


_NULL = _NullCtx()

# --- optional allocator tracing -------------------------------------------------------------
# `torch.cuda.memory_allocated()` reads allocator state maintained by the issuing CPU thread and
# does NOT synchronise, so sampling it at region boundaries yields a faithful, cheap timeline of
# the allocator high-water mark in program order. That is what identifies which single op sets
# peak memory -- something a whole-run `max_memory_allocated()` can never tell you.
_MEMTRACE = False
_TRACE = []


class _MemCtx:
    __slots__ = ("name", "before")

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.before = torch.cuda.memory_allocated()
        return None

    def __exit__(self, *exc):
        after = torch.cuda.memory_allocated()
        _TRACE.append((self.name, self.before, after))
        return False


def enable(flag: bool = True):
    global _ENABLED
    _ENABLED = bool(flag)


def enable_memtrace(flag: bool = True):
    global _MEMTRACE
    _MEMTRACE = bool(flag)
    _TRACE.clear()


def get_memtrace():
    return list(_TRACE)


def is_enabled() -> bool:
    return _ENABLED


def rf(name: str):
    """Return a profiler region named `name`, or a no-op ctx if profiling is disabled."""
    if _MEMTRACE:
        return _MemCtx(name)
    if _ENABLED:
        return torch.profiler.record_function(name)
    return _NULL
