"""
Scan Dispatch — inference-default scan selector (FRONT 4) — by Opus 4.8
======================================================================

WIRES the doubling parallel scan as the inference default, with an automatic
CPU fallback to the tight sequential loop.  This is the *deployment* switch the
SCAN_DEPLOYMENT_NOTES.md asked for — distinct from
`parallel_scan_integration.py::use_parallel_scan`, which is a *measurement*
harness.

The measured crossover (analysis/SCAN_DEPLOYMENT_NOTES.md):
  * GPU / MPS : the O(log T)-depth doubling scan `parallel_linear_scan` beats the
    sequential Python loop ~4-7x at the deployment sizes (measured, MPS).
  * CPU       : no parallel hardware to amortise the extra O(log T) passes, so the
    tight sequential loop wins (parallel is 0.2x-0.8x there).
  * Blelloch  : strictly dominated on both devices (its `index_copy` scatter kills
    its O(T)-work advantage); NEVER selected.

The whole switch is one symbol of indirection.  The reference layer
(`SelectiveRapiditySqrtScanLayer.forward`) calls the *module-global*
`sequential_linear_scan` and resolves it AT CALL TIME against
`moebius_scan_transformer_selective`'s namespace.  So rebinding that one global
to `dispatch_linear_scan` reroutes BOTH the causal and the forward+reverse
non-causal code paths, with ZERO edits to the frozen reference file.

The result is machine-precision identical to the sequential reference on every
device (the doubling scan matches the loop to <1e-5 fp32 / ~1e-13 fp64, and the
CPU path IS the loop), so this is safe to leave on for inference.

Reference: Foss 2026, "From Markov Chains to Minkowski Space".
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
REF = os.path.join(REPO, "reference")

# Make both the reference module and the scan implementations importable.
for _p in (REF, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import moebius_scan_transformer_selective as ref  # the frozen reference module
from parallel_scan import parallel_linear_scan, sequential_linear_scan


# ───────────────────────────────────────────────────────────────────────────
# The dispatcher — same signature as sequential_linear_scan / parallel_linear_scan.
# ───────────────────────────────────────────────────────────────────────────

def dispatch_linear_scan(a: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Device-routed scan for z_t = γ_t · z_{t-1} + a_t.  Shapes (B, T, H, D).

    Picks the doubling parallel scan on GPU/MPS (measured win) and the sequential
    loop on CPU (measured win).  Drop-in replacement for the reference
    `sequential_linear_scan`: same signature, same shapes, machine-precision-equal
    output.  Routing keys off where the tensor actually LIVES, so a model moved
    between devices needs no reconfiguration.
    """
    if a.is_cuda or a.is_mps:
        # GPU / MPS: pure slice/cat/mul/add doubling scan wins (measured ~4-7x).
        return parallel_linear_scan(a, gamma)
    # CPU (and any other backend): the tight sequential loop wins (measured).
    return sequential_linear_scan(a, gamma)


# ───────────────────────────────────────────────────────────────────────────
# Deployment switch — rebind the one call-time symbol the reference resolves.
# ───────────────────────────────────────────────────────────────────────────

def enable_parallel_inference():
    """Make the device-routed dispatcher the inference default.

    Rebinds `moebius_scan_transformer_selective.sequential_linear_scan` to the
    dispatcher.  Every SelectiveRapiditySqrtScanLayer.forward call (causal and
    non-causal) then routes through it: doubling on GPU/MPS, sequential on CPU.
    Idempotent and reversible via `disable_parallel_inference`.
    """
    ref.sequential_linear_scan = dispatch_linear_scan


def disable_parallel_inference():
    """Restore the original pure sequential scan as the global symbol."""
    ref.sequential_linear_scan = sequential_linear_scan


class parallel_inference:
    """Context manager form of `enable_parallel_inference`.

    Inside the block the reference layer uses the device-routed dispatcher; on
    exit the original sequential scan is restored, whatever it was on entry.
    """

    def __enter__(self):
        self._orig = ref.sequential_linear_scan
        ref.sequential_linear_scan = dispatch_linear_scan
        return self

    def __exit__(self, *exc):
        ref.sequential_linear_scan = self._orig
        return False


def pick_device() -> str:
    """Auto-detect the best available device: mps > cuda > cpu."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def which_scan(device) -> str:
    """Report which scan the dispatcher would pick for `device` (no compute)."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type in ("cuda", "mps"):
        return "parallel_linear_scan (doubling)"
    return "sequential_linear_scan (loop)"


if __name__ == "__main__":
    dev = pick_device()
    print(f"[scan_dispatch] auto device = {dev}")
    print(f"[scan_dispatch] would dispatch to: {which_scan(dev)}")
    print(f"[scan_dispatch] cpu would dispatch to: {which_scan('cpu')}")
