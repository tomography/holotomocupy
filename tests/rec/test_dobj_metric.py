#!/usr/bin/env python
"""Unit test for the ||obj^{n+1} - obj^n||^2 convergence metric.

    PYTHONPATH=../../src python test_dobj_metric.py

Rec._apply_step_obj fuses the object update with the two reductions that feed
conv.csv's dobj2 / dobj_rel columns, so that the metric costs no copy of the
previous object and no extra pass over it.  That fusion is the thing worth
testing: the update must still be exactly obj + alpha*eta, and the two norms
must be the norms of the arrays the update produced.

Checked against a plain numpy reference, on the chunked (host obj) path and on
the all-GPU path, since _apply_step_obj branches on which one it is.
"""
import sys

import numpy as np
import cupy as cp

from holotomocupy.chunking import Chunking
from holotomocupy.rec_mpi import Rec

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAILED.append(name)


class _Shim:
    """The two attributes _apply_step_obj actually touches."""
    def __init__(self, chunk):
        self.cl_chunking = Chunking(1 << 22, chunk)
        self.gpu_batch = self.cl_chunking.gpu_batch


def run(host, chunk=3, nz=20, m=8, alpha=-0.37):
    rng = np.random.default_rng(0)
    obj = (rng.standard_normal((nz, m, m)) + 1j*rng.standard_normal((nz, m, m))).astype('complex64')
    eta = (rng.standard_normal((nz, m, m)) + 1j*rng.standard_normal((nz, m, m))).astype('complex64')

    # Reference in float64, from the definition.
    ref_obj = obj.astype('complex128') + alpha*eta.astype('complex128')
    ref_d2  = float(np.sum(np.abs(ref_obj - obj.astype('complex128'))**2))
    ref_o2  = float(np.sum(np.abs(ref_obj)**2))

    v = {'obj': obj.copy() if host else cp.asarray(obj)}
    e = {'obj': eta.copy() if host else cp.asarray(eta)}
    d2, o2 = Rec._apply_step_obj(_Shim(chunk), v, e, alpha)

    got_obj = v['obj'] if host else v['obj'].get()
    tag = "host obj (chunked)" if host else "device obj"
    check(f"{tag}: obj <- obj + alpha*eta",
          np.allclose(got_obj, ref_obj.astype('complex64'), rtol=1e-6, atol=1e-6),
          f"max|diff|={np.abs(got_obj - ref_obj).max():.3e}")
    check(f"{tag}: dobj2 == ||alpha*eta||^2",
          abs(d2 - ref_d2) <= 1e-5*ref_d2, f"{d2:.8e} vs {ref_d2:.8e}")
    check(f"{tag}: ||obj_new||^2",
          abs(o2 - ref_o2) <= 1e-5*ref_o2, f"{o2:.8e} vs {ref_o2:.8e}")


print("1. chunked path -- obj on the host, streamed in chunks")
run(host=True)
print("2. all-GPU path -- obj already resident")
run(host=False)
# A chunk size that does not divide nz, and one that swallows it whole: the
# accumulators must not depend on how the pass was cut up.
print("3. chunk size independence")
for chunk in (1, 7, 64):
    print(f"   chunk={chunk}")
    run(host=True, chunk=chunk)

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(1)
print("\nall checks passed")
