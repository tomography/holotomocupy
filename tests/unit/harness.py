"""A two-function test harness: no pytest, no new dependency.

Tests are plain `test_*` functions that call `check(...)`.  `run(globals())`
executes every one of them in definition order and returns an exit code, so
the files are runnable scripts; pytest collects them too, if it is installed.
"""
import sys
import traceback

import numpy as np

_RESULTS = []


def check(name, ok, detail=''):
    _RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f'   {detail}' if detail else ''),
          flush=True)
    assert ok, f'{name}: {detail}'          # so pytest sees the failure too


def close(name, got, want, rtol=1e-5, atol=0.0):
    """Relative check with the numbers in the message, which is what you read."""
    err = abs(got - want) / max(abs(want), 1e-30) if want or atol == 0 else abs(got - want)
    check(name, abs(got - want) <= atol + rtol * abs(want),
          f'{got:.8g} vs {want:.8g}   rel {err:.2e}')
    return err


def order(h, e, floor=0.0):
    """Fitted convergence order of errors `e` at step sizes `h` (log-log slope).

    Points at or below `floor` are dropped: once the residual reaches float32
    rounding it stops falling, and fitting through it reads as a low order
    when the derivative is in fact exact.  Returns inf when fewer than two
    points survive -- nothing is left to measure, and that is a pass.
    """
    h, e = np.asarray(h, float), np.asarray(np.abs(e), float)
    ok = e > max(floor, 0.0)
    if ok.sum() < 2:
        return np.inf
    return float(np.polyfit(np.log(h[ok]), np.log(e[ok]), 1)[0])


def run(ns, argv=None):
    """Run every test_* in `ns`; return 0 if all passed."""
    names = [k for k, v in ns.items() if k.startswith('test_') and callable(v)]
    want = (argv or sys.argv[1:])
    if want:
        names = [k for k in names if any(w in k for w in want)]
    for k in names:
        print(f'\n--- {k}', flush=True)
        try:
            ns[k]()
        except AssertionError:
            pass                            # check() already reported it
        except Exception:
            _RESULTS.append((k, False))
            print(f'[FAIL] {k}   raised:', flush=True)
            traceback.print_exc()
    bad = [n for n, ok in _RESULTS if not ok]
    print(f'\n{len(_RESULTS) - len(bad)}/{len(_RESULTS)} checks passed'
          + (f'; FAILED: {", ".join(bad)}' if bad else ''))
    return 1 if bad else 0
