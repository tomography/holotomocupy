#!/usr/bin/env python
"""
Parse the log produced by rec_iterative_mpi_syn.py and print:
  * max process memory + max GPU memory across the whole run
  * per-function timing breakdown for the requested BH iter (default: the last)
  * grouped category totals (gradient / hessian / redist / linear_batch / min /
    allreduce), with the hessian bucket broken down per hessian() invocation
    when the solver runs the classic three-sweep path

`@timer`-decorated methods log a line of the form
    [rank=0] FUNCNAME: 0.1234 sec, process memory 1.23 GB, GPU memory 4.56 GB
and `error_debug` emits `[rank=0] iter=N: ...sec err=...` (or `Initial err=...`
for the pre-loop warmup). Timer lines emitted between consecutive iter= markers
belong to the iter that closes the bucket.
"""

import argparse
import re
import sys
from collections import defaultdict, Counter


# ── regex patterns ──────────────────────────────────────────────────────────
ANSI = re.compile(r'\033\[[0-9;]*m')
TIMER_RE = re.compile(
    r'\[rank=\d+\]\s+(\w+):\s+([\d.]+)\s+sec,\s+process\s+memory\s+([\d.]+)\s+GB,'
    r'\s+GPU\s+memory\s+([\d.]+)\s+GB'
)
# Only error_debug's timing line closes an iter bucket: rec_mpi also logs
# "iter=N: coeff_cache hits=..." / "iter=N: apply_F_cache hits=..." right after
# it, and without the "<float>sec" anchor those would re-open bucket N and
# overwrite it with the (empty) list of calls seen since.
ITER_RE = re.compile(r'\[rank=\d+\]\s+iter=(-?\d+):\s+[\d.]+sec')
INIT_RE = re.compile(r'\[rank=\d+\]\s+Initial\s+err=')
INFO_RE = re.compile(r'\[rank=\d+\]\s+(machine|job|perf-test|BH done):\s*(.+)')


# ── category mapping ────────────────────────────────────────────────────────
# Each iter's @timer calls are bucketed by function name into these groups.
# Names are the @timer'd method names as they appear in the log:
#   *_cascade / *_cascade3  the cascade sweeps in rec_mpi (3 = the fused form,
#                           one sweep yielding B(g,g), B(g,e), B(e,e))
#   gradient / hessian / hessian3   the extra_terms regularizers (Laplacian,
#                           prb fit), which log under their own bare names
# The hessian bucket is additionally split per hessian() invocation at print
# time by split_hessians(); see the categories-rendering loop below.
CATEGORIES = {
    'gradient':     {'gradients_cascade', 'adj_tomo', 'fwd_tomo', 'gradient'},
    'hessian':      {'hessian_cascade', 'hessian_cascade3', 'hessian', 'hessian3'},
    'redist':       {'redist'},
    'linear_batch': {'linear_batch', 'linear_redot_batch'},
    'min':          {'min'},
    'allreduce':    {'allreduce', 'allreduce2', 'allreduce_scalars'},
}


def parse(path):
    """Walk the log; return (max_proc_gb, max_gpu_gb, iters_dict, info_lines).

    iters_dict[N] = list of (funcname, elapsed_sec, proc_mem_gb, gpu_mem_gb)
    where N is the BH iter index (-1 for the pre-loop warmup).
    info_lines is a list of (label, text) tuples for machine/job/perf-test
    metadata, preserved in log order.
    """
    pending = []                       # accumulates calls until next iter= marker
    iters = {}
    info_lines = []
    max_proc = max_gpu = 0.0
    with open(path) as f:
        for raw in f:
            line = ANSI.sub('', raw)
            m = INFO_RE.search(line)
            if m:
                info_lines.append((m.group(1), m.group(2).rstrip()))
                # don't 'continue' — INFO lines never match the patterns below
            m = ITER_RE.search(line)
            if m:
                iters[int(m.group(1))] = pending
                pending = []
                continue
            if INIT_RE.search(line):
                iters[-1] = pending
                pending = []
                continue
            m = TIMER_RE.search(line)
            if m:
                fname = m.group(1)
                t = float(m.group(2)); p = float(m.group(3)); g = float(m.group(4))
                pending.append((fname, t, p, g))
                if p > max_proc: max_proc = p
                if g > max_gpu:  max_gpu  = g
    # Any pending lines after the last iter= belong to no closed bucket — drop
    return max_proc, max_gpu, iters, info_lines


def summarise_iter(calls):
    """Aggregate a list of (fname, t, p, g) tuples by function name."""
    by_fn = defaultdict(lambda: [0, 0.0])       # fn -> [count, total_sec]
    for fn, t, _, _ in calls:
        by_fn[fn][0] += 1
        by_fn[fn][1] += t
    return by_fn


# Inside one BH iter `Rec.hessian()` emits a cascade sweep first, then the
# extra_terms regularizers — in that order, with no other timed calls between
# them. Splitting on the cascade names gives the individual hessian-call
# instances: 3 per iter on the classic path (beta-num, beta-den, alpha), 1 on
# the fused path, where a single hessian_cascade3 yields all three forms.
_HESSIAN_START = {'hessian_cascade', 'hessian_cascade3'}
_HESSIAN_INNER = CATEGORIES['hessian']

def split_hessians(calls):
    instances = []
    current = None
    for fn, t, *_ in calls:
        if fn in _HESSIAN_START:
            current = []
            instances.append(current)
        if current is not None and fn in _HESSIAN_INNER:
            current.append((fn, t))
        elif fn not in _HESSIAN_INNER:
            current = None
    return instances


def report(path, iter_no=None, out=None, show_info=True, show_header=True):
    """Write the summary of `path` to `out`; return a shell-style exit code.

    Split out of main() so a benchmark can append its own summary to the log it
    just wrote (test.py / test_mosaic.py do that) without shelling out.
    show_info=False drops the machine/job block, which is already in the log.
    iter_no=None breaks down the last iter present in the log, which is the
    steady-state one whatever --niter the run used.
    """
    out = sys.stdout if out is None else out
    def p(*a, **kw):
        kw.setdefault('file', out)
        print(*a, **kw)

    try:
        max_proc, max_gpu, iters, info_lines = parse(path)
    except FileNotFoundError:
        print(f"log not found: {path}", file=sys.stderr)
        return 1

    if show_header:
        p(f"=== {path} ===")
    if info_lines and show_info:
        p("── machine / job ───────────────────────────────────────")
        for label, text in info_lines:
            p(f"{label}: {text}")
        p()
    p(f"max process memory: {max_proc:.3f} GB")
    p(f"max GPU memory    : {max_gpu:.3f} GB")
    p(f"iters seen        : {sorted(iters)}")

    if iter_no is None:
        iter_no = max(iters) if iters else 0
    if iter_no not in iters:
        p(f"\niter={iter_no} not in log (available: {sorted(iters)})")
        return 2

    calls = iters[iter_no]
    if not calls:
        p(f"\niter={iter_no}: no timer calls recorded "
          f"(was log_level=DEBUG when the run started?)")
        return 3


    by_fn = summarise_iter(calls)
    total = sum(t for _, t in by_fn.values())

    p(f"\n── iter {iter_no}: per-function summary ────────────────────────")
    p(f"{'function':<24} {'calls':>6} {'total[s]':>12} {'mean[ms]':>12} {'pct':>7}")
    for fn in sorted(by_fn, key=lambda k: -by_fn[k][1]):
        cnt, tot = by_fn[fn]
        mean_ms  = (tot / cnt) * 1e3
        pct      = 100 * tot / total if total else 0
        p(f"{fn:<24} {cnt:>6} {tot:>12.4f} {mean_ms:>12.3f} {pct:>6.1f}%")
    p(f"{'TOTAL (timed)':<24} {sum(c for c, _ in by_fn.values()):>6} {total:>12.4f}")
    min_time = by_fn.get('min', (0, 0.0))[1]
    p(f"{'TOTAL - min':<24} {'':>6} {total - min_time:>12.4f}")

    p(f"\n── iter {iter_no}: grouped categories ──────────────────────────")
    seen = set()
    hessian_instances = split_hessians(calls)
    p(f"{'category':<16} {'calls':>6} {'total[s]':>12} {'pct':>7}  members")
    for cat, members in CATEGORIES.items():
        cnt = sum(c for fn, (c, _) in by_fn.items() if fn in members)
        tot = sum(t for fn, (_, t) in by_fn.items() if fn in members)
        pct = 100 * tot / total if total else 0
        present = sorted(members & set(by_fn))
        seen.update(present)
        p(f"{cat:<16} {cnt:>6} {tot:>12.4f} {pct:>6.1f}%  {present}")
        # On the classic path the bucket is three separate hessian() calls, and
        # which one is expensive matters; break them out under the total. The
        # fused path has a single instance, so the sub-row would just repeat it.
        if cat == 'hessian' and len(hessian_instances) > 1:
            for k, inst in enumerate(hessian_instances, start=1):
                icnt = len(inst)
                itot = sum(t for _, t in inst)
                ipct = 100 * itot / total if total else 0
                iprs = sorted({fn for fn, _ in inst})
                p(f"{'  #' + str(k):<16} {icnt:>6} {itot:>12.4f} {ipct:>6.1f}%  {iprs}")
    other_fns = sorted(set(by_fn) - seen)
    if other_fns:
        cnt = sum(by_fn[fn][0] for fn in other_fns)
        tot = sum(by_fn[fn][1] for fn in other_fns)
        pct = 100 * tot / total if total else 0
        p(f"{'other':<16} {cnt:>6} {tot:>12.4f} {pct:>6.1f}%  {other_fns}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log', nargs='?', default='perf.log',
                    help='path to the perf log file (default: perf.log)')
    ap.add_argument('--iter', type=int, default=None,
                    help='which BH iter to break down (default: the last one in the log)')
    args = ap.parse_args()
    sys.exit(report(args.log, args.iter))


if __name__ == '__main__':
    main()
