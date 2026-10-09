import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/local/ssd/vnikitin/convbin/'

def load(tag):
    return {lv: np.genfromtxt(f'{R}{tag}.conv_bin{lv}.csv', delimiter=',', names=True)
            for lv in (2, 1)}

#  tag,               label,              colour,    style, in the clean 2x2?
ARMS = [
    ('psf00_binned',     'psf00_binned       intensity  $\\sigma$=0      lam 4e-4/2e-4', '#1f77b4', '-',  True),
    ('psf10_binned',     'psf10_binned       intensity  $\\sigma$=1.0  lam 4e-4/2e-4', '#1f77b4', '--', True),
    ('psf00_amp',        'psf00_amp[b2,b1]   amplitude  $\\sigma$=0      lam 1e-4/5e-5', '#d62728', '-',  True),
    ('psf10_amp',        'psf10_amp[b2,b1]   amplitude  $\\sigma$=1.0  lam 1e-4/5e-5', '#d62728', '--', True),
    ('psf10_amp_binned', 'psf10_amp_binned   amplitude  $\\sigma$=1.0  lam 1e-4/0    ', '#7f7f7f', ':',  False),
]
D = {t: load(t) for t, *_ in ARMS}

def series(d):
    """iteration, err, dobj_rel and CUMULATIVE wall clock across both levels.

    `time` in the CSV is seconds for that 32-iteration block, not a running
    total, so each level is cumsum'd and bin 1 is offset by bin 2's total.
    Both levels carry an iter=-1 seed row (time 0) and bin 1 additionally
    re-evaluates the bin-2 result on the finer grid at iter=1024; that row is
    not a new iteration, so it is dropped from the curves but its time is
    kept in the clock."""
    b2, b1 = d[2], d[1]
    m2 = b2['iter'] >= 0
    t2 = np.cumsum(b2['time'][m2])
    t1 = np.cumsum(b1['time'][b1['iter'] >= 0])[1:] + t2[-1]   # [1:] drops iter 1024
    m1 = b1['iter'] > 1024
    return (np.concatenate([b2['iter'][m2],      b1['iter'][m1]]),
            np.concatenate([b2['err'][m2],       b1['err'][m1]]),
            np.concatenate([t2, t1]),
            np.concatenate([b2['dobj_rel'][m2],  b1['dobj_rel'][m1]]))

fig = plt.figure(figsize=(15.5, 16.0))
gs = fig.add_gridspec(3, 2, hspace=.30, wspace=.20)
axA, axB, axC, axD, axE, axF = (fig.add_subplot(gs[i, j])
    for i, j in ((0,0),(0,1),(1,0),(1,1),(2,0),(2,1)))

rows = []
for tag, lab, c, ls, clean in ARMS:
    it, er, tt, dv = series(D[tag])
    lw = 1.3 if not clean else 1.9
    al = .75 if not clean else 1.0
    kw = dict(color=c, ls=ls, lw=lw, alpha=al, label=lab)
    ref = er[it == 32][0]
    axA.semilogy(it, er, **kw)
    axB.semilogy(it, er/ref, **kw)
    ok = np.isfinite(dv)
    axC.semilogy(it[ok], dv[ok], **kw)
    axD.semilogy(tt/3600.0, er/ref, **kw)
    b1 = it >= 1056
    axE.plot(it[b1], er[b1]/er[it == 1056][0], **kw)
    blk = np.diff(np.concatenate([[0.0], tt]))
    axF.plot(it, blk, **kw)
    g = lambda n: er[it == n][0]
    rows.append((lab.split()[0], g(0), g(32), g(1024), g(32)/g(1024),
                 g(1056), g(1280), g(1056)/g(1280), tt[-1]/3600.0, dv[ok][-1]))

for ax, t, yl, xl in [
        (axA, 'A.  err as recorded  --  levels NOT comparable across the model axis', 'err', 'cumulative iteration'),
        (axB, 'B.  err / err(iter 32)  --  each arm against itself', 'relative err', 'cumulative iteration'),
        (axC, 'C.  dobj_rel  --  relative object change per 32-iteration block', 'dobj_rel', 'cumulative iteration'),
        (axD, 'D.  relative err vs wall clock, 24 nodes', 'relative err', 'wall clock [h]'),
        (axE, 'E.  bin 1 only, err / err(iter 1056)  --  where the arms separate', 'relative err', 'cumulative iteration'),
        (axF, 'F.  seconds per 32-iteration block', 'seconds', 'cumulative iteration')]:
    if ax not in (axD, axE):
        ax.axvline(1024, color='k', lw=.9, alpha=.45)
    ax.set_title(t, fontsize=10.5, loc='left')
    ax.set_xlabel(xl, fontsize=9.5); ax.set_ylabel(yl, fontsize=9.5)
    ax.grid(alpha=.3, which='both'); ax.tick_params(labelsize=8.5)
axA.legend(fontsize=8, loc='upper right', framealpha=.93, prop={'family': 'monospace', 'size': 7.6})
axA.annotate('bin 2 $\\rightarrow$ bin 1\n(err jumps: finer grid,\nsame object)', xy=(1024, axA.get_ylim()[0]*40),
             xytext=(640, axA.get_ylim()[0]*40), fontsize=8, color='#444444',
             arrowprops=dict(arrowstyle='->', color='#666666', lw=.9))

fig.suptitle(
    'ctxl_HT_4K_RD300_007p5nm  --  convergence of the truncated bin2$\\rightarrow$bin1 arms, iterations 0..1280\n'
    'The clean 2x2 is the four solid/dashed curves.  psf00_amp / psf10_amp are the bin-2 and bin-1 rungs of the full\n'
    'three-level amplitude arms (lam 1e-4/5e-5 = the intensity ladder / 4, as the model rescaling requires); a separate\n'
    'psf00_amp_binned was never built.  psf10_amp_binned (grey, lam=0 at bin 1) is a different regulariser, not part of the square.',
    fontsize=10.5, y=.985)
axE.grid(alpha=.3, which='both'); axE.legend(fontsize=7.6, prop={'family':'monospace','size':7.6})
plt.savefig(R+'fig_conv_binned.png', dpi=110, bbox_inches='tight')

h = (f'{"arm":<17} {"err@0":>10} {"err@32":>10} {"err@1024":>10} {"b2 drop":>8} '
     f'{"err@1056":>10} {"err@1280":>10} {"b1 drop":>8} {"hours":>6} {"final dobj_rel":>14}')
print(h); print('-'*len(h))
for r in rows:
    print(f'{r[0]:<17} {r[1]:>10.3e} {r[2]:>10.3e} {r[3]:>10.3e} {r[4]:>7.2f}x '
          f'{r[5]:>10.3e} {r[6]:>10.3e} {r[7]:>7.3f}x {r[8]:>6.2f} {r[9]:>14.2e}')
print()
key = {r[0]: r for r in rows}
for a, b, tag in (('psf00_binned', 'psf10_binned', 'intensity'),
                  ('psf00_amp[b2,b1]', 'psf10_amp[b2,b1]', 'amplitude')):
    ra, rb = key[a], key[b]
    print(f'psf axis, {tag}: psf10 vs psf00  '
          f'err@1024 {rb[3]/ra[3]-1:+.2%},  err@1280 {rb[6]/ra[6]-1:+.2%},  '
          f'wall clock {rb[8]/ra[8]-1:+.1%}')
print()
for a, b, tag in (('psf00_binned', 'psf00_amp[b2,b1]', 'sigma=0'),
                  ('psf10_binned', 'psf10_amp[b2,b1]', 'sigma=1.0')):
    ra, rb = key[a], key[b]
    print(f'model axis, {tag:9s}: amplitude vs intensity  '
          f'err level {rb[6]/ra[6]:.3f}x (different units, not a quality ratio),  '
          f'b2 drop {ra[4]:.2f}x -> {rb[4]:.2f}x,  wall clock {rb[8]/ra[8]-1:+.1%}')
