"""Figures for nelder_mead.pdf.  Run from this directory."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'figure.dpi': 200})
CM = dict(reflection='tab:blue', expansion='tab:green',
          **{'outside contraction': 'tab:orange',
             'inside contraction': 'tab:red', 'shrink': 'tab:purple',
             'converged': '0.4'})

# ---------------------------------------------------------------- figure 1 --
# the five moves, on one triangle, in the plane.  The triangle is laid out so
# that the line x_h -> c runs left to right: every trial point lives on it.
B, G_, W = np.array([0.0, 0.0]), np.array([1.6, 1.9]), np.array([-1.2, 1.1])
c = (B + G_) / 2
moves = [('inside contraction\n$c-\\frac{1}{2}(c-x_h)$', -0.5, 'tab:red', +1),
         ('outside contraction\n$c+\\frac{1}{2}(c-x_h)$', 0.5, 'tab:orange', +1),
         ('reflection\n$x_r=c+(c-x_h)$', 1.0, 'tab:blue', -1),
         ('expansion\n$x_e=c+2(c-x_h)$', 2.0, 'tab:green', -1)]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.7, 1]))
for a in ax:
    a.add_patch(Polygon(np.array([B, G_, W]), closed=True, fc='0.92', ec='0.5', lw=1))
    for p, lab, dx, dy, ha in ((B, '$x_l$ best', 0.05, -0.28, 'center'),
                               (G_, '$x_s$', 0.12, -0.05, 'left'),
                               (W, '$x_h$ worst', -0.12, 0.0, 'right')):
        a.plot(*p, 'ko', ms=4)
        a.text(p[0] + dx, p[1] + dy, lab, ha=ha, va='center', fontsize=8)
    a.plot(*c, 'k+', ms=9, mew=1.5)
    a.set_aspect('equal')
    a.axis('off')
for a in ax:
    a.text(c[0] - 0.10, c[1] - 0.26, '$c$', ha='right', va='top', fontsize=8)
d = c - W
ax[0].plot(*np.array([W, c + 2.25 * d]).T, '-', color='0.7', lw=0.8, zorder=0)
for lab, t, col, side in moves:
    p = c + t * d
    ax[0].plot(*p, 'o', color=col, ms=5)
    ax[0].annotate('', xy=p, xytext=W,
                   arrowprops=dict(arrowstyle='->', color=col, lw=1.1, alpha=0.75,
                                   shrinkA=4, shrinkB=5,
                                   connectionstyle=f'arc3,rad={-0.16 * side}'))
    ax[0].text(p[0], p[1] + side * 0.42, lab, color=col, ha='center',
               va='bottom' if side > 0 else 'top', fontsize=7)
ax[0].set_title('the four trial points: all on the ray $x_h \\to c$', fontsize=9)
ax[0].set_xlim(-1.9, 5.3); ax[0].set_ylim(-0.85, 2.5)
Vs = np.array([B, B + 0.5 * (G_ - B), B + 0.5 * (W - B)])
ax[1].add_patch(Polygon(Vs, closed=True, fc='none', ec='tab:purple', lw=1.6))
for p, q in ((G_, Vs[1]), (W, Vs[2])):
    ax[1].annotate('', xy=q, xytext=p,
                   arrowprops=dict(arrowstyle='->', color='tab:purple', lw=1.3,
                                   shrinkA=4, shrinkB=2))
ax[1].set_title('shrink: every other vertex halves\nits distance to $x_l$ '
                '($n$ evaluations)', color='tab:purple', fontsize=9)
ax[1].set_xlim(-1.9, 2.3); ax[1].set_ylim(-0.85, 2.5)
fig.tight_layout()
fig.savefig('fig_moves.pdf')
plt.close(fig)

# ---------------------------------------------------------------- figure 2 --
# the real objective on a plane, and the simplex crossing it
z = np.load('nm_landscape.npz')
Z, gx, gy, R = z['Z'], z['gx'], z['gy'], float(z['R'])
Vh, Fh, mv = z['Vh'], z['Fh'], z['moves']
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3),
                       gridspec_kw=dict(width_ratios=[1.25, 1]))
a = ax[0]
im = a.contourf(gx, gy, Z, levels=40, cmap='viridis')
a.contour(gx, gy, Z, levels=14, colors='w', linewidths=0.3, alpha=0.6)
plt.colorbar(im, ax=a, label='entropy (nats)')
for i, V in enumerate(Vh):
    a.add_patch(Polygon(V, closed=True, fc='none', lw=0.9,
                        ec=CM.get(str(mv[i]), '0.4'), alpha=0.85))
a.plot(0, 0, 'w*', ms=9, mec='k', label='start (no correction)')
a.plot(R, 0, 'wo', ms=6, mec='k', label='best degree-2 shifts')
a.plot(*Vh[-1][np.argmin(Fh[-1])], 'rx', ms=8, mew=2, label='where it stops')
a.set(xlabel='px rms along the answer', ylabel='px rms across it',
      title='31 simplices on the real objective\n(a 2-D slice of the degree-2 stage)')
a.legend(fontsize=6.5, loc='upper left')
a = ax[1]
names, counts = np.unique(mv[mv != 'converged'], return_counts=True)
a.barh(names, counts, color=[CM[k] for k in names])
for k, v in zip(names, counts):
    a.text(v + 0.2, k, str(v), va='center', fontsize=8)
a.set(xlabel='times used in those 31 iterations',
      title='which move was taken\n(30 moves, 62 evaluations)')
a.tick_params(labelsize=8)
a.set_xlim(0, counts.max() * 1.25)
fig.tight_layout()
fig.savefig('fig_landscape.pdf')
plt.close(fig)

# ---------------------------------------------------------------- figure 3 --
# the real 9-D run, as a ladder of stages
t = np.loadtxt('simple_n384_th192_nz16_trace.csv', delimiter=',', skiprows=1)
ev = np.arange(len(t))
# stage boundaries read off the trace itself: during stage k the coefficients
# above degree k are still exactly zero
nzd = (np.abs(t[:, 2:].reshape(len(t), 6, 2)) > 0).any(axis=2)
deg = np.where(nzd.any(1), 5 - np.argmax(nzd[:, ::-1], axis=1), 0)
bnd = np.r_[0, np.flatnonzero(np.diff(deg))[1:] + 1, len(t)]   # drop the 0->1 row
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
ax[0].plot(ev, t[:, 0], '.', ms=1.5, color='0.75', label='every evaluation')
ax[0].plot(ev, np.minimum.accumulate(t[:, 0]), 'tab:red', lw=1.4, label='best so far')
ax[1].plot(ev, t[:, 1], '.', ms=1.5, color='0.75', label='every evaluation')
# the pixel error of whichever point is best *by entropy* so far -- this is the
# only curve the search could actually deliver if stopped at that evaluation
arg = np.zeros(len(t), dtype=int)
for i in range(1, len(t)):
    arg[i] = i if t[i, 0] < t[arg[i - 1], 0] else arg[i - 1]
ax[1].plot(ev, t[arg, 1], 'tab:blue', lw=1.4, label='best by entropy so far')
for a in ax:
    for k in range(5):
        if k:
            a.axvline(bnd[k], color='0.6', lw=0.7, ls=':')
        x, lo, hi = 0.5 * (bnd[k] + bnd[k + 1]), bnd[k], bnd[k + 1]
        a.annotate(f'deg {k + 1} ({hi - lo})',
                   xy=(x, 1.02 + 0.115 * (k % 3)), xycoords=('data', 'axes fraction'),
                   fontsize=6.5, color='0.3', annotation_clip=False,
                   ha='left' if k == 0 else 'center', va='bottom')
ax[0].axhline(4.77979, color='k', lw=0.8, ls='--')
ax[0].text(len(t), 4.78015, 'entropy at the true shifts ', ha='right', va='bottom',
           fontsize=7)
ax[0].set(xlabel='objective evaluation', ylabel='entropy (nats)')
ax[1].set(xlabel='objective evaluation', ylabel='identifiable error (px rms)')
for a in ax:
    a.grid(alpha=0.3)
    a.legend(fontsize=7)
fig.tight_layout(rect=(0, 0, 1, 0.87))
fig.savefig('fig_ladder.pdf')
plt.close(fig)
print('wrote fig_moves.pdf fig_landscape.pdf fig_ladder.pdf')
