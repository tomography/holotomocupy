#!/usr/bin/env python
"""Plot Peter's correct_correct3D.txt drift curves for the four 20260829 scans.

    python plot_correct3d.py [-o out.png]

correct_correct3D.txt is nabu's per-projection sample-motion correction: one row
per frame plus a trailing duplicate row (ntheta+1), two columns.  Column 0 is the
VERTICAL shift, column 1 the HORIZONTAL one, both in 2x2-BINNED detector pixels
-- see the reference_motion.mat pixelsize, 9 nm where the scan is 4.5 nm.  The
right-hand axis of each panel therefore carries the unbinned number, which is
what our configs want.

The FT scans have 12000 frames (4000 angles x 3 lateral positions) and the HT
scans 4000; the curves are smooth across the triplets, i.e. this is a drift in
time, not something indexed by scan position.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/beams/VNIKITIN/eagle/vnikitin/20260829"

SCANS = [
    ("AtomiumS1 FT", f"{BASE}/AtomiumS1/Atomium_S1_FT_4K_RD300_004p5nm_0001_/correct_correct3D.txt", 4.5),
    ("AtomiumS1 HT", f"{BASE}/AtomiumS1/Atomium_S1_HT_4K_RD300_004p5nm_0004_/correct_correct3D.txt", 4.5),
    ("ctxl FT",      f"{BASE}/ctxl/ctxl_FT_4K_RD300_007p5nm_0003_/correct_correct3D.txt",           7.5),
    ("ctxl HT",      f"{BASE}/ctxl/ctxl_HT_4K_RD300_007p5nm_0001_/correct_correct3D.txt",           7.5),
]

p = argparse.ArgumentParser()
p.add_argument("-o", "--out", default="/tmp/correct3d.png")
a = p.parse_args()

fig, axes = plt.subplots(2, 4, figsize=(19, 7.5), sharex="col")

for j, (name, path, voxel) in enumerate(SCANS):
    d = np.loadtxt(path)
    # drop the trailing duplicate row: nabu writes ntheta+1
    d = d[:-1]
    nf = len(d)
    ang = np.arange(nf) / nf * 180.0

    for i, (col, lab, c) in enumerate(((0, "vertical", "tab:blue"),
                                       (1, "horizontal", "tab:red"))):
        ax = axes[i, j]
        ax.plot(ang, d[:, col], c, lw=0.8)
        ax.axhline(0, color="0.7", lw=0.5)
        ax.grid(alpha=0.3)
        pp = d[:, col].max() - d[:, col].min()
        ax.set_title(f"{name}  {lab}\n{nf} frames, p-p {pp:.2f} binned px"
                     f" = {pp*2:.2f} px = {pp*2*voxel:.0f} nm", fontsize=9)
        if i == 1:
            ax.set_xlabel("angle (deg)")
        if j == 0:
            ax.set_ylabel(f"{lab} shift (2x2-binned det px)")
        # right axis in unbinned pixels
        r = ax.twinx()
        r.set_ylim(np.array(ax.get_ylim()) * 2)
        if j == 3:
            r.set_ylabel("unbinned det px")

fig.suptitle("correct_correct3D.txt -- per-frame sample drift, ESRF 20260829", y=0.98)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(a.out, dpi=110)
print("wrote", a.out)

# --- numbers, so the plot does not have to be squinted at ---------------------
print(f"\n{'scan':14s} {'frames':>7s} {'v p-p':>9s} {'h p-p':>9s} "
      f"{'v p-p (nm)':>11s} {'h p-p (nm)':>11s}")
for name, path, voxel in SCANS:
    d = np.loadtxt(path)[:-1]
    v = d[:, 0].max() - d[:, 0].min()
    h = d[:, 1].max() - d[:, 1].min()
    print(f"{name:14s} {len(d):7d} {v:9.3f} {h:9.3f} "
          f"{v*2*voxel:11.0f} {h*2*voxel:11.0f}")

# ctxl FT carries a *_HT_source.txt sibling -- check whether the two ctxl curves
# are literally the same motion resampled, which the ranges suggest.
cf = np.loadtxt(SCANS[2][1])[:-1]
ch = np.loadtxt(SCANS[3][1])[:-1]
cfr = cf[::3]                      # 12000 -> 4000, one per angle
print("\nctxl FT[::3] vs ctxl HT   max|diff| v %.4f  h %.4f  (binned px)"
      % (np.abs(cfr[:, 0] - ch[:, 0]).max(), np.abs(cfr[:, 1] - ch[:, 1]).max()))

# --- overlay, in nm, because the four are not four independent measurements ---
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4.5))
for name, path, voxel in SCANS:
    d = np.loadtxt(path)[:-1]
    ang = np.arange(len(d)) / len(d) * 180.0
    for i in (0, 1):
        ax2[i].plot(ang, d[:, i] * 2 * voxel, lw=1.0, label=name)
for i, lab in enumerate(("vertical", "horizontal")):
    ax2[i].set_title(lab); ax2[i].set_xlabel("angle (deg)")
    ax2[i].set_ylabel("sample drift (nm)"); ax2[i].grid(alpha=0.3)
    ax2[i].axhline(0, color="0.7", lw=0.5); ax2[i].legend(fontsize=8)
fig2.suptitle("Same curves in nm -- ctxl FT is exactly 0.6 x AtomiumS1 FT "
              "(the 4.5->7.5 nm voxel rescale), so only two of the four are "
              "independent", fontsize=10)
fig2.tight_layout(rect=(0, 0, 1, 0.93))
out2 = os.path.splitext(a.out)[0] + "_overlay.png"
fig2.savefig(out2, dpi=110)
print("wrote", out2)
