"""
Publication figures for the AtomiumL1 step-6 (BH) reconstructions.

Covers the three bin-0 runs that finished on 2026-08-24:

    HT 4 dist   AtomiumL1_HT_014nm_rec6            ndist=4  rot_shift=-14.375964
    HT 1 dist   AtomiumL1_HT_014nm_rec6_1dist      ndist=1  rot_shift=-14.375964
    largedisp   AtomiumL1_FT_large_rand_disp_..._rec6  ndist=1  rot_shift=-11.5

Produces, at dpi=300, under FIG_DIR:

    fig01_convergence.png                     F0 vs iter (absolute + relative)
    fig02_recon_iter{IT}.png                  3 runs x (horiz, zoom, vert, zoom)
    fig03_recon_{run}_iter{IT}.png            per-run detail, 2 zooms per slice
    fig04_probe_{run}.png                     recovered probe, amp + phase
    fig05_positions_{run}.png                 pos_init - pos_final(1536)
    fig06_positions_summary.png               all runs, dy/dx vs angle
    fig07_median_{run}_iter{IT}.png            zooms, 3D median filtered

Run with the holotomocupy env (needs h5py + matplotlib):

    ~/miniforge3/envs/holotomocupy/bin/python plot_results.py

Notes
-----
* conv.csv is rewritten from scratch by each hierarchical level, so only the
  bin-0 segment (iters 1280..1536) survives for these runs.  The `iter=-1` row
  is the error of the bin-1 solution upsampled onto the bin-0 grid, i.e. before
  the first bin-0 update; it is reported in the panel text, not as a curve
  point, because it sits an order of magnitude above the rest.
* Absolute errors are not directly comparable between the 1-distance and
  4-distance runs (F0 is normalised by data_size, which scales with ndist), so
  panel (b) shows each run relative to its own iter-1280 value.
* Positions: the initial values come from /exchange/cshifts_final in the
  steps15 file, with rotation_center_shift added to the x component, exactly as
  Reader.read_pos builds them at bin=0 (scale=1).  Plotted quantity is
  init - recovered(1536), in bin-0 detector pixels.
"""

import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
ROOT    = "/data3/vnikitin/ESRF/atomium/20250607/AtomiumL1_rec"
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
DPI     = 300
ITERS   = [1344, 1408, 1536]
FINAL   = 1536
VOXEL_NM = 14.0          # bin-0 voxel size, from the step-6 run header

# categorical slots 1-3 of the validated default palette (light mode);
# all-pairs CVD dE 9.2 / normal-vision dE 24.0
COLORS = {"HT_4dist": "#2a78d6", "HT_1dist": "#eb6834", "largedisp": "#1baf7a"}
MARKERS = {"HT_4dist": "o", "HT_1dist": "s", "largedisp": "^"}
DASHES  = {"HT_4dist": (None, None), "HT_1dist": (5, 2), "largedisp": (1.5, 1.5)}

RUNS = {
    "HT_4dist": dict(
        label="HT, 4 distances",
        out=f"{ROOT}/AtomiumL1_HT_014nm_rec6",
        src=f"{ROOT}/AtomiumL1_HT_014nm_rec/AtomiumL1_HT_014nm.h5",
        ndist=4, rot_shift=-14.375964,
    ),
    "HT_1dist": dict(
        label="HT, 1 distance",
        out=f"{ROOT}/AtomiumL1_HT_014nm_rec6_1dist",
        src=f"{ROOT}/AtomiumL1_HT_014nm_rec/AtomiumL1_HT_014nm.h5",
        ndist=1, rot_shift=-14.375964,
    ),
    "largedisp": dict(
        label="large_disp, 1 distance",
        out=f"{ROOT}/AtomiumL1_FT_large_rand_disp_014nm_rec6",
        src=f"{ROOT}/AtomiumL1_FT_large_rand_disp_014nm_rec/"
            f"AtomiumL1_FT_large_rand_disp_014nm.h5",
        ndist=1, rot_shift=-11.5,
    ),
}

ZMID = 1024          # horizontal slice index (z)
YMID = 1024          # vertical slice index (y), matches the writer's _vert TIFF

# zoom boxes, given as (xc, yc) centres in the displayed image's own
# coordinates; (x0, y0, size) corners are derived below.
ZOOM_SIZE = 206
ZOOM_H_C = [(675, 1290), (1115, 1525)]     # horizontal (axial) slice, (x, y)
ZOOM_V_C = [(775, 970), (1380, 1340)]      # vertical slice,           (x, z)

VLIM = (-4.2, 0.3)                         # fixed grey window, all runs/iters
CROP = 128                                 # margin dropped from full-slice panels
MED_R = 1                                  # 3D median filter radius -> 3^3


def _boxes(centres, s=ZOOM_SIZE):
    return [(int(cx - s // 2), int(cy - s // 2), s) for cx, cy in centres]


ZOOM_H = _boxes(ZOOM_H_C)
ZOOM_V = _boxes(ZOOM_V_C)
ZOOM_TAG = ["1", "2"]

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.edgecolor": "#9a9a95",
    "axes.linewidth": 0.6,
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "text.color": "#0b0b0b",
    "axes.labelcolor": "#0b0b0b",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def ckpt(run, it):
    return f"{RUNS[run]['out']}/checkpoints/checkpoint_{it:04d}.h5"


def read_slices(run, it):
    """Return (horizontal z=ZMID, vertical y=YMID) real-part slices."""
    with h5py.File(ckpt(run, it), "r") as f:
        return f["obj_re"][ZMID], f["obj_re"][:, YMID, :]


def window(img, lo=0.2, hi=99.8):
    """Robust display window from percentiles of the central disk."""
    ny, nx = img.shape
    yy, xx = np.ogrid[:ny, :nx]
    m = (yy - ny / 2) ** 2 + (xx - nx / 2) ** 2 < (0.46 * min(ny, nx)) ** 2
    v = img[m]
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def scalebar(ax, img_w_px, length_um=5.0, color="black", px_nm=None):
    """Draw a scale bar in the lower-right corner of an image axis.

    px_nm is the size of one displayed pixel in the object plane; it defaults
    to the bin-0 voxel, which is what the reconstruction panels show.
    """
    px_nm = VOXEL_NM if px_nm is None else px_nm
    n_px = length_um * 1000.0 / px_nm
    if n_px > 0.6 * img_w_px:                       # shrink for zoom panels
        length_um = 2.0
        n_px = length_um * 1000.0 / px_nm
    if n_px > 0.6 * img_w_px:
        length_um = 1.0
        n_px = length_um * 1000.0 / px_nm
    pad = 0.045 * img_w_px
    y0 = ax.get_ylim()[0] - pad                     # ylim is inverted for images
    x1 = ax.get_xlim()[1] - pad
    stroke = [pe.withStroke(linewidth=2.6, foreground="#ffffff")]
    ax.plot([x1 - n_px, x1], [y0, y0], color=color, lw=2.4,
            solid_capstyle="butt", path_effects=stroke)
    ax.text(x1 - n_px / 2, y0 - 0.018 * img_w_px,
            f"{length_um:g} " + r"$\mu$m", color=color, ha="center", va="bottom",
            fontsize=7, path_effects=[pe.withStroke(linewidth=1.8,
                                                    foreground="#ffffff")])


def show(ax, img, vlim, title=None, cmap="gray", bar=True, barlen=5.0):
    ax.imshow(img, cmap=cmap, vmin=vlim[0], vmax=vlim[1], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#9a9a95"); s.set_linewidth(0.6)
    if title:
        ax.set_title(title, pad=3)
    if bar:
        scalebar(ax, img.shape[1], barlen)


def mark_zooms(ax, boxes, tags):
    for (x0, y0, s), t in zip(boxes, tags):
        ax.add_patch(Rectangle((x0, y0), s, s, fill=False,
                               ec="#eda100", lw=0.9))
        ax.text(x0 - 12, y0 - 12, t, color="#eda100", fontsize=8.5,
                fontweight="bold", ha="right", va="bottom",
                path_effects=[pe.withStroke(linewidth=1.8,
                                            foreground="#111111")])


def trim(img, c=CROP):
    """Drop a c-pixel margin from a full slice, for display only."""
    return img[c:img.shape[0] - c, c:img.shape[1] - c]


def shift_boxes(boxes, c=CROP):
    """Zoom boxes expressed in the trimmed image's coordinates."""
    return [(x0 - c, y0 - c, s) for x0, y0, s in boxes]


def crop(img, box):
    x0, y0, s = box
    return img[y0:y0 + s, x0:x0 + s]


def crop3d_median(run, it, orient, box, r=MED_R):
    """Zoom box cut out of a 3D-median-filtered volume.

    Only the box plus an r-voxel halo is read and filtered, so the cost scales
    with the window, not with the 2048^3 volume.  ``orient`` is "h" for the
    axial slice at z=ZMID (box is (x, y)) or "v" for the vertical slice at
    y=YMID (box is (x, z)).
    """
    from scipy.ndimage import median_filter
    x0, a0, s = box                              # a0 = y for "h", z for "v"
    with h5py.File(ckpt(run, it), "r") as f:
        d = f["obj_re"]
        nz, ny, nx = d.shape
        xs, xe = max(0, x0 - r), min(nx, x0 + s + r)
        if orient == "h":
            ys, ye = max(0, a0 - r), min(ny, a0 + s + r)
            sub = d[ZMID - r:ZMID + r + 1, ys:ye, xs:xe]
            a_s = ys
        else:
            zs, ze = max(0, a0 - r), min(nz, a0 + s + r)
            sub = d[zs:ze, YMID - r:YMID + r + 1, xs:xe]
            a_s = zs
    sub = median_filter(sub.astype("float32"), size=2 * r + 1, mode="nearest")
    img = sub[r] if orient == "h" else sub[:, r, :]
    return img[a0 - a_s:a0 - a_s + s, x0 - xs:x0 - xs + s]


# --------------------------------------------------------------------------
# fig 00 - the measured data and the acquisition geometry, side by side
# --------------------------------------------------------------------------
DATA_VLIM   = (0.75, 1.25)      # hologram display window, I / I_ref
DATA_THETAS = [0, 600, 1200, 1799]      # 0, 60, 120, 179.9 deg
DATA_BIN    = 1                 # holograms are shown at 1024^2 (2x binned)


# Raw EDF scan directories -- the only place the frame timing lives.
RAW = "/data3/vnikitin/ESRF/atomium/20250607/AtomiumL1"


def scan_info(run):
    """Frame timing, read out of the raw ESRF .info files.

    The reconstruction h5 keeps geometry but not exposure, so this goes back
    to <RAW>/<pfile>_<k+1>_/<pfile>_<k+1>_.info -- the same directories
    config_steps15.conf reads.  Returns {} when the raw scans are not
    mounted; the exposure rows are then simply left out of the table.
    """
    pfile = os.path.basename(RUNS[run]["src"])[:-3]
    scans = []
    for k in range(RUNS[run]["ndist"]):
        f = f"{RAW}/{pfile}_{k+1}_/{pfile}_{k+1}_.info"
        if not os.path.exists(f):
            return {}
        d = {}
        for line in open(f):
            key, sep, val = line.partition("=")
            if sep:
                d[key.strip()] = val.strip()
        scans.append(d)
    num = lambda key, cast=float: [cast(d[key]) for d in scans]
    return dict(count=num("Count_time"), latency=num("Latency_time"),
                tomo_n=num("TOMO_N", int), ref_n=num("REF_N", int),
                dark_n=num("DARK_N", int))


def acq(run):
    """Acquisition parameters, read out of the run's own data file."""
    with h5py.File(RUNS[run]["src"], "r") as f:
        e = f["exchange"]
        a = dict(
            energy   = float(e["energy"][0]),
            det_px   = float(e["detector_pixelsize"][0]),
            fdd      = float(e["focusdetectordistance"][0]),
            z1       = e["z1"][:].astype("float64"),
            voxel    = e["voxelsize"][:].astype("float64"),
            theta    = e["theta"][:, 0].astype("float64"),
            shifts   = e["cshifts_final"][:].astype("float64"),
            n        = e["pdata0"].shape[-1],
        )
    a["mag"]  = a["fdd"] / a["z1"]
    a["lam"]  = 12.39841984 / a["energy"]            # angstrom
    a["ndist"] = len(a["z1"])
    return a


def hologram(run, dist, theta_idx):
    """One flat-corrected hologram, divided by the reference so the beam
    structure does not swamp the fringes."""
    with h5py.File(RUNS[run]["src"], "r") as f:
        e = f["exchange"]
        d = e[f"pdata{dist}_{DATA_BIN}"][theta_idx].astype("float32")
        r = e[f"pref_{DATA_BIN}"][dist].astype("float32")
    return d / r


def fig_data():
    """Example holograms + acquisition geometry for the two acquisitions."""
    A = {r: acq(r) for r in ("HT_4dist", "largedisp")}
    ht, ld = A["HT_4dist"], A["largedisp"]
    px_nm = lambda a, j: a["voxel"][j] * 1e9 * 2 ** DATA_BIN   # displayed pixel

    fig = plt.figure(figsize=(9.2, 10.6))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1.18],
                          hspace=0.20, wspace=0.05,
                          left=0.076, right=0.985, top=0.935, bottom=0.075)

    rows = []
    # (a) HT, all four distances, same angle
    for j in range(ht["ndist"]):
        ax = fig.add_subplot(gs[0, j])
        show(ax, hologram("HT_4dist", j, DATA_THETAS[0]), DATA_VLIM, bar=False)
        scalebar(ax, ht["n"] // 2 ** DATA_BIN, 5.0, px_nm=px_nm(ht, j))
        ax.set_title(f"dist {j}   $z_1$ = {ht['z1'][j]*1e3:.3f} mm\n"
                     f"M = {ht['mag'][j]:.1f}$\\times$, "
                     f"{ht['voxel'][j]*1e9:.2f} nm",
                     fontsize=6.6, pad=2.5, color="#52514e")
        if j == 0:
            rows.append((ax, "(a)  AtomiumL1_HT  --  four propagation "
                             "distances, one angle ($\\theta$ = 0$^\\circ$)"))

    # (b) HT, one distance, four angles;  (c) large_disp, same four angles
    for r, (run, dist, tag) in enumerate((
            ("HT_4dist",  0, "(b)  AtomiumL1_HT  --  dist 0, four angles "
                             "(sample stays put: drift of a few px)"),
            ("largedisp", 0, "(c)  AtomiumL1_large_disp  --  its single "
                             "distance, same four angles (sample is moved "
                             "by up to $\\pm$300 px)")), start=1):
        a = A[run if run in A else "largedisp"]
        for i, t in enumerate(DATA_THETAS):
            ax = fig.add_subplot(gs[r, i])
            show(ax, hologram(run, dist, t), DATA_VLIM, bar=False)
            scalebar(ax, a["n"] // 2 ** DATA_BIN, 5.0, px_nm=px_nm(a, dist))
            ax.set_title(r"$\theta$ = " + f"{a['theta'][t]:.1f}" + r"$^\circ$",
                         fontsize=6.6, pad=2.5, color="#52514e")
            if i == 0:
                rows.append((ax, tag))

    for ax, tag in rows:
        pos = ax.get_position()
        fig.text(pos.x0, pos.y1 + 0.021, tag, fontsize=8.2, va="bottom",
                 ha="left", color="#0b0b0b", fontweight="bold")

    # (d) where the sample sat on the detector, both acquisitions
    axs_ = fig.add_subplot(gs[3, 0])
    n = ht["n"]
    axs_.add_patch(Rectangle((-n / 2, -n / 2), n, n, fill=False,
                             ec="#c3c2b7", lw=0.8))
    for j in range(ld["ndist"]):
        axs_.plot(ld["shifts"][:, j, 1], ld["shifts"][:, j, 0], ls="none",
                  marker="o", ms=1.6, mew=0, alpha=0.45,
                  color=COLORS["largedisp"], label="large_disp" if j == 0 else None)
    for j in range(ht["ndist"]):
        axs_.plot(ht["shifts"][:, j, 1], ht["shifts"][:, j, 0], ls="none",
                  marker="o", ms=1.6, mew=0, alpha=0.55,
                  color=COLORS["HT_4dist"], label="HT (4 dist)" if j == 0 else None)
    axs_.set_aspect("equal")
    axs_.set_xlim(-n / 2 - 60, n / 2 + 60); axs_.set_ylim(-n / 2 - 60, n / 2 + 60)
    axs_.set_xlabel(r"$\Delta x$  [detector px]", fontsize=7)
    axs_.set_ylabel(r"$\Delta y$  [detector px]", fontsize=7)
    axs_.tick_params(labelsize=6.5)
    axs_.set_title("(d)  sample position", loc="left",
                   fontsize=8.2, fontweight="bold", pad=6)
    axs_.text(0.03, 0.965, "all 1800 angles, both scans", fontsize=6.3,
              color="#898781", va="top", ha="left", transform=axs_.transAxes)
    axs_.legend(frameon=False, fontsize=6.5, loc="lower left",
                handletextpad=0.2, borderpad=0.1, markerscale=3.2)
    for s_ in ("top", "right"):
        axs_.spines[s_].set_visible(False)

    # inset: the HT cluster, which is a few px across at this scale
    ins = axs_.inset_axes([0.60, 0.62, 0.38, 0.38])
    for j in range(ht["ndist"]):
        ins.plot(ht["shifts"][:, j, 1], ht["shifts"][:, j, 0], ls="none",
                 marker="o", ms=1.0, mew=0, alpha=0.5, color=COLORS["HT_4dist"])
    ins.set_aspect("equal")
    ins.set_xlim(-60, 110); ins.set_ylim(-40, 40)
    ins.tick_params(labelsize=5, length=2, pad=1)
    ins.set_title("HT, zoomed", fontsize=5.8, color="#52514e", pad=2)
    for sp in ins.spines.values():
        sp.set_color("#c3c2b7"); sp.set_linewidth(0.6)

    # (e) the numbers
    axt = fig.add_subplot(gs[3, 1:]); axt.axis("off")
    fmt = lambda v, s, f="{:.3f}": " / ".join(f.format(x * s) for x in v)
    std_ht = ht["shifts"].std(0)            # (ndist, 2), y and x
    rng_ht = ht["shifts"].max(0) - ht["shifts"].min(0)
    std_ld = ld["shifts"].std(0)[0]
    rng_ld = (ld["shifts"].max(0) - ld["shifts"].min(0))[0]
    table = [
        ("X-ray energy",           f"{ht['energy']:.2f} keV  "
                                   f"($\\lambda$ = {ht['lam']:.4f} $\\AA$)", "same"),
        ("detector",               f"{ht['n']} $\\times$ {ht['n']} px, "
                                   f"{ht['det_px']*1e6:.3f} $\\mu$m pixel", "same"),
        ("focus-detector distance", f"{ht['fdd']*1e2:.1f} cm", "same"),
        ("propagation distances",  f"{ht['ndist']}:  "
                                   + fmt(ht["z1"], 1e3) + " mm",
                                   f"{ld['ndist']}:  {ld['z1'][0]*1e3:.3f} mm"),
        ("magnification",          fmt(ht["mag"], 1, "{:.1f}") + r"$\times$",
                                   f"{ld['mag'][0]:.1f}" + r"$\times$"),
        ("voxel size",             fmt(ht["voxel"], 1e9, "{:.2f}") + " nm",
                                   f"{ld['voxel'][0]*1e9:.2f} nm"),
        ("field of view",          f"{ht['n']*ht['voxel'][0]*1e6:.2f} "
                                   r"$\mu$m  (at the finest voxel)",
                                   f"{ld['n']*ld['voxel'][0]*1e6:.2f} "
                                   r"$\mu$m"),
        ("angles",                 f"{len(ht['theta'])} over "
                                   f"{ht['theta'][-1]:.1f}$^\\circ$ "
                                   f"({ht['theta'][1]-ht['theta'][0]:.3f}$^\\circ$ step)",
                                   "same"),
        ("frames recorded",        f"{ht['ndist']*len(ht['theta'])}",
                                   f"{ld['ndist']*len(ld['theta'])}"),
        ("sample shift, std",      " / ".join(f"{v:.1f}" for v in
                                              std_ht.mean(1)) + " px",
                                   f"{std_ld.mean():.0f} px  "
                                   f"({std_ld.mean()*ld['voxel'][0]*1e6:.2f} "
                                   r"$\mu$m)"),
        ("sample shift, range",    " / ".join(f"{v:.0f}" for v in
                                              rng_ht.mean(1)) + " px",
                                   f"{rng_ld.mean():.0f} px  "
                                   f"({100*rng_ld.mean()/ld['n']:.0f}% of the "
                                   "field)"),
    ]
    si_ht, si_ld = scan_info("HT_4dist"), scan_info("largedisp")
    if si_ht and si_ld:
        # frames on the sample only -- flats and darks are not sample dose
        dose = lambda si: sum(c * t for c, t in zip(si["count"], si["tomo_n"]))
        d_ht, d_ld = dose(si_ht), dose(si_ld)
        same = (set(si_ht["count"]) | set(si_ld["count"]) == {si_ht["count"][0]}
                and si_ht["latency"][0] == si_ld["latency"][0])
        i = [t[0] for t in table].index("frames recorded") + 1
        table[i:i] = [
            ("exposure per frame",
             f"{si_ht['count'][0]:.2f} s  "
             f"(+ {si_ht['latency'][0]:.2f} s readout latency)",
             "same" if same else f"{si_ld['count'][0]:.2f} s"),
            ("exposure on the sample",
             f"{d_ht:.0f} s  =  {d_ht/60:.0f} min",
             f"{d_ld:.0f} s  =  {d_ld/60:.0f} min  "
             f"({d_ht/d_ld:.0f}$\\times$ less)"),
        ]
    x_p, x_1, x_2 = 0.005, 0.40, 0.72
    y = 0.985
    dy = min(0.082, 0.90 / (len(table) + 0.5))
    axt.text(x_p, y, "acquisition parameter", fontsize=7.4, fontweight="bold",
             va="top", transform=axt.transAxes)
    axt.text(x_1, y, "AtomiumL1_HT", fontsize=7.4, fontweight="bold",
             va="top", color=COLORS["HT_4dist"], transform=axt.transAxes)
    axt.text(x_2, y, "AtomiumL1_large_disp", fontsize=7.4, fontweight="bold",
             va="top", color=COLORS["largedisp"], transform=axt.transAxes)
    y -= 0.046
    axt.plot([x_p, 1.0], [y, y], color="#898781", lw=0.8, clip_on=False,
             transform=axt.transAxes)
    for k, (name, v1, v2) in enumerate(table):
        yy = y - 0.018 - (k + 0.5) * dy
        if k:
            axt.plot([x_p, 1.0], [yy + dy / 2, yy + dy / 2], color="#e1e0d9",
                     lw=0.5, clip_on=False, transform=axt.transAxes)
        axt.text(x_p, yy, name, fontsize=6.8, va="center", color="#52514e",
                 transform=axt.transAxes)
        axt.text(x_1, yy, v1, fontsize=6.8, va="center", color="#0b0b0b",
                 transform=axt.transAxes)
        axt.text(x_2, yy, v2, fontsize=6.8, va="center", color="#0b0b0b",
                 transform=axt.transAxes)

    fig.text(0.076, 0.006,
             "Holograms are flat-corrected and divided by the reference, "
             f"shown at {2**DATA_BIN}$\\times$ binning, grey window "
             f"[{DATA_VLIM[0]:.2f}, {DATA_VLIM[1]:.2f}] of $I/I_{{ref}}$.  "
             "Scale bars are in the object plane, so they differ between "
             "distances.\nBoth scans share the source, the detector and the "
             "angular sampling; they differ in the number of distances and in "
             "how far the sample was displaced between angles.",
             fontsize=6.3, color="#52514e", va="bottom", linespacing=1.5)
    save(fig, "fig00_data_and_geometry.png", tight=False)


# start_iter of each hierarchical level (config_step6_bin{2,1,0}.conf)
LEVEL_START = {2: 0, 1: 1024, 0: 1280}
LEVEL_LABEL = {2: "bin 2\n512$^3$", 1: "bin 1\n1024$^3$", 0: "bin 0\n2048$^3$"}


def load_conv_full(run):
    """Per-level error history, when it exists.

    conv.csv holds only the last level that ran (every level rewrites it), so
    the full ladder comes either from recompute_conv.py
    (conv_recomputed_bin{b}.csv) or, for runs made after the rec_mpi.py fix,
    from conv_bin{b}.csv.  Returns {bin: (iters, errs, is_level_initial)};
    empty when neither file exists.
    """
    import csv
    out = {}
    for b in (2, 1, 0):
        for name in (f"conv_recomputed_bin{b}.csv", f"conv_bin{b}.csv"):
            path = f"{RUNS[run]['out']}/{name}"
            if not os.path.exists(path):
                continue
            it, err, ini = [], [], []
            with open(path) as fh:
                for row in csv.DictReader(fh):
                    i, e = int(row["iter"]), float(row["err"])
                    flag = int(row.get("initial", 0) or 0)
                    if i < 0:            # conv_bin{b}.csv tags it as iter=-1
                        i, flag = LEVEL_START[b], 1
                    it.append(i); err.append(e); ini.append(flag)
            o = np.argsort(it)
            out[b] = (np.array(it)[o], np.array(err)[o], np.array(ini)[o])
            break
    return out


def load_conv(run):
    import csv
    it, err = [], []
    init_err = None
    with open(f"{RUNS[run]['out']}/conv.csv") as fh:
        for row in csv.DictReader(fh):
            i, e = int(row["iter"]), float(row["err"])
            if i < 0:
                init_err = e
            else:
                it.append(i); err.append(e)
    return np.array(it), np.array(err), init_err


def load_pos_delta(run, it=FINAL):
    """init - recovered, shape [ntheta, ndist, 2] with [..., 0]=y, [..., 1]=x."""
    cfg = RUNS[run]
    nd = cfg["ndist"]
    with h5py.File(ckpt(run, it), "r") as f:
        rec = f["pos"][:, :nd].astype("float32")
    ntheta = rec.shape[0]
    with h5py.File(cfg["src"], "r") as f:
        ini = f["/exchange/cshifts_final"][:ntheta, :nd].astype("float32")
    ini = ini.copy()
    ini[..., 1] += np.float32(cfg["rot_shift"])     # bin=0 -> scale=1
    return ini, rec, ini - rec


def save(fig, name, tight=True):
    os.makedirs(FIG_DIR, exist_ok=True)
    p = f"{FIG_DIR}/{name}"
    fig.savefig(p, dpi=DPI, **({"bbox_inches": "tight"} if tight else {}))
    plt.close(fig)
    print("wrote", p, f"({os.path.getsize(p)/1e6:.2f} MB)")


# --------------------------------------------------------------------------
# fig 01 - convergence
# --------------------------------------------------------------------------
def fig_convergence():
    """F0 vs iteration.  Only the bin-0 level has a surviving history."""
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.5))
    short = {"HT_4dist": "4 dist", "HT_1dist": "1 dist",
             "largedisp": "large_disp"}
    notes = []
    for run, cfg in RUNS.items():
        it, err, e0 = load_conv(run)
        c = COLORS[run]
        for ax, y in ((axs[0], err), (axs[1], err / err[0])):
            ax.semilogy(it, y, color=c, lw=1.6, dashes=DASHES[run],
                        marker=MARKERS[run], ms=3.6, mew=0, zorder=3,
                        label=cfg["label"])
        axs[0].annotate(short[run], xy=(it[-1], err[-1]),
                        xytext=(5, 0), textcoords="offset points",
                        color=c, fontsize=7, fontweight="bold",
                        va="center", ha="left", zorder=4)
        notes.append(f"{short[run]} {e0:.2e}")

    axs[0].set_xlabel("iteration"); axs[0].set_ylabel(r"$F_0$  (data-fit error)")
    axs[0].set_title("(a) absolute error", loc="left")
    axs[1].set_xlabel("iteration")
    axs[1].set_ylabel(r"$F_0\,/\,F_0(1280)$")
    axs[1].set_title("(b) relative to first bin-0 iterate", loc="left")

    for ax in axs:
        ax.grid(True, which="both", color="#e3e3df", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axs[0].set_xlim(1268, 1618)
    axs[1].set_xlim(1268, 1548)

    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.175), handlelength=2.4,
               columnspacing=1.8)
    fig.subplots_adjust(wspace=0.33, bottom=0.36, top=0.90,
                        left=0.10, right=0.98)
    fig.text(0.5, 0.115,
             "bin-0 level only -- conv.csv is rewritten from scratch by each "
             "hierarchical level, so the bin-2 and bin-1 error histories are "
             "gone.\n"
             "$F_0$ before the first bin-0 update (bin-1 solution upsampled to "
             "2048$^3$): " + ", ".join(notes) + ".\n"
             "Absolute values are not comparable across runs -- $F_0$ is "
             "normalised by data_size, which scales with ndist; use panel (b).",
             ha="center", va="top", fontsize=6.3, color="#52514e",
             linespacing=1.5)
    save(fig, "fig01_convergence.png", tight=False)


def fig_convergence_full():
    """F0 over the whole hierarchical ladder, every checkpointed iteration.

    Needs conv_recomputed_bin{b}.csv (recompute_conv.py) or conv_bin{b}.csv;
    silently skipped when a run has neither.
    """
    data = {run: load_conv_full(run) for run in RUNS}
    have = {run: d for run, d in data.items() if d}
    if not have:
        print("fig01b skipped: no per-level error history on disk "
              "(run recompute_conv_all.sh first)")
        return

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.0))
    short = {"HT_4dist": "4 dist", "HT_1dist": "1 dist",
             "largedisp": "large_disp"}
    missing = [short[r] for r in RUNS if r not in have]

    # level bands
    edges = [(0, 1024, 2), (1024, 1280, 1), (1280, 1536, 0)]
    for ax in axs:
        for k, (a, b_, lvl) in enumerate(edges):
            if k % 2:
                ax.axvspan(a, b_, color="#f2f1ec", lw=0, zorder=0)
            ax.axvline(a, color="#c3c2b7", lw=0.7, ls=(0, (3, 3)), zorder=1)

    for run, lv in have.items():
        c = COLORS[run]
        first = None
        for b in (2, 1, 0):
            if b not in lv:
                continue
            it, err, ini = lv[b]
            if first is None:
                first = err[0]
            for ax, y in ((axs[0], err), (axs[1], err / first)):
                ax.semilogy(it, y, color=c, lw=1.5, dashes=DASHES[run],
                            marker=MARKERS[run], ms=3.0, mew=0, zorder=3,
                            label=RUNS[run]["label"] if b == 2 else None)
                m = ini.astype(bool)
                if m.any():          # level hand-over: open marker
                    ax.semilogy(it[m], y[m], ls="none", marker="o", ms=5.5,
                                mfc="none", mec=c, mew=1.1, zorder=4)
        last_b = min(lv)
        it, err, _ = lv[last_b]
        axs[0].annotate(short[run], xy=(it[-1], err[-1]),
                        xytext=(5, 0), textcoords="offset points",
                        color=c, fontsize=7, fontweight="bold",
                        va="center", ha="left", zorder=4)

    for ax, lbl in zip(axs, (r"$F_0$  (data-fit error)",
                             r"$F_0\,/\,F_0(\mathrm{start})$")):
        ax.set_xlabel("iteration"); ax.set_ylabel(lbl)
        ax.grid(True, which="both", color="#e3e3df", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.set_xlim(-40, 1660)
    # titles clear the in-band level labels
    axs[0].set_title("(a) absolute error", loc="left", pad=20)
    axs[1].set_title("(b) relative to the start of the ladder", loc="left", pad=20)

    for ax in axs:
        for a, b_, lvl in edges:
            ax.text(0.5 * (a + b_), 1.012, LEVEL_LABEL[lvl], ha="center",
                    va="bottom", fontsize=6.0, color="#898781",
                    linespacing=1.3, transform=ax.get_xaxis_transform())

    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.215), handlelength=2.4,
               columnspacing=1.8)
    fig.subplots_adjust(wspace=0.33, bottom=0.40, top=0.83,
                        left=0.10, right=0.98)
    note = ("Every checkpointed iteration of all three levels.  Open circles "
            "are the level hand-overs -- the previous level's solution "
            "upsampled to the finer grid,\nevaluated before the first update "
            "there; the jump across them is a change of grid and of "
            "data_size, not a step of the solver.\n"
            "Absolute values are not comparable across runs or across levels "
            "-- $F_0$ is normalised by data_size, which scales with ndist and "
            "with $n^2$; use panel (b).")
    if missing:
        note += ("\nNo per-level history for: " + ", ".join(missing)
                 + " -- run recompute_conv_all.sh there.")
    fig.text(0.5, 0.155, note, ha="center", va="top", fontsize=6.3,
             color="#52514e", linespacing=1.5)
    save(fig, "fig01b_convergence_all_iters.png", tight=False)


# --------------------------------------------------------------------------
# fig 02 - three runs side by side, one figure per iteration
# --------------------------------------------------------------------------
def fig_recon_compare(vlims):
    for it in ITERS:
        fig, axs = plt.subplots(3, 4, figsize=(9.0, 7.0))
        for r, (run, cfg) in enumerate(RUNS.items()):
            hz, vt = read_slices(run, it)
            vl = vlims[run]
            show(axs[r, 0], trim(hz), vl, bar=True)
            mark_zooms(axs[r, 0], shift_boxes(ZOOM_H[:1]), ["1"])
            show(axs[r, 1], crop(hz, ZOOM_H[0]), vl, bar=True, barlen=2.0)
            show(axs[r, 2], trim(vt), vl, bar=True)
            mark_zooms(axs[r, 2], shift_boxes(ZOOM_V[1:]), ["2"])
            show(axs[r, 3], crop(vt, ZOOM_V[1]), vl, bar=True, barlen=2.0)
            axs[r, 0].set_ylabel(cfg["label"].replace(", ", ",\n"), fontsize=7.5)
            if r == 0:
                axs[r, 0].set_title(f"axial slice  $z$ = {ZMID}", pad=4)
                axs[r, 1].set_title("zoom 1", pad=4)
                axs[r, 2].set_title(f"vertical slice  $y$ = {YMID}", pad=4)
                axs[r, 3].set_title("zoom 2", pad=4)
        fig.suptitle(f"AtomiumL1 -- BH reconstruction, "
                     r"Re$\,\{$obj$\}$, iteration " + str(it) +
                     f"   (bin 0, {VOXEL_NM:g} nm voxel)", y=0.985, fontsize=9)
        fig.subplots_adjust(wspace=0.03, hspace=0.06)
        save(fig, f"fig02_recon_iter{it}.png")


# --------------------------------------------------------------------------
# fig 03 - per-run detail, one figure per (run, iteration)
# --------------------------------------------------------------------------
def fig_recon_detail(vlims):
    for run, cfg in RUNS.items():
        vl = vlims[run]
        for it in ITERS:
            hz, vt = read_slices(run, it)
            fig, axs = plt.subplots(2, 3, figsize=(7.6, 5.3))
            show(axs[0, 0], trim(hz), vl, title=f"axial slice  $z$ = {ZMID}")
            mark_zooms(axs[0, 0], shift_boxes(ZOOM_H), ZOOM_TAG)
            for j, b in enumerate(ZOOM_H):
                show(axs[0, j + 1], crop(hz, b), vl,
                     title=f"zoom {ZOOM_TAG[j]}", barlen=2.0)
            show(axs[1, 0], trim(vt), vl, title=f"vertical slice  $y$ = {YMID}")
            mark_zooms(axs[1, 0], shift_boxes(ZOOM_V), ZOOM_TAG)
            for j, b in enumerate(ZOOM_V):
                show(axs[1, j + 1], crop(vt, b), vl,
                     title=f"zoom {ZOOM_TAG[j]}", barlen=2.0)
            fig.suptitle(f"AtomiumL1 -- {cfg['label']} -- iteration {it}"
                         r"   (Re$\,\{$obj$\}$, grey window "
                         f"[{vl[0]:.2f}, {vl[1]:.2f}])",
                         y=0.98, fontsize=9)
            fig.subplots_adjust(wspace=0.04, hspace=0.10)
            save(fig, f"fig03_recon_{run}_iter{it}.png")


# --------------------------------------------------------------------------
# fig 07 - the same zoom regions, after a 3D median filter
# --------------------------------------------------------------------------
def fig_recon_detail_median(vlims):
    k = 2 * MED_R + 1
    for run, cfg in RUNS.items():
        vl = vlims[run]
        for it in ITERS:
            fig, axs = plt.subplots(2, 2, figsize=(5.6, 6.0))
            for j, b in enumerate(ZOOM_H):
                show(axs[0, j], crop3d_median(run, it, "h", b), vl,
                     title=f"axial $z$={ZMID} -- zoom {ZOOM_TAG[j]}", barlen=2.0)
            for j, b in enumerate(ZOOM_V):
                show(axs[1, j], crop3d_median(run, it, "v", b), vl,
                     title=f"vertical $y$={YMID} -- zoom {ZOOM_TAG[j]}",
                     barlen=2.0)
            fig.suptitle(f"AtomiumL1 -- {cfg['label']} -- iteration {it}\n"
                         f"3D median filter, radius {MED_R} ({k}x{k}x{k} voxels)"
                         f"   (Re$\\,\\{{$obj$\\}}$, grey window "
                         f"[{vl[0]:.2f}, {vl[1]:.2f}])",
                         y=0.985, fontsize=9, linespacing=1.4)
            fig.subplots_adjust(wspace=0.04, hspace=0.12)
            save(fig, f"fig07_median_{run}_iter{it}.png")


# --------------------------------------------------------------------------
# fig 04 - recovered probes
# --------------------------------------------------------------------------
def fig_probes():
    for run, cfg in RUNS.items():
        nd = cfg["ndist"]
        with h5py.File(ckpt(run, FINAL), "r") as f:
            amp = f["prb_abs"][:nd]
            pha = f["prb_phase"][:nd]
        # multi-distance: one column per distance, amplitude over phase.
        # single distance: side by side, so the page is not mostly margin.
        if nd > 1:
            fig, axs = plt.subplots(2, nd, figsize=(2.35 * nd + 0.9, 5.0),
                                    squeeze=False)
            pair = [(axs[0, k], axs[1, k]) for k in range(nd)]
            axs[0, 0].set_ylabel("amplitude  $|q|$", fontsize=8)
            axs[1, 0].set_ylabel(r"phase  $\arg q$  [rad]", fontsize=8)
            for k in range(nd):
                axs[0, k].set_title(f"distance {k}", pad=4)
        else:
            fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.4), squeeze=False)
            pair = [(axs[0, 0], axs[0, 1])]
            axs[0, 0].set_title("amplitude  $|q|$", fontsize=9, pad=4)
            axs[0, 1].set_title(r"phase  $\arg q$  [rad]", fontsize=9, pad=4)

        for k, (ax_a, ax_p) in enumerate(pair):
            a, p = amp[k], pha[k]
            im0 = ax_a.imshow(a, cmap="gray",
                              vmin=float(np.percentile(a, 0.5)),
                              vmax=float(np.percentile(a, 99.5)))
            im1 = ax_p.imshow(p, cmap="twilight_shifted",
                              vmin=-np.pi, vmax=np.pi)
            for ax, im in ((ax_a, im0), (ax_p, im1)):
                ax.set_xticks([]); ax.set_yticks([])
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=6, length=2)
                cb.outline.set_linewidth(0.4)

        fig.suptitle(f"AtomiumL1 -- {cfg['label']} -- recovered probe, "
                     f"iteration {FINAL}", y=0.99, fontsize=9)
        fig.subplots_adjust(wspace=0.12, hspace=0.06)
        save(fig, f"fig04_probe_{run}.png")


# --------------------------------------------------------------------------
# fig 05 / 06 - positions
# --------------------------------------------------------------------------
DIST_C = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]


def fig_positions():
    for run, cfg in RUNS.items():
        ini, rec, d = load_pos_delta(run)
        nd = cfg["ndist"]
        th = np.arange(d.shape[0])
        fig, axs = plt.subplots(1, 3, figsize=(9.8, 3.3))

        for k in range(nd):
            c = DIST_C[k]
            axs[0].plot(th, d[:, k, 0], color=c, lw=0.55, alpha=0.85,
                        label=f"distance {k}")
            axs[1].plot(th, d[:, k, 1], color=c, lw=0.55, alpha=0.85)
            axs[2].scatter(d[:, k, 1], d[:, k, 0], s=1.4, color=c,
                           alpha=0.5, lw=0)

        axs[0].set_ylabel(r"$\Delta y$  [px]")
        axs[1].set_ylabel(r"$\Delta x$  [px]")
        for ax in (axs[0], axs[1]):
            ax.set_xlabel("projection index")
            ax.axhline(0, color="#9a9a95", lw=0.6, zorder=0)
        axs[0].set_title(r"(a) $\Delta y$ = init $-$ recovered", loc="left")
        axs[1].set_title(r"(b) $\Delta x$ = init $-$ recovered", loc="left")

        axs[2].set_xlabel(r"$\Delta x$  [px]")
        axs[2].set_ylabel(r"$\Delta y$  [px]")
        axs[2].set_title("(c) joint distribution", loc="left")
        axs[2].axhline(0, color="#9a9a95", lw=0.6, zorder=0)
        axs[2].axvline(0, color="#9a9a95", lw=0.6, zorder=0)
        axs[2].set_aspect("equal", adjustable="datalim")

        for ax in axs:
            ax.grid(True, color="#e3e3df", lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)

        stats = "   ".join(
            f"d{k}: RMS $\\Delta y$={np.sqrt((d[:, k, 0]**2).mean()):.3f}, "
            f"$\\Delta x$={np.sqrt((d[:, k, 1]**2).mean()):.3f} px"
            for k in range(nd))
        fig.suptitle(f"AtomiumL1 -- {cfg['label']} -- position correction "
                     f"(initial $-$ recovered at iteration {FINAL})",
                     y=1.0, fontsize=9)
        fig.tight_layout()
        if nd > 1:
            h, l = axs[0].get_legend_handles_labels()
            leg = fig.legend(h, l, frameon=False, ncol=nd, loc="lower center",
                             bbox_to_anchor=(0.5, -0.085), columnspacing=1.6,
                             handlelength=1.6, fontsize=8)
            for ln in leg.get_lines():
                ln.set_linewidth(1.8)
            y_note = -0.175
        else:
            y_note = -0.055
        fig.text(0.5, y_note,
                 stats + f"\ninitial scan positions span "
                 f"y $\\in$ [{ini[..., 0].min():.1f}, {ini[..., 0].max():.1f}], "
                 f"x $\\in$ [{ini[..., 1].min():.1f}, {ini[..., 1].max():.1f}] px "
                 f"(bin-0 detector pixels; rotation_center_shift "
                 f"{cfg['rot_shift']:+.4f} px already added to x)",
                 ha="center", va="top", fontsize=6.4, color="#52514e")
        save(fig, f"fig05_positions_{run}.png")


def fig_positions_summary():
    fig, axs = plt.subplots(3, 2, figsize=(7.4, 6.2), sharex=True)
    for r, (run, cfg) in enumerate(RUNS.items()):
        ini, rec, d = load_pos_delta(run)
        nd = cfg["ndist"]
        th = np.arange(d.shape[0])
        for k in range(nd):
            c = DIST_C[k]
            axs[r, 0].plot(th, d[:, k, 0], color=c, lw=0.5, alpha=0.85,
                           label=f"distance {k}")
            axs[r, 1].plot(th, d[:, k, 1], color=c, lw=0.5, alpha=0.85)
        axs[r, 0].set_ylabel(r"$\Delta y$  [px]")
        axs[r, 1].set_ylabel(r"$\Delta x$  [px]")
        axs[r, 0].text(0.015, 0.97, cfg["label"], transform=axs[r, 0].transAxes,
                       va="top", ha="left", fontsize=7.5, color="#0b0b0b")
        for ax in axs[r]:
            ax.axhline(0, color="#9a9a95", lw=0.6, zorder=0)
            ax.grid(True, color="#e3e3df", lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    axs[-1, 0].set_xlabel("projection index")
    axs[-1, 1].set_xlabel("projection index")
    fig.suptitle("AtomiumL1 -- position correction, initial $-$ recovered at "
                 f"iteration {FINAL}", y=0.995, fontsize=9)
    fig.tight_layout()
    h, l = axs[0, 0].get_legend_handles_labels()
    leg = fig.legend(h, l, frameon=False, ncol=len(l), loc="lower center",
                     bbox_to_anchor=(0.5, -0.042), columnspacing=1.8,
                     handlelength=1.6, fontsize=8)
    for ln in leg.get_lines():
        ln.set_linewidth(1.8)
    fig.text(0.5, -0.072, "distance colors apply to the 4-distance run; "
             "the single-distance runs use the first color only",
             ha="center", va="top", fontsize=6.4, color="#52514e")
    save(fig, "fig06_positions_summary.png")


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# fig 00b - the out-of-grid detector mask (Rec._build_data_mask)
# --------------------------------------------------------------------------
MASK_MARGIN = 2.0                 # mask_oob_margin, both runs
NOBJ        = 2048                # object grid actually used, both runs
# ordinal blue ramp, steps 250/350/450/600 -- the four distances are ordered
DIST_C   = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]
GRID_C   = "#0b0b0b"
DROP_C   = "#d03b3b"


def _shrink3(sh):
    """/exchange/shrink as [ntheta, ndist, 2] (y, x).

    Written per axis since the shrinkage became a fitted variable; older
    files store one value per (angle, distance), which applies to both axes.
    """
    sh = np.asarray(sh).astype("float64")
    return sh if sh.ndim == 3 else np.repeat(sh[:, :, None], 2, axis=2)


def mask_axes(n, nz, nobj, nzobj, ed, r, margin=MASK_MARGIN):
    """The 1-D keep/drop decisions of Rec._build_data_mask, verbatim.

    A cubic B-spline tap set {floor(v)-1 .. floor(v)+2} fits in [0, N-1] iff
    1 <= v < N-2, i.e. |v - (N-1)/2| <= min((N-1)/2 - 1, (N-3) - (N-1)/2).

    ed is per axis, (y, x) -- shrinkage is fitted separately for the two
    directions, so the y and x bounds can differ. A scalar still works and
    means "the same in both".
    """
    ed = np.atleast_1d(ed)
    ed_y, ed_x = float(ed[0]), float(ed[-1])
    cx, cy = (nobj - 1) * 0.5, (nzobj - 1) * 0.5
    half_x = min(cx - 1.0, (nobj - 3) - cx)
    half_y = min(cy - 1.0, (nzobj - 3) - cy)
    ax = np.abs(np.arange(n, dtype="float64") - (n - 1) * 0.5)
    ay = np.abs(np.arange(nz, dtype="float64") - (nz - 1) * 0.5)
    return (ed_x * ax <= half_x - r[1] - margin,
            ed_y * ay <= half_y - r[0] - margin)


def mask_frac(n, nobj, ed, r, margin=MASK_MARGIN):
    mx, my = mask_axes(n, n, nobj, nobj, ed, r, margin)
    return mx.mean() * my.mean()


def mask_params(run, it=1280):
    """eff_demag and |pos| worst cases, exactly as precalc sees them.

    eff_demag = (1 + shrink)/norm_magnification, maximised over all angles;
    |pos| is the per-distance, per-component max of the positions the level
    starts from -- the bin-1 checkpoint, upsampled by Reader.read_checkpoint.
    """
    cfg = RUNS[run]
    nd  = cfg["ndist"]
    with h5py.File(cfg["src"], "r") as f:
        z1 = f["exchange/z1"][:nd].astype("float64")
        sh = (_shrink3(f["exchange/shrink"][:, :nd])
              if "exchange/shrink" in f else np.zeros((1, nd, 2)))
    norm_mag = (z1[0] / z1)                       # magnifications / mag[0]
    ed = ((1.0 + sh) / norm_mag[None, :, None]).max(axis=0)   # [nd, 2] (y, x)
    with h5py.File(ckpt(run, it), "r") as f:
        pos   = f["pos"][:, :nd].astype("float64")
        scale = NOBJ / f["prb_abs"].shape[-1]     # read_pos_checkpoint
    r = np.abs(pos * scale).max(axis=0)           # [nd, 2], (y, x)
    return ed, r



MASK_THETAS = DATA_THETAS         # the same four angles fig 00 shows
MASK_IT     = 1280                # bin 0 starts here; positions come from the
                                  # bin-1 checkpoint, upsampled x2 on read
MASK_SUB    = 4                   # the 2048^2 masks are drawn at 512^2
LD_C        = "#1baf7a"


def _tint(c, f=0.72):
    """c blended f of the way towards white -- a fill light enough to label."""
    r, g, b = mcolors.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def mask_per_angle(run, it=MASK_IT):
    """eff_demag and the starting positions, per angle and per distance.

    The same numbers precalc works from: eff_demag = (1 + shrink) /
    norm_magnification, and the positions the level starts at -- read from
    the bin-1 checkpoint and upsampled by Reader.read_pos_checkpoint.
    """
    cfg, nd = RUNS[run], RUNS[run]["ndist"]
    with h5py.File(cfg["src"], "r") as f:
        z1 = f["exchange/z1"][:nd].astype("float64")
        sh = (_shrink3(f["exchange/shrink"][:, :nd])
              if "exchange/shrink" in f else None)
        theta = f["exchange/theta"][:, 0].astype("float64")
    with h5py.File(ckpt(run, it), "r") as f:
        pos = (f["pos"][:, :nd].astype("float64")
               * (NOBJ / f["prb_abs"].shape[-1]))
    if sh is None:
        sh = np.zeros_like(pos)
    return (1.0 + sh) / (z1[0] / z1)[None, :, None], pos, theta


def mask_exact(n, nobj, ed, r, margin=MASK_MARGIN):
    """The box _build_data_mask builds for one (angle, distance) pair.

    A detector pixel survives only if all four cubic B-spline taps of its
    back-mapped position stay inside the object grid, i.e. 1 <= x < nobj - 2,
    shrunk by `margin` at both ends.  No worst case over angles: since
    2026-08 this asymmetric, per-angle interval is exactly what the solver
    applies -- one rectangle per (distance, angle), frozen for the level.
    """
    ed = np.atleast_1d(ed)
    t = np.arange(n, dtype="float64") - (n - 1) * 0.5
    y = float(ed[0])  * t - r[0] + (nobj - 1) * 0.5
    x = float(ed[-1]) * t - r[1] + (nobj - 1) * 0.5
    return ((y >= 1.0 + margin) & (y < nobj - 2.0 - margin),
            (x >= 1.0 + margin) & (x < nobj - 2.0 - margin))


def mask_frac_all_angles(n, nobj, ed_t, pos_t, margin=MASK_MARGIN):
    """Mean kept fraction of mask_exact over every angle -- all 1800, not the
    four the figure draws."""
    t = np.arange(n, dtype="float64") - (n - 1) * 0.5
    f = 1.0
    for a in (0, 1):
        v = ed_t[:, a][:, None] * t[None, :] - pos_t[:, a][:, None] \
            + (nobj - 1) * 0.5
        f = f * ((v >= 1.0 + margin) & (v < nobj - 2.0 - margin)).mean(axis=1)
    return float(f.mean())


def _mask_cell(ax, my, mx, color, frozen=None, n=2048):
    """One keep/drop map, optionally with the applied mask outlined on it."""
    img = np.outer(my[::MASK_SUB], mx[::MASK_SUB])
    ax.imshow(img, cmap=mcolors.ListedColormap(["#efe3e1", _tint(color)]),
              vmin=0, vmax=1, interpolation="nearest")
    if frozen is not None and frozen[0].any():
        fx, fy = frozen                       # mask_axes returns (keep_x, keep_y)
        ax.add_patch(Rectangle(
            (np.argmax(fx) / MASK_SUB - 0.5, np.argmax(fy) / MASK_SUB - 0.5),
            fx.sum() / MASK_SUB, fy.sum() / MASK_SUB, fill=False,
            ec=DROP_C, lw=0.9, ls=(0, (3, 2)), zorder=3))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#9a9a95"); s.set_linewidth(0.6)
    return 100.0 * mx.mean() * my.mean()


def fig_masking():
    """Detector masks per angle and per distance, both acquisitions."""
    n, nc = 2048, len(MASK_THETAS) + 1
    rows = []
    for run, tag, cols in (("HT_4dist", "AtomiumL1_HT", DIST_C),
                           ("largedisp", "large_disp", [LD_C])):
        ed_t, pos, theta = mask_per_angle(run)
        ed_max, r_max = mask_params(run)
        for k in range(RUNS[run]["ndist"]):
            rows.append(dict(tag=tag, k=k, theta=theta, ed=ed_t[:, k],
                             pos=pos[:, k], color=cols[k % len(cols)],
                             mean=100.0 * mask_frac_all_angles(
                                 n, NOBJ, ed_t[:, k], pos[:, k]),
                             frozen=mask_axes(n, n, NOBJ, NOBJ,
                                              ed_max[k], r_max[k])))

    fig = plt.figure(figsize=(9.4, 10.9))
    gs = fig.add_gridspec(len(rows), nc, hspace=0.16, wspace=0.06,
                          left=0.140, right=0.988, top=0.893, bottom=0.112)
    kept = np.zeros((len(rows), nc))
    for i, row in enumerate(rows):
        first = None
        for j, t in enumerate(MASK_THETAS):
            ax = fig.add_subplot(gs[i, j])
            first = first or ax
            my, mx = mask_exact(n, NOBJ, row["ed"][t], row["pos"][t])
            kept[i, j] = _mask_cell(ax, my, mx, row["color"], row["frozen"])
            ax.set_title(f"{kept[i, j]:.1f}% kept", fontsize=6.4, pad=2.2,
                         color="#52514e")
            if i == 0:
                bb = ax.get_position()
                fig.text(bb.x0 + bb.width / 2, 0.914, r"$\theta$ = "
                         + f"{row['theta'][t]:.1f}" + r"$^\circ$",
                         fontsize=8.0, ha="center", va="bottom",
                         fontweight="bold", color="#0b0b0b")
        ax = fig.add_subplot(gs[i, nc - 1])
        kept[i, -1] = _mask_cell(ax, row["frozen"][1], row["frozen"][0],
                                 row["color"])
        ax.set_title(f"{kept[i, -1]:.1f}% kept", fontsize=6.4, pad=2.2,
                     color=DROP_C)
        if i == 0:
            bb = ax.get_position()
            fig.text(bb.x0 + bb.width / 2, 0.914,
                     r"one box for all $\theta$",
                     fontsize=8.0, ha="center", va="bottom",
                     fontweight="bold", color=DROP_C)
        bb = first.get_position()
        fig.text(0.132, (bb.y0 + bb.y1) / 2,
                 f"{row['tag']}\ndist {row['k']}\n"
                 f"eff_demag {row['ed'].mean():.3f}\n"
                 f"{row['mean']:.1f}% kept, all $\\theta$",
                 fontsize=7.2, ha="right", va="center", color=row["color"],
                 fontweight="bold", linespacing=1.5)

    # group headers: the first four columns are what runs now, the last is
    # the single rectangle the solver used before the mask went per-angle
    top_row = fig.axes[:nc]
    l, r_ = top_row[0].get_position(), top_row[nc - 2].get_position()
    fig.text((l.x0 + r_.x1) / 2, 0.936,
             "what is actually used  --  one frozen box per angle",
             fontsize=8.4, ha="center", va="bottom", fontweight="bold",
             color="#0b0b0b")
    b = top_row[nc - 1].get_position()
    fig.text(b.x0 + b.width / 2, 0.936,
             "previous behaviour", fontsize=8.4, ha="center",
             va="bottom", fontweight="bold", color=DROP_C)

    fig.suptitle("Out-of-grid detector mask  (Rec._build_data_mask)  --  "
                 f"AtomiumL1, bin 0, n = {n}, "
                 f"$n_{{obj}}$ = {NOBJ}, margin = {MASK_MARGIN:.0f} px",
                 fontsize=10.5, y=0.982)
    fig.text(0.5, 0.952,
             r"$x = \mathrm{eff\_demag}\,(t_x - \frac{n-1}{2}) - r_x + "
             r"\frac{n_{obj}-1}{2}$,     keep the pixel only if all four "
             r"cubic B-spline taps land in the grid: "
             r"$1 + m \leq x < n_{obj} - 2 - m$  ($m$ = margin)",
             fontsize=8.2, ha="center", va="bottom", color="#52514e")

    note = [
        "Blue/green: detector pixels that keep a weight at that angle and "
        "that distance.  Pink: pixels whose model value would come from the "
        "shift kernel's boundary condition",
        "-- a mirrored copy of the sample -- rather than from the sample "
        "itself.  The support is always a rectangle: it shrinks with "
        "eff_demag and it slides with the sample position $r$.",
        "Columns 1-4 are what the solver applies.  It keeps one rectangle "
        "per (distance, angle) -- the exact, off-centre interval above, "
        f"shrunk by a {MASK_MARGIN:.0f} px margin -- built once",
        "before the loop from the starting positions and then frozen for the "
        "whole level, so $F_0$ stays comparable across iterations.  Stored as "
        "four integers per angle, never as an array.",
        "The last column is what ran before 2026-08: ONE rectangle per "
        "distance, the intersection of all 1800 per-angle supports, applied "
        "to every angle (dashed red above).  Letting",
        "the box slide with the sample is the whole gain -- it is worth "
        "little where eff_demag does the cropping (rows 1-4) and a factor "
        "1.77 on large_disp, which loses the field",
        f"instead to a $\\pm$300 px displacement against $n_{{obj}}$ = n = "
        f"{NOBJ}, a grid exactly the detector's size.  $F_0$ still divides by "
        "the full $n_\\theta n_{dist} n_z n$, so a run's",
        "error scales with the kept fraction and is comparable neither with "
        "an unmasked run nor with one made before the mask went per-angle.  "
        "mask_oob=0 turns masking off.",
    ]
    fig.text(0.5, 0.006, "\n".join(note), fontsize=6.7, ha="center",
             va="bottom", color="#52514e", linespacing=1.58)
    save(fig, "fig00b_data_mask.png", tight=False)
    for i, row in enumerate(rows):
        print(f"    {row['tag']:>13s} dist {row['k']}: applied per-angle "
              + ", ".join(f"{v:.1f}%" for v in kept[i, :-1])
              + f" (all-theta mean {row['mean']:.1f}%)"
              + f"   legacy single box {kept[i, -1]:.1f}%")



# --------------------------------------------------------------------------
# fig 00c - the same masks, drawn on the holograms they act on
# --------------------------------------------------------------------------
def _data_cell(ax, img, my, mx, frozen=None, alpha=0.60):
    """One hologram with its dropped pixels washed out in red."""
    n = len(mx)
    step = n // img.shape[-1]                 # detector px per displayed px
    ax.imshow(img, cmap="gray", vmin=DATA_VLIM[0], vmax=DATA_VLIM[1],
              interpolation="nearest")
    drop = ~np.outer(my[::step], mx[::step])
    wash = np.zeros(drop.shape + (4,))
    wash[..., :3] = mcolors.to_rgb(DROP_C)
    wash[..., 3] = alpha * drop
    ax.imshow(wash, interpolation="nearest")
    if frozen is not None and frozen[0].any():
        fx, fy = frozen
        ax.add_patch(Rectangle(
            (np.argmax(fx) / step - 0.5, np.argmax(fy) / step - 0.5),
            fx.sum() / step, fy.sum() / step, fill=False,
            ec="#ffd400", lw=1.0, ls=(0, (3, 2)), zorder=3))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#9a9a95"); s.set_linewidth(0.6)
    return 100.0 * mx.mean() * my.mean()


def fig_masking_data():
    """The masks of fig 00b, laid over the holograms they actually multiply."""
    n, nc = 2048, len(MASK_THETAS) + 1
    rows = []
    for run, tag, cols in (("HT_4dist", "AtomiumL1_HT", DIST_C),
                           ("largedisp", "large_disp", [LD_C])):
        ed_t, pos, theta = mask_per_angle(run)
        ed_max, r_max = mask_params(run)
        for k in range(RUNS[run]["ndist"]):
            rows.append(dict(run=run, tag=tag, k=k, theta=theta,
                             ed=ed_t[:, k], pos=pos[:, k],
                             color=cols[k % len(cols)],
                             mean=100.0 * mask_frac_all_angles(
                                 n, NOBJ, ed_t[:, k], pos[:, k]),
                             frozen=mask_axes(n, n, NOBJ, NOBJ,
                                              ed_max[k], r_max[k])))

    fig = plt.figure(figsize=(9.4, 10.9))
    gs = fig.add_gridspec(len(rows), nc, hspace=0.16, wspace=0.06,
                          left=0.140, right=0.988, top=0.893, bottom=0.112)
    kept = np.zeros((len(rows), nc))
    for i, row in enumerate(rows):
        first = None
        for j, t in enumerate(MASK_THETAS):
            ax = fig.add_subplot(gs[i, j])
            first = first or ax
            img = hologram(row["run"], row["k"], t)
            my, mx = mask_exact(n, NOBJ, row["ed"][t], row["pos"][t])
            kept[i, j] = _data_cell(ax, img, my, mx, row["frozen"])
            ax.set_title(f"{kept[i, j]:.1f}% kept", fontsize=6.4, pad=2.2,
                         color="#52514e")
            if i == 0:
                bb = ax.get_position()
                fig.text(bb.x0 + bb.width / 2, 0.914, r"$\theta$ = "
                         + f"{row['theta'][t]:.1f}" + r"$^\circ$",
                         fontsize=8.0, ha="center", va="bottom",
                         fontweight="bold", color="#0b0b0b")
        # last column: the frozen mask, on the theta = 0 hologram
        ax = fig.add_subplot(gs[i, nc - 1])
        img = hologram(row["run"], row["k"], MASK_THETAS[0])
        kept[i, -1] = _data_cell(ax, img, row["frozen"][1], row["frozen"][0])
        ax.set_title(f"{kept[i, -1]:.1f}% kept", fontsize=6.4, pad=2.2,
                     color=DROP_C)
        if i == 0:
            bb = ax.get_position()
            fig.text(bb.x0 + bb.width / 2, 0.914,
                     r"one box for all $\theta$",
                     fontsize=8.0, ha="center", va="bottom",
                     fontweight="bold", color=DROP_C)
        bb = first.get_position()
        fig.text(0.132, (bb.y0 + bb.y1) / 2,
                 f"{row['tag']}\ndist {row['k']}\n"
                 f"eff_demag {row['ed'].mean():.3f}\n"
                 f"{row['mean']:.1f}% kept, all $\\theta$",
                 fontsize=7.2, ha="right", va="center", color=row["color"],
                 fontweight="bold", linespacing=1.5)

    # group headers: the first four columns are what runs now, the last is
    # the single rectangle the solver used before the mask went per-angle
    top_row = fig.axes[:nc]
    l, r_ = top_row[0].get_position(), top_row[nc - 2].get_position()
    fig.text((l.x0 + r_.x1) / 2, 0.936,
             "what is actually used  --  one frozen box per angle",
             fontsize=8.4, ha="center", va="bottom", fontweight="bold",
             color="#0b0b0b")
    b = top_row[nc - 1].get_position()
    fig.text(b.x0 + b.width / 2, 0.936,
             "previous behaviour", fontsize=8.4, ha="center",
             va="bottom", fontweight="bold", color=DROP_C)

    fig.suptitle("The out-of-grid mask on the data it multiplies  --  "
                 f"AtomiumL1, bin 0, n = {n}, "
                 f"$n_{{obj}}$ = {NOBJ}, margin = {MASK_MARGIN:.0f} px",
                 fontsize=10.5, y=0.982)
    fig.text(0.5, 0.952,
             "Flat-corrected holograms divided by the reference, grey window "
             f"[{DATA_VLIM[0]:.2f}, {DATA_VLIM[1]:.2f}] of $I/I_{{ref}}$, "
             "shown at 2$\\times$ binning;  red wash = the pixels the mask "
             "drops",
             fontsize=8.2, ha="center", va="bottom", color="#52514e")

    note = [
        "The masks of the previous figure, laid over the holograms they act "
        "on.  The red wash marks the pixels the mask drops: their model value "
        "would be manufactured by the",
        "shift kernel's boundary condition -- a mirrored copy of the sample "
        "-- instead of coming from the sample, so the solver zeroes their "
        "residual.  Columns 1-4 are what it applies:",
        "one frozen, off-centre box per (distance, angle).  The yellow dashes "
        "are the single box used for every angle before 2026-08; the last "
        "column is that box alone, on the $\\theta$ = 0$^\\circ$ hologram.",
        "Two different causes.  Rows 1-4: eff_demag > 1 spreads the detector "
        f"over more object voxels than the {NOBJ}$^2$ grid holds, so the "
        "field is cropped, hard at dist 3.",
        "Last row: eff_demag = 1 as in row 1, but large_disp moves the sample "
        f"by up to $\\pm$300 px while $n_{{obj}}$ = n = {NOBJ}, so the "
        "window slides off a grid exactly its own size --",
        "which is why the last row does not look like the first.  "
        f"$n_{{obj}}$ = 2688 would keep all of it; the configs as they stand "
        f"run $n_{{obj}}$ = n, so the mask is what makes that survivable.",
    ]
    fig.text(0.5, 0.006, "\n".join(note), fontsize=6.7, ha="center",
             va="bottom", color="#52514e", linespacing=1.58)
    save(fig, "fig00c_data_mask_on_data.png", tight=False)
    for i, row in enumerate(rows):
        print(f"    {row['tag']:>13s} dist {row['k']}: applied per-angle "
              + ", ".join(f"{v:.1f}%" for v in kept[i, :-1])
              + f" (all-theta mean {row['mean']:.1f}%)"
              + f"   legacy single box {kept[i, -1]:.1f}%")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    # one fixed grey window for every run and every iteration, so nothing is
    # rescaled away between panels
    vlims = {run: VLIM for run in RUNS}
    print(f"display window {VLIM[0]:+.2f} .. {VLIM[1]:+.2f} (all runs)")

    fig_data()
    fig_masking()
    fig_masking_data()
    fig_convergence()
    fig_convergence_full()
    fig_recon_compare(vlims)
    fig_recon_detail(vlims)
    fig_recon_detail_median(vlims)
    fig_probes()
    fig_positions()
    fig_positions_summary()


if __name__ == "__main__":
    main()
