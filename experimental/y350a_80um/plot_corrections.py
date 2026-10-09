"""Plot motion correction files (correct.txt, correct_motion.txt) and save as PNGs.

Usage:
    python plot_corrections.py <root_dir>

Finds all correct*.txt files recursively under <root_dir> and saves one PNG
per file with x and y coordinate plots.
"""

import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_correction(fpath, out_dir):
    data = np.loadtxt(fpath)
    if data.ndim == 1:
        data = data[:, np.newaxis]

    title = os.path.relpath(fpath, out_dir)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(title, fontsize=10)

    labels = ['x', 'y']
    for ax, col, label in zip(axes, range(min(2, data.shape[1])), labels):
        ax.plot(data[:, col])
        ax.set_ylabel(f'{label} (px)')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('angle index')

    png_name = fpath.replace(os.sep, '_').replace('/', '_').lstrip('_') + '.png'
    out_path = os.path.join(out_dir, png_name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  saved → {out_path}')


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    out_dir = root

    paths = sorted(glob.glob(os.path.join(root, '**', 'correct*.txt'), recursive=True))
    if not paths:
        print(f'No correct*.txt files found under {root}')
        sys.exit(1)

    print(f'Found {len(paths)} file(s)')
    for p in paths:
        print(f'  {p}')
        plot_correction(p, out_dir)

    print('Done.')


if __name__ == '__main__':
    main()
