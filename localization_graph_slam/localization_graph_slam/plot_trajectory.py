#!/usr/bin/env python3
"""
Plot robot trajectories on a single figure.

── Auto-discovery ────────────────────────────────────────────────────────────
  python3 plot_trajectory.py --base-dir ~/slam_data/real [--save result/real]
  python3 plot_trajectory.py --base-dir ~/slam_data/sim  [--save result/sim]

  Real robot — scans for subdirs combined/, imu_only/, line_only/:
    combined/odometry_trajectory.csv   ← dead-reckoning odometry
    imu_only/slam_trajectory.csv       ← GraphSLAM with IMU only
    line_only/slam_trajectory.csv      ← GraphSLAM with line features only
    combined/slam_trajectory.csv       ← GraphSLAM combined

  Simulation — same subdirs, plus ground_truth_trajectory.csv in each:
    combined/slam_trajectory.csv
    combined/ground_truth_trajectory.csv   (GT taken from first available dir)
    imu_only/slam_trajectory.csv
    line_only/slam_trajectory.csv

  Any subset may be absent — missing files are silently skipped.

── Explicit paths ─────────────────────────────────────────────────────────────
  python3 plot_trajectory.py \\
      [--gt        <path>]   \\
      [--odom      <path>]   \\
      [--imu-slam  <path>]   \\
      [--line-slam <path>]   \\
      [--combined  <path>]   \\
      [--save result/sim]
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── colours & styles ──────────────────────────────────────────────────────────

STYLES = {
    'gt':        dict(color='#000000', lw=2.2, ls='-',  label='Ground Truth',                zorder=5),
    'combined':  dict(color='#8B0000', lw=1.8, ls='-',  label='GraphSLAM — Combined',        zorder=4),
    'imu_only':  dict(color='#2196F3', lw=1.4, ls='-',  label='GraphSLAM — IMU only',        zorder=3),
    'line_only': dict(color='#4CAF50', lw=1.4, ls='-',  label='GraphSLAM — Line features',   zorder=3),
    'odometry':  dict(color='#888888', lw=1.2, ls='--', label='Odometry (dead-reckoning)',   zorder=2),
}

# Draw order for the legend / rendering (GT first, then best to worst)
DRAW_ORDER = ['gt', 'combined', 'imu_only', 'line_only', 'odometry']


# ── I/O helpers ───────────────────────────────────────────────────────────────

def try_load(path):
    """Return structured numpy array or None if file is missing / unreadable."""
    if not path or not os.path.isfile(path):
        return None
    try:
        data = np.genfromtxt(path, delimiter=',', names=True)
        if data.ndim == 0 or data.size == 0:
            return None
        data = data[np.argsort(data['time_s'])]
        return data
    except Exception as e:
        print(f"[warn] could not load {path}: {e}")
        return None


def xy(data):
    return data['x'], data['y']


# ── Auto-discovery helpers ────────────────────────────────────────────────────

def _discover(base, paths):
    """Fill missing entries in `paths` by scanning the base directory."""
    base = os.path.expanduser(base)

    # Variant subdirectory layout (both sim and real)
    variant_map = {
        'combined':  ('slam_trajectory.csv',   'combined'),
        'imu_only':  ('slam_trajectory.csv',   'imu_only'),
        'line_only': ('slam_trajectory.csv',   'line_only'),
        'odometry':  ('odometry_trajectory.csv', 'combined'),
    }
    for key, (fname, subdir) in variant_map.items():
        if not paths.get(key):
            candidate = os.path.join(base, subdir, fname)
            if os.path.isfile(candidate):
                paths[key] = candidate

    # Ground truth — pick from first subdir that has it (sim only)
    if not paths.get('gt'):
        for subdir in ('combined', 'imu_only', 'line_only', ''):
            candidate = os.path.join(base, subdir, 'ground_truth_trajectory.csv')
            if os.path.isfile(candidate):
                paths['gt'] = candidate
                break

    # Legacy flat layout (backward compat — no subdirs)
    if not paths.get('combined'):
        candidate = os.path.join(base, 'slam_trajectory.csv')
        if os.path.isfile(candidate):
            paths['combined'] = candidate
    if not paths.get('gt'):
        candidate = os.path.join(base, 'ground_truth_trajectory.csv')
        if os.path.isfile(candidate):
            paths['gt'] = candidate
    if not paths.get('odometry'):
        candidate = os.path.join(base, 'odometry_trajectory.csv')
        if os.path.isfile(candidate):
            paths['odometry'] = candidate

    return paths


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_trajectories(sources: dict, save_path=None):
    """
    sources: dict mapping style-key → numpy structured array (or None).
    """
    present = {k: sources[k] for k in DRAW_ORDER if sources.get(k) is not None}
    if not present:
        print("No trajectory data to plot.")
        return

    is_sim = 'gt' in present

    title = ('GraphSLAM — Trajectory Comparison (Simulation)'
             if is_sim else
             'GraphSLAM — Trajectory Comparison (Real Robot)')

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    for key, data in present.items():
        style = STYLES[key]
        x, y = xy(data)
        ax.plot(x, y, **style)
        ax.plot(x[0],  y[0],  'o', color=style['color'], ms=6,  zorder=10)
        ax.plot(x[-1], y[-1], 'x', color=style['color'], ms=8,  mew=2, zorder=10)

    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('y (m)', fontsize=11)
    ax.legend(fontsize=9, loc='best')
    ax.set_aspect('equal')
    ax.grid(True, lw=0.4, alpha=0.7)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot SLAM trajectory variants on one figure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--base-dir',  default=None,
                        help='Base data directory for auto-discovery')
    parser.add_argument('--gt',        default=None, help='Ground-truth CSV (sim)')
    parser.add_argument('--odom',      default=None, help='Odometry CSV path')
    parser.add_argument('--imu-slam',  default=None, help='IMU-only SLAM CSV path')
    parser.add_argument('--line-slam', default=None, help='Line-only SLAM CSV path')
    parser.add_argument('--combined',  default=None, help='Combined SLAM CSV path')
    parser.add_argument('--save',      default=None,
                        help='Directory to save PNG (omit = interactive window)')
    args = parser.parse_args()

    paths = {
        'gt':        args.gt,
        'odometry':  args.odom,
        'imu_only':  args.imu_slam,
        'line_only': args.line_slam,
        'combined':  args.combined,
    }

    if args.base_dir:
        paths = _discover(args.base_dir, paths)

    if all(v is None for v in paths.values()):
        parser.error('Nothing found — provide --base-dir or at least one explicit path.')

    sources = {k: try_load(v) for k, v in paths.items()}

    loaded = [k for k, v in sources.items() if v is not None]
    if not loaded:
        sys.exit('No CSV files could be loaded — check paths.')
    print(f"Plotting: {', '.join(loaded)}")

    save_path = None
    if args.save:
        save_dir = os.path.expanduser(args.save)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'slam_trajectory_comparison.png')

    plot_trajectories(sources, save_path)


if __name__ == '__main__':
    main()
