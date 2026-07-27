#!/usr/bin/env python3
"""
Convert a FAST-LIO2 PCD map to a 2D occupancy grid (pgm + yaml) for Nav2.

Usage (in map_tools conda env):
    conda activate map_tools
    python3 scripts/pcd_to2d_map.py \
        --input  /root/data/fast_lio_map.pcd \
        --output /root/data/fast_lio_map_2d \
        --resolution 0.05 --z_min -0.25 --z_max 2.0 \
        --min_points 2 --outlier_neighbors 20 --outlier_std 2.0
"""

import argparse

import numpy as np
import open3d as o3d
import yaml
from PIL import Image
from scipy.ndimage import binary_closing, binary_opening


def pcd_to_2d_map(pcd_file, out_prefix, resolution, z_min, z_max,
                  min_points, outlier_neighbors, outlier_std):
    pcd = o3d.io.read_point_cloud(pcd_file)
    print(f'Loaded {len(pcd.points)} points from {pcd_file}')

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=outlier_neighbors, std_ratio=outlier_std)
    print(f'{len(pcd.points)} points after statistical outlier removal')

    pts = np.asarray(pcd.points)
    pts = pts[(pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)]
    print(f'{len(pts)} points remain after z-slice [{z_min}, {z_max}]')

    if len(pts) == 0:
        raise ValueError('No points in z range — check z_min/z_max values')

    x, y = pts[:, 0], pts[:, 1]
    ox = x.min() - 0.5
    oy = y.min() - 0.5
    W = int((x.max() - ox + 0.5) / resolution) + 1
    H = int((y.max() - oy + 0.5) / resolution) + 1

    px = ((x - ox) / resolution).astype(int)
    py = ((y - oy) / resolution).astype(int)
    py_img = H - 1 - py                            # flip y for image coords

    counts = np.zeros((H, W), dtype=np.int32)
    np.add.at(counts, (py_img, px), 1)
    occupied = counts >= min_points

    # Close small gaps in walls, then remove isolated noise specks
    occupied = binary_closing(occupied, iterations=1)
    occupied = binary_opening(occupied, iterations=1)

    grid = np.where(occupied, 0, 255).astype(np.uint8)  # 0=occupied, 255=free

    pgm_path = out_prefix + '.pgm'
    yaml_path = out_prefix + '.yaml'

    Image.fromarray(grid).save(pgm_path)

    meta = {
        'image': pgm_path,
        'resolution': float(resolution),
        'origin': [float(ox), float(oy), 0.0],
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196,
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(meta, f, default_flow_style=False)

    print(f'Saved {W}x{H} map → {pgm_path}')
    print(f'Map yaml → {yaml_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert FAST-LIO2 PCD map to 2D Nav2 map')
    parser.add_argument('--input',             default='/root/data/fast_lio_map.pcd',
                        help='Input PCD file path')
    parser.add_argument('--output',            default='/root/data/fast_lio_map_2d',
                        help='Output prefix (no extension)')
    parser.add_argument('--resolution',        type=float, default=0.05,
                        help='Map resolution in metres per pixel (default: 0.05)')
    parser.add_argument('--z_min',             type=float, default=-0.25,
                        help='Minimum z height to include (default: -0.25)')
    parser.add_argument('--z_max',             type=float, default=2.0,
                        help='Maximum z height to include (default: 2.0)')
    parser.add_argument('--min_points',        type=int,   default=2,
                        help='Min hits per cell to mark as occupied (default: 2)')
    parser.add_argument('--outlier_neighbors', type=int,   default=20,
                        help='Neighbours for statistical outlier removal (default: 20)')
    parser.add_argument('--outlier_std',       type=float, default=2.0,
                        help='Std-ratio threshold for outlier removal (default: 2.0)')
    args = parser.parse_args()

    pcd_to_2d_map(args.input, args.output, args.resolution, args.z_min, args.z_max,
                  args.min_points, args.outlier_neighbors, args.outlier_std)
