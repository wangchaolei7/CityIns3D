#!/usr/bin/env python3
import argparse
import os
from glob import glob

import numpy as np
from tqdm import tqdm


def split_point_cloud(points, size=50.0, stride=50.0):
    limit_max = np.amax(points[:, 0:3], axis=0)
    limit_min = np.amin(points[:, 0:3], axis=0)
    width = int(np.ceil((limit_max[0] - limit_min[0] - size) / stride)) + 1
    depth = int(np.ceil((limit_max[1] - limit_min[1] - size) / stride)) + 1
    width = max(width, 1)
    depth = max(depth, 1)
    cells = [
        (limit_min[0] + x * stride, limit_min[1] + y * stride)
        for x in range(width)
        for y in range(depth)
    ]

    blocks = []
    for block_idx, (x0, y0) in enumerate(cells):
        xcond = (points[:, 0] >= x0) & (points[:, 0] <= x0 + size)
        ycond = (points[:, 1] >= y0) & (points[:, 1] <= y0 + size)
        mask = xcond & ycond
        block = points[mask]
        blocks.append((block_idx, block, (x0, y0)))
    return blocks


def pad_scene_prefix(stem):
    parts = stem.split("_", 1)
    if not parts:
        return stem
    prefix = parts[0]
    if prefix.isdigit():
        prefix = prefix.zfill(2)
    return prefix if len(parts) == 1 else f"{prefix}_{parts[1]}"


def load_txt_points(path):
    points = np.genfromtxt(path, delimiter=",", dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 8:
        raise ValueError(f"Unexpected txt shape for {path}: {points.shape}")
    return points[:, :8]


def save_block_txt(block, output_path):
    np.savetxt(
        output_path,
        block,
        fmt=["%.6f", "%.6f", "%.6f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f"],
        delimiter=",",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split STPLS3D Synthetic_v3_InstanceSegmentation txt scenes into 50x50m blocks "
            "using ISBNet-style non-overlapping XY crops, and save blocks as .txt."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="/data1/wangcl/dataset/open_3d/STPLS3D_Open3DIS/3D/Synthetic_v3_InstanceSegmentation",
        help="Directory containing source txt scenes.",
    )
    parser.add_argument(
        "--output-dir",
        default="/data1/wangcl/dataset/open_3d/STPLS3D_Open3DIS/3D/Synthetic_v3_InstanceSegmentation/stpls3d_block_50",
        help="Directory to save split txt blocks.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Optional scene stem without extension, e.g. 5_points_GTv3. Can be passed multiple times.",
    )
    parser.add_argument("--crop-size", type=float, default=50.0)
    parser.add_argument("--stride", type=float, default=50.0)
    parser.add_argument(
        "--min-points",
        type=int,
        default=10000,
        help="Skip blocks with fewer points, following ISBNet prepare_data logic.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_input_files(input_dir, scene_names):
    if scene_names:
        return [os.path.join(input_dir, f"{scene}.txt") for scene in scene_names]
    return sorted(glob(os.path.join(input_dir, "*.txt")))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    input_files = resolve_input_files(args.input_dir, args.scene)
    if not input_files:
        raise FileNotFoundError(f"No txt files found in {args.input_dir}")

    saved_scene_names = []
    total_saved = 0
    for txt_path in tqdm(input_files):
        stem = os.path.splitext(os.path.basename(txt_path))[0]
        base_name = pad_scene_prefix(stem)
        points = load_txt_points(txt_path)
        blocks = split_point_cloud(points, size=args.crop_size, stride=args.stride)

        kept = 0
        for _, block, _ in blocks:
            if block.shape[0] < args.min_points:
                continue
            scene_name = f"{base_name}_{kept:02d}"
            output_path = os.path.join(args.output_dir, f"{scene_name}.txt")
            if os.path.exists(output_path) and not args.overwrite:
                saved_scene_names.append(scene_name)
                kept += 1
                total_saved += 1
                continue
            if not args.dry_run:
                save_block_txt(block, output_path)
            saved_scene_names.append(scene_name)
            kept += 1
            total_saved += 1

        print(f"[split] {stem} -> kept_blocks={kept}")

    split_list_path = os.path.join(args.output_dir, "scene_list.txt")
    if not args.dry_run:
        with open(split_list_path, "w") as f:
            for scene_name in saved_scene_names:
                f.write(scene_name + "\n")

    print(f"[done] saved_blocks={total_saved} output_dir={args.output_dir}")
    print(f"[done] scene_list={split_list_path}")


if __name__ == "__main__":
    main()
