#!/usr/bin/env python3
import argparse
import os
import shutil
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare txt-based STPLS3D groundtruth/original directories from 50x50 block txt files, "
            "so downstream CityIns3D code can use txt files with a directory layout similar to the old ply version."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="/data1/wangcl/dataset/open_3d/STPLS3D_Open3DIS/3D/Synthetic_v3_InstanceSegmentation/stpls3d_block_50",
        help="Directory containing block txt files such as 05_points_GTv3_00.txt.",
    )
    parser.add_argument(
        "--output-root",
        default="/data1/wangcl/dataset/open_3d/STPLS3D_Open3DIS/3D/stpls3d_txt_version",
        help="Root directory for txt-version groundtruth/original folders.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Optional scene stem without extension. Can be passed multiple times.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="How to materialize files into target directories.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_input_files(input_dir, scene_names):
    if scene_names:
        return [os.path.join(input_dir, f"{scene}.txt") for scene in scene_names]
    return sorted(glob(os.path.join(input_dir, "*.txt")))


def safe_remove(path):
    if not os.path.lexists(path):
        return
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.unlink(path)


def materialize(src, dst, mode):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    args = parse_args()

    source_files = resolve_input_files(args.input_dir, args.scene)
    if not source_files:
        raise FileNotFoundError(f"No txt files found in {args.input_dir}")

    original_dir = os.path.join(args.output_root, "original_ply_files")
    gt_dir = os.path.join(args.output_root, "groundtruth")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    scene_names = []
    for src in source_files:
        scene_name = os.path.splitext(os.path.basename(src))[0]
        scene_names.append(scene_name)
        for target_dir in (original_dir, gt_dir):
            dst = os.path.join(target_dir, f"{scene_name}.txt")
            if os.path.lexists(dst):
                if not args.overwrite:
                    continue
                safe_remove(dst)
            materialize(src, dst, args.link_mode)

    split_path = os.path.join(args.output_root, "scene_list.txt")
    with open(split_path, "w", encoding="utf-8") as handle:
        for scene_name in scene_names:
            handle.write(scene_name + "\n")

    print(f"[done] scenes={len(scene_names)}")
    print(f"[done] original_ply_files={original_dir}")
    print(f"[done] groundtruth={gt_dir}")
    print(f"[done] scene_list={split_path}")


if __name__ == "__main__":
    main()
