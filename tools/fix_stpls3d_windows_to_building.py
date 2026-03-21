#!/usr/bin/env python3
import argparse
import os
from glob import glob

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge mis-labeled STPLS3D window points into nearby building instances. "
            "By default, semantic 17 is treated as window and merged into semantic 1."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="/data1/wangcl/dataset/open_3d/STPLS3D_Open3DIS/3D/groundtruth",
        help="Directory containing input groundtruth PLY files.",
    )
    parser.add_argument(
        "--output-dir",
        default="/data1/wangcl/dataset/open_3d/STPLS3D_Open3DIS/3D/groundtruth_window17_to_building",
        help="Directory to save fixed PLY files.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Optional scene id without .ply extension. Can be passed multiple times.",
    )
    parser.add_argument(
        "--window-sem",
        type=int,
        default=17,
        help="Semantic id to treat as windows and merge.",
    )
    parser.add_argument(
        "--building-sem",
        type=int,
        default=1,
        help="Target building semantic id.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=16,
        help="Number of nearest building points used for instance voting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def scene_paths(input_dir, scene_names):
    if scene_names:
        return [os.path.join(input_dir, f"{scene}.ply") for scene in scene_names]
    return sorted(glob(os.path.join(input_dir, "*.ply")))


def weighted_vote(target_instances, distances):
    weights = 1.0 / np.maximum(distances, 1e-6)
    score = {}
    for inst_id, weight in zip(target_instances.reshape(-1), weights.reshape(-1)):
        inst_id = int(inst_id)
        score[inst_id] = score.get(inst_id, 0.0) + float(weight)
    best_inst = max(score.items(), key=lambda item: item[1])[0]
    return best_inst


def merge_windows_into_buildings(vertex, window_sem=17, building_sem=1, k=16):
    coords = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    semantic = np.asarray(vertex["semantic"], dtype=np.int32).copy()
    instance = np.asarray(vertex["instance"], dtype=np.int32).copy()

    building_mask = semantic == building_sem
    window_mask = semantic == window_sem

    num_building = int(building_mask.sum())
    num_window = int(window_mask.sum())
    if num_window == 0:
        return semantic, instance, {
            "window_points": 0,
            "window_instances": 0,
            "reassigned_points": 0,
            "reassigned_instances": 0,
            "building_points": num_building,
        }
    if num_building == 0:
        raise RuntimeError("No building points found; cannot merge windows into buildings.")

    building_coords = coords[building_mask]
    building_instance = instance[building_mask]
    valid_building = building_instance >= 1000
    if not np.any(valid_building):
        raise RuntimeError("No valid building instance ids found.")
    building_coords = building_coords[valid_building]
    building_instance = building_instance[valid_building]

    tree = cKDTree(building_coords)
    k = max(1, min(int(k), building_coords.shape[0]))

    window_indices = np.flatnonzero(window_mask)
    window_instances = instance[window_mask]

    reassigned_instances = 0
    for win_inst in np.unique(window_instances):
        component_idx = window_indices[window_instances == win_inst]
        component_coords = coords[component_idx]
        distances, nn_idx = tree.query(component_coords, k=k, workers=-1)
        if k == 1:
            distances = distances[:, None]
            nn_idx = nn_idx[:, None]
        target_inst = weighted_vote(building_instance[nn_idx], distances)
        semantic[component_idx] = building_sem
        instance[component_idx] = target_inst
        reassigned_instances += 1

    return semantic, instance, {
        "window_points": num_window,
        "window_instances": int(len(np.unique(window_instances))),
        "reassigned_points": num_window,
        "reassigned_instances": reassigned_instances,
        "building_points": int(num_building),
    }


def save_ply_like(src_ply, semantic, instance, output_path):
    vertex = src_ply["vertex"].data.copy()
    vertex["semantic"] = semantic
    vertex["instance"] = instance
    element = PlyElement.describe(vertex, "vertex")
    PlyData([element], text=src_ply.text).write(output_path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    paths = scene_paths(args.input_dir, args.scene)
    if not paths:
        raise FileNotFoundError(f"No PLY files found in {args.input_dir}")

    total_window_points = 0
    total_window_instances = 0
    changed_scenes = 0

    for path in tqdm(paths):
        scene_name = os.path.basename(path)
        output_path = os.path.join(args.output_dir, scene_name)
        if os.path.exists(output_path) and not args.overwrite:
            continue

        ply = PlyData.read(path)
        semantic, instance, stats = merge_windows_into_buildings(
            ply["vertex"],
            window_sem=args.window_sem,
            building_sem=args.building_sem,
            k=args.k,
        )
        save_ply_like(ply, semantic, instance, output_path)

        total_window_points += stats["window_points"]
        total_window_instances += stats["window_instances"]
        if stats["window_points"] > 0:
            changed_scenes += 1
        print(
            f"[fix] {scene_name} window_points={stats['window_points']} "
            f"window_instances={stats['window_instances']} "
            f"building_points={stats['building_points']} "
            f"saved={output_path}"
        )

    print(
        f"[summary] scenes={len(paths)} changed_scenes={changed_scenes} "
        f"window_points={total_window_points} window_instances={total_window_instances}"
    )


if __name__ == "__main__":
    main()
