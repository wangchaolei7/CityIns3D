import argparse
import os
import time

import yaml
from munch import Munch

from util3d.utonia_superpoints import UtoniaSuperpointGenerator


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate Utonia superpoints with local voxel-graph over-segmentation."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml.")
    parser.add_argument("--scene", type=str, default=None, help="Only process one scene id.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as handle:
        return Munch.fromDict(yaml.safe_load(handle.read()))


def load_scene_ids(cfg, scene_override=None):
    if scene_override is not None:
        return [scene_override]
    with open(cfg.data.split_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def main():
    args = get_parser().parse_args()
    cfg = load_config(args.config)
    generator = UtoniaSuperpointGenerator(cfg)

    scene_ids = load_scene_ids(cfg, args.scene)
    for scene_id in scene_ids:
        output_dir = getattr(cfg.superpoint, "save_dir", cfg.data.spp_path)
        label_path = os.path.join(output_dir, f"{scene_id}.pth")
        if not args.overwrite and os.path.exists(label_path):
            print(f"[Utonia-SP] exists, skip: {scene_id}")
            continue

        start_time = time.perf_counter()
        point_labels, stats = generator.generate_scene(scene_id)
        label_path = generator.save_scene(scene_id, point_labels)
        elapsed = time.perf_counter() - start_time

        print(
            f"[Utonia-SP] saved {scene_id} "
            f"labels={label_path} "
            f"num_points={stats['num_points']} "
            f"num_voxels={stats['num_voxels']} "
            f"num_clusters={stats['num_clusters']} "
            f"num_superpoints={stats['num_superpoints']} "
            f"noise={stats['num_noise_points']} "
            f"mode={stats['cluster_mode']} "
            f"edges={stats.get('num_graph_edges', 0)} "
            f"kept_edges={stats.get('num_kept_edges', 0)} "
            f"small_merged={stats.get('num_small_merged', 0)} "
            f"saved_spp_feat={stats['used_saved_spp_feat']} "
            f"elapsed={elapsed:.2f}s"
        )


if __name__ == "__main__":
    main()
