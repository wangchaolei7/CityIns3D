import argparse
import os
import sys
import time

import yaml
from munch import Munch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from util3d.tgc_superpoints import TgcSuperpointGenerator


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate STPLS3D superpoints using torch-graph-components."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml.")
    parser.add_argument("--scene", type=str, default=None, help="Only process one scene id.")
    parser.add_argument(
        "--method",
        type=str,
        default="tgc_utonia",
        choices=["tgc_native", "tgc_utonia"],
        help="Superpoint variant to generate.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional explicit output directory.")
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
    generator = TgcSuperpointGenerator(cfg)

    scene_ids = load_scene_ids(cfg, args.scene)
    output_dir = args.output_dir or generator._resolve_output_dir(args.method)
    os.makedirs(output_dir, exist_ok=True)

    for scene_id in scene_ids:
        label_path = os.path.join(output_dir, f"{scene_id}.pth")
        if not args.overwrite and os.path.exists(label_path):
            print(f"[TGC-SP] exists, skip: {scene_id}")
            continue

        start_time = time.perf_counter()
        point_labels, stats = generator.generate_scene(scene_id, args.method)
        label_path = generator.save_scene(scene_id, point_labels, args.method, output_dir=output_dir)
        elapsed = time.perf_counter() - start_time

        print(
            f"[TGC-SP] saved {scene_id} "
            f"method={args.method} "
            f"labels={label_path} "
            f"num_points={stats['num_points']} "
            f"num_voxels={stats['num_voxels']} "
            f"num_edges={stats['num_edges']} "
            f"num_superpoints={stats['num_superpoints']} "
            f"depth={stats['depth']} "
            f"device={stats['device']} "
            f"voxel={stats['voxel_size']:.3f} "
            f"reg={stats['reg']:.2f} "
            f"min_size={stats['min_size']} "
            f"used_utonia_feature={stats['used_utonia_feature']} "
            f"load={stats['load_time']:.2f}s "
            f"voxelize={stats['voxel_time']:.2f}s "
            f"build={stats['build_time']:.2f}s "
            f"partition={stats['partition_time']:.2f}s "
            f"elapsed={elapsed:.2f}s"
        )


if __name__ == "__main__":
    main()
