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
    parser.add_argument("--scene-list", type=str, default=None, help="Override scene list path.")
    parser.add_argument(
        "--original-root",
        type=str,
        default=None,
        help="Optional override for cfg.data.original_ply. If scene-list is omitted, scenes can be discovered here.",
    )
    parser.add_argument("--worker-id", type=int, default=0, help="Worker shard index.")
    parser.add_argument("--num-workers", type=int, default=1, help="Total worker shards.")
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


def discover_scene_ids(scene_root: str):
    scene_ids = []
    if scene_root is None or not os.path.isdir(scene_root):
        return scene_ids
    for name in sorted(os.listdir(scene_root)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in {".txt", ".ply"}:
            continue
        scene_ids.append(stem)
    return scene_ids


def load_scene_ids(
    cfg,
    scene_override=None,
    scene_list_path=None,
    scene_root_override=None,
    worker_id=0,
    num_workers=1,
):
    if scene_override is not None:
        return [scene_override]

    if num_workers <= 0:
        raise ValueError(f"num-workers must be >= 1, got {num_workers}")

    if scene_list_path is not None:
        with open(scene_list_path, "r", encoding="utf-8") as handle:
            scene_ids = sorted(line.strip() for line in handle if line.strip())
        source_desc = scene_list_path
    else:
        scene_ids = discover_scene_ids(scene_root_override)
        source_desc = scene_root_override
        if not scene_ids:
            split_path = getattr(cfg.data, "split_path", None)
            if split_path and os.path.exists(split_path):
                with open(split_path, "r", encoding="utf-8") as handle:
                    scene_ids = sorted(line.strip() for line in handle if line.strip())
                source_desc = split_path
            else:
                scene_ids = discover_scene_ids(cfg.data.original_ply)
                source_desc = cfg.data.original_ply

    if not scene_ids:
        raise FileNotFoundError(
            "Cannot resolve scene ids from --scene-list, split_path, or original scene root."
        )

    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError(f"worker-id must be in [0, {num_workers}), got {worker_id}")

    scene_ids = scene_ids[worker_id::num_workers]
    print(
        f"[TGC-SP] scene_source={source_desc} "
        f"worker={worker_id}/{num_workers} num_scenes={len(scene_ids)}"
    )
    return scene_ids


def main():
    args = get_parser().parse_args()
    cfg = load_config(args.config)
    if args.original_root is not None:
        cfg.data.original_ply = args.original_root
    generator = TgcSuperpointGenerator(cfg)

    scene_ids = load_scene_ids(
        cfg,
        scene_override=args.scene,
        scene_list_path=args.scene_list,
        scene_root_override=args.original_root,
        worker_id=args.worker_id,
        num_workers=args.num_workers,
    )
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
            f"source={stats['source_format']} "
            f"voxel={stats['voxel_size']:.3f} "
            f"conn={stats['graph_connectivity']} "
            f"reg={stats['reg']:.2f} "
            f"min_size={stats['min_size']} "
            f"k={stats['k']} "
            f"w_adj={stats['w_adjacency']:.2f} "
            f"used_utonia_feature={stats['used_utonia_feature']} "
            f"load={stats['load_time']:.2f}s "
            f"voxelize={stats['voxel_time']:.2f}s "
            f"build={stats['build_time']:.2f}s "
            f"partition={stats['partition_time']:.2f}s "
            f"elapsed={elapsed:.2f}s"
        )


if __name__ == "__main__":
    main()
