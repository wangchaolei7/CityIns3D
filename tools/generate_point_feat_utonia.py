import argparse
import os
import time

import torch
import yaml
from munch import Munch

from open3dis.dataset_outdoor.stpls3d_io import resolve_scene_path
from util3d.utonia_point_features import UtoniaPointFeatureExtractor


def get_parser():
    parser = argparse.ArgumentParser(description="Generate point-level Utonia features.")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml.")
    parser.add_argument("--scene", type=str, default=None, help="Only process one scene id.")
    parser.add_argument(
        "--scene-list",
        type=str,
        default=None,
        help="Optional scene list txt. If omitted, scenes can be discovered from --original-root.",
    )
    parser.add_argument(
        "--original-root",
        type=str,
        default=None,
        help="Optional override for cfg.data.original_ply. Can point to txt or ply scene directory.",
    )
    parser.add_argument("--worker-id", type=int, default=0, help="Shard worker id.")
    parser.add_argument("--num-workers", type=int, default=1, help="Total shard workers.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing feature files.",
    )
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


def load_scene_ids(cfg, scene_override=None, scene_list_override=None, scene_root_override=None):
    if scene_override is not None:
        return [scene_override]

    if scene_list_override is not None:
        with open(scene_list_override, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    discovered = discover_scene_ids(scene_root_override)
    if discovered:
        return discovered

    split_path = getattr(cfg.data, "split_path", None)
    if split_path and os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    discovered = discover_scene_ids(cfg.data.original_ply)
    if discovered:
        return discovered

    raise FileNotFoundError("Cannot resolve scene ids from --scene-list, split_path, or original scene root.")


def main():
    args = get_parser().parse_args()
    cfg = load_config(args.config)
    if args.original_root is not None:
        cfg.data.original_ply = args.original_root

    output_dir = cfg.data.point_features_path
    os.makedirs(output_dir, exist_ok=True)

    extractor = UtoniaPointFeatureExtractor(cfg)
    scene_ids = load_scene_ids(
        cfg,
        args.scene,
        scene_list_override=args.scene_list,
        scene_root_override=args.original_root,
    )
    if args.num_workers > 1:
        scene_ids = scene_ids[args.worker_id :: args.num_workers]

    for scene_id in scene_ids:
        output_path = os.path.join(output_dir, f"{scene_id}.pth")

        try:
            scene_path = resolve_scene_path(cfg.data.original_ply, scene_id)
        except FileNotFoundError as exc:
            print(f"[Utonia] missing point cloud, skip: {exc}")
            continue
        if os.path.exists(output_path) and not args.overwrite:
            print(f"[Utonia] exists, skip: {output_path}")
            continue

        start_time = time.perf_counter()
        feature_pack, meta = extractor.extract_from_file(scene_path)
        torch.save(feature_pack["spp_feat"], output_path)
        elapsed = time.perf_counter() - start_time
        print(
            f"[Utonia] saved {scene_id} -> {output_path} "
            f"shape={tuple(feature_pack['spp_feat'].shape)} "
            f"dtype={feature_pack['spp_feat'].dtype} "
            f"num_points={meta['num_points']} feat_dim={meta['feat_dim']} "
            f"elapsed={elapsed:.2f}s"
        )

        del feature_pack
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
