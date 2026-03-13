import argparse
import os
import time

import torch
import yaml
from munch import Munch

from util3d.utonia_point_features import UtoniaPointFeatureExtractor


def get_parser():
    parser = argparse.ArgumentParser(description="Generate point-level Utonia features.")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml.")
    parser.add_argument("--scene", type=str, default=None, help="Only process one scene id.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing feature files.",
    )
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

    output_dir = cfg.data.point_features_path
    os.makedirs(output_dir, exist_ok=True)

    extractor = UtoniaPointFeatureExtractor(cfg)
    scene_ids = load_scene_ids(cfg, args.scene)

    for scene_id in scene_ids:
        ply_path = os.path.join(cfg.data.original_ply, f"{scene_id}.ply")
        output_path = os.path.join(output_dir, f"{scene_id}.pth")

        if not os.path.exists(ply_path):
            print(f"[Utonia] missing point cloud, skip: {ply_path}")
            continue
        if os.path.exists(output_path) and not args.overwrite:
            print(f"[Utonia] exists, skip: {output_path}")
            continue

        start_time = time.perf_counter()
        feature_pack, meta = extractor.extract_from_file(ply_path)
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
