import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml
from munch import Munch
from tqdm import tqdm


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from open3dis.src.clustering.clustering_lifted_part_utonia_stpls3d import (  # noqa: E402
    process_lifted_part_utonia_stpls3d,
)


def rle_encode_gpu_batch(masks: torch.Tensor):
    n_inst, length = masks.shape[:2]
    zeros_tensor = torch.zeros((n_inst, 1), dtype=torch.bool, device=masks.device)
    masks = torch.cat([zeros_tensor, masks.bool(), zeros_tensor], dim=1)

    rles = []
    for i in range(n_inst):
        mask = masks[i]
        runs = torch.nonzero(mask[1:] != mask[:-1]).view(-1) + 1
        runs[1::2] -= runs[::2]
        counts = runs.cpu().numpy()
        rles.append(dict(length=length, counts=counts))
    return rles


def get_parser():
    parser = argparse.ArgumentParser(description="Generate lifted-part 3D proposals for STPLS3D")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--scene", type=str, default=None, help="Optional single scene id")
    parser.add_argument("--scene-list", type=str, default=None, help="Optional scene list override")
    parser.add_argument("--worker-id", type=int, default=0, help="Worker shard index")
    parser.add_argument("--num-workers", type=int, default=1, help="Total number of worker shards")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output dir override")
    return parser


def resolve_output_dir(cfg, cli_output_dir=None):
    if cli_output_dir:
        return cli_output_dir
    exp_cfg = cfg.exp
    subdir = getattr(exp_cfg, "clustering_3d_output_lifted_part", "3d_ins_img16_lifted_part")
    return os.path.join(exp_cfg.save_dir, exp_cfg.exp_name, subdir)


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))

    if args.scene:
        scene_ids = [args.scene]
    else:
        scene_list_path = args.scene_list or cfg.data.split_path
        with open(scene_list_path, "r") as file:
            scene_ids = sorted([line.rstrip("\n") for line in file])
        if args.num_workers > 1:
            if args.worker_id < 0 or args.worker_id >= args.num_workers:
                raise ValueError(
                    f"worker-id must be in [0, {args.num_workers}), got {args.worker_id}"
                )
            scene_ids = scene_ids[args.worker_id::args.num_workers]
        print(
            f"[LiftedPart] scene_list={scene_list_path} "
            f"worker={args.worker_id}/{args.num_workers} num_scenes={len(scene_ids)}"
        )

    output_dir = resolve_output_dir(cfg, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for scene_id in tqdm(scene_ids):
        output_path = os.path.join(output_dir, f"{scene_id}.pth")
        if os.path.exists(output_path) and not args.overwrite:
            print(f"[LiftedPart] skip existing scene={scene_id} -> {output_path}")
            continue

        start = time.perf_counter()
        proposals3d, confidence, stats = process_lifted_part_utonia_stpls3d(scene_id, cfg)
        elapsed = time.perf_counter() - start

        if proposals3d is None:
            print(f"[LiftedPart] scene={scene_id} produced no proposals.")
            continue

        cluster_dict = {
            "ins": rle_encode_gpu_batch(proposals3d),
            "conf": confidence.detach().cpu(),
            "meta": stats,
        }
        torch.save(cluster_dict, output_path)

        print(
            f"[LiftedPart] saved {scene_id} -> {output_path} "
            f"raw_parts={stats.get('raw_parts', 0)} "
            f"valid_parts={stats.get('valid_parts', 0)} "
            f"merged_groups={stats.get('merged_groups', 0)} "
            f"final_proposals={stats.get('final_proposals', 0)} "
            f"avg_spp_per_proposal={stats.get('avg_spp_per_proposal', 0.0):.2f} "
            f"collect={stats.get('collect_time', 0.0):.2f}s "
            f"merge={stats.get('merge_time', 0.0):.2f}s "
            f"elapsed={elapsed:.2f}s"
        )
