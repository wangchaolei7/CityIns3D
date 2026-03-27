import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pycocotools.mask
import torch
import torch.nn.functional as F

from open3dis.dataset_outdoor import build_dataset
from open3dis.src.clustering.geometry_grow_stpls3d import (
    aggregate_spp_features,
    build_spp_adjacency_point_knn,
    build_spp_members,
)
from open3dis.src.clustering.lifted_part_proposal_modes import (
    LiftedPartProposalContext,
    PartGroup,
    build_lifted_part_proposals,
    lifted_part_mode_requires_point_indices,
    resolve_lifted_part_proposal_mode,
)
from open3dis.src.mapper_zbuffer import PointCloudToImageMapper


def _torch_load_local(path: str, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _resolve_scene_file(base_dirs: Sequence[Optional[str]], scene_id: str, exts: Sequence[str]) -> Optional[str]:
    for base_dir in base_dirs:
        if not base_dir:
            continue
        for ext in exts:
            candidate = os.path.join(base_dir, f"{scene_id}{ext}")
            if os.path.exists(candidate):
                return candidate
    return None


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    return F.normalize(x, dim=1, p=2, eps=1e-6)


def _compact_labels(labels: torch.Tensor) -> torch.Tensor:
    if labels.numel() == 0:
        return labels.to(torch.long)
    unique = torch.unique(labels)
    order = torch.argsort(unique)
    unique = unique[order]
    return torch.bucketize(labels, unique)


def _decode_frame_masks(encoded_masks: Sequence[dict]) -> List[np.ndarray]:
    masks = []
    for mask_rle in encoded_masks:
        decoded = pycocotools.mask.decode(mask_rle)
        if decoded.ndim == 3:
            decoded = decoded[..., 0]
        masks.append(decoded.astype(bool, copy=False))
    return masks


def _extract_frame_mask_data(frame_id, grounded_data_dict):
    if frame_id in grounded_data_dict:
        frame_data = grounded_data_dict[frame_id]
        masks = frame_data.get("masks", [])
        conf = frame_data.get("conf", None)
        return masks, conf

    if "masks" in grounded_data_dict and "frame_id" in grounded_data_dict:
        try:
            index = grounded_data_dict["frame_id"].index(frame_id)
        except ValueError:
            return None, None
        masks = grounded_data_dict["masks"][index]
        conf = None
        return masks, conf

    return None, None


def _support_similarity(
    left: torch.Tensor,
    right: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    if metric == "cosine":
        return left @ right.T

    if metric == "jsd":
        left_prob = left / left.sum(dim=1, keepdim=True).clamp_min(1e-6)
        right_prob = right / right.sum(dim=1, keepdim=True).clamp_min(1e-6)
        left_expanded = left_prob[:, None, :]
        right_expanded = right_prob[None, :, :]
        mean_prob = 0.5 * (left_expanded + right_expanded)
        kl_left = (left_expanded * (left_expanded.clamp_min(1e-6).log() - mean_prob.clamp_min(1e-6).log())).sum(dim=-1)
        kl_right = (right_expanded * (right_expanded.clamp_min(1e-6).log() - mean_prob.clamp_min(1e-6).log())).sum(dim=-1)
        jsd = 0.5 * (kl_left + kl_right)
        return 1.0 - jsd

    raise ValueError(f"Unsupported support affinity metric: {metric}")


def _build_part_groups(
    part_support: torch.Tensor,
    part_features: torch.Tensor,
    part_conf: torch.Tensor,
) -> List[PartGroup]:
    groups = []
    for idx in range(part_support.shape[0]):
        conf = float(part_conf[idx].item())
        groups.append(
            PartGroup(
                member_parts=[idx],
                support_raw=part_support[idx] * conf,
                feature_raw=part_features[idx] * conf,
                conf_sum=conf,
            )
        )
    return groups


def _merge_groups_progressively(
    groups: List[PartGroup],
    *,
    thresholds: Sequence[float],
    affinity_alpha: float,
    support_metric: str,
) -> Tuple[List[PartGroup], List[Dict[str, float]]]:
    stage_stats = []
    if len(groups) <= 1:
        return groups, stage_stats

    for stage_idx, threshold in enumerate(thresholds):
        if len(groups) <= 1:
            break

        remaining = set(range(len(groups)))
        merged_groups = []

        while remaining:
            seed_idx = max(remaining, key=lambda idx: groups[idx].conf_sum)
            remaining.remove(seed_idx)
            region = groups[seed_idx]

            while remaining:
                candidate_ids = sorted(remaining)
                bank_support = torch.stack([groups[idx].support for idx in candidate_ids], dim=0)
                bank_feature = torch.stack([groups[idx].feature for idx in candidate_ids], dim=0)

                support_sim = _support_similarity(
                    region.support.unsqueeze(0),
                    bank_support,
                    support_metric,
                )[0]
                feature_sim = torch.matmul(bank_feature, region.feature)
                combined = affinity_alpha * support_sim + (1.0 - affinity_alpha) * feature_sim

                best_value, best_local = torch.max(combined, dim=0)
                if float(best_value.item()) < threshold:
                    break

                best_idx = candidate_ids[int(best_local.item())]
                remaining.remove(best_idx)
                region = region.merged_with(groups[best_idx])

            merged_groups.append(region)

        stage_stats.append(
            {
                "stage": float(stage_idx),
                "threshold": float(threshold),
                "num_groups": float(len(merged_groups)),
            }
        )
        groups = merged_groups

    return groups, stage_stats


def _collect_lifted_parts(
    scene_id: str,
    cfg,
    *,
    loader,
    pointcloud_mapper,
    points: torch.Tensor,
    spp: torch.Tensor,
    spp_counts: torch.Tensor,
    point_features: torch.Tensor,
    groundedsam_data_dict: Dict,
    collect_point_indices: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], Dict[str, float]]:
    device = points.device
    n_spp = int(spp.max().item() + 1)

    part_min_points = int(getattr(cfg.cluster, "part_min_points", 50))
    part_min_spp = int(getattr(cfg.cluster, "part_min_spp", 1))
    part_min_support = float(getattr(cfg.cluster, "part_min_support", 1e-4))
    part_min_confidence = float(getattr(cfg.cluster, "part_min_confidence", 0.0))

    part_support_rows = []
    part_features = []
    part_conf = []
    part_point_indices = []

    raw_parts = 0
    valid_parts = 0

    for frame_idx in range(0, len(loader), cfg.data.img_interval):
        frame = loader[frame_idx]
        frame_id = frame["frame_id"]
        encoded_masks, mask_conf = _extract_frame_mask_data(frame_id, groundedsam_data_dict)
        if encoded_masks is None:
            continue

        raw_parts += len(encoded_masks)
        if len(encoded_masks) == 0:
            continue

        pose = loader.read_pose(frame["pose_path"])
        depth = loader.read_depth(frame["depth_path"])
        intrinsic = loader.read_intrinsic(frame["intrinsic_path"])

        rgb_dim_hw = cfg.data.img_dim
        mapping = torch.ones((points.shape[0], 4), dtype=torch.int32, device=device)
        mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(
            pose,
            points,
            rgb_dim_hw,
            depth=depth,
            intrinsic=intrinsic,
            id_1=scene_id,
            id_2=frame_id,
        )

        mapping_np = mapping.detach().cpu().numpy()
        visible_idx = np.nonzero(mapping_np[:, 3] == 1)[0]
        if visible_idx.size == 0:
            continue

        visible_rows = mapping_np[visible_idx, 1]
        visible_cols = mapping_np[visible_idx, 2]
        frame_masks = _decode_frame_masks(encoded_masks)

        if isinstance(mask_conf, torch.Tensor):
            frame_conf = mask_conf.detach().cpu().numpy().astype(np.float32, copy=False)
        elif mask_conf is None:
            frame_conf = np.ones((len(frame_masks),), dtype=np.float32)
        else:
            frame_conf = np.asarray(mask_conf, dtype=np.float32)

        for mask_id, mask in enumerate(frame_masks):
            conf = float(frame_conf[mask_id]) if mask_id < len(frame_conf) else 1.0
            if conf < part_min_confidence:
                continue

            selected = mask[visible_rows, visible_cols]
            if not np.any(selected):
                continue

            highlight_points_np = visible_idx[selected]
            if highlight_points_np.size < part_min_points:
                continue

            highlight_points = torch.as_tensor(highlight_points_np, device=device, dtype=torch.long)
            part_spp = spp[highlight_points]
            unique_spp, counts = torch.unique(part_spp, return_counts=True)
            if unique_spp.numel() < part_min_spp:
                continue

            coverage = counts.to(torch.float32) / spp_counts[unique_spp].clamp_min(1.0)
            keep_mask = coverage >= part_min_support
            if not torch.any(keep_mask):
                continue

            unique_spp = unique_spp[keep_mask]
            coverage = coverage[keep_mask]
            if unique_spp.numel() < part_min_spp:
                continue

            support_row = torch.zeros((n_spp,), dtype=torch.float32, device=device)
            support_row[unique_spp] = coverage
            feature = point_features[highlight_points].mean(dim=0, keepdim=True)
            feature = F.normalize(feature, dim=1, p=2, eps=1e-6)[0]

            part_support_rows.append(support_row)
            part_features.append(feature)
            part_conf.append(conf)
            if collect_point_indices:
                part_point_indices.append(highlight_points.clone())
            valid_parts += 1

    if not part_support_rows:
        empty_support = torch.zeros((0, n_spp), dtype=torch.float32, device=device)
        empty_feature = torch.zeros((0, point_features.shape[1]), dtype=torch.float32, device=device)
        empty_conf = torch.zeros((0,), dtype=torch.float32, device=device)
        return empty_support, empty_feature, empty_conf, [], {
            "raw_parts": float(raw_parts),
            "valid_parts": float(valid_parts),
        }

    support = torch.stack(part_support_rows, dim=0)
    features = torch.stack(part_features, dim=0)
    conf = torch.tensor(part_conf, dtype=torch.float32, device=device)
    return support, features, conf, part_point_indices, {
        "raw_parts": float(raw_parts),
        "valid_parts": float(valid_parts),
    }


@torch.inference_mode()
def process_lifted_part_utonia_stpls3d(scene_id: str, cfg):
    input_exp_name = getattr(cfg.exp, "input_exp_name", cfg.exp.exp_name)
    exp_path = os.path.join(cfg.exp.save_dir, input_exp_name)
    mask2d_path = os.path.join(exp_path, cfg.exp.mask2d_output, f"{scene_id}.pth")
    if not os.path.exists(mask2d_path):
        raise FileNotFoundError(f"Missing 2D mask file for {scene_id}: {mask2d_path}")

    scene_dir = os.path.join(cfg.data.datapath, scene_id)
    loader = build_dataset(root_path=scene_dir, cfg=cfg)
    device = torch.device(getattr(cfg.foundation_model, "device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    pointcloud_mapper = PointCloudToImageMapper(
        image_dim=cfg.data.rgb_img_dim,
        cut_bound=cfg.data.cut_num_pixel_boundary,
        device=str(device),
    )

    points = torch.from_numpy(loader.read_pointcloud()).to(device=device, dtype=torch.float32)
    n_points = points.shape[0]

    spp_file = _resolve_scene_file([getattr(cfg.data, "spp_path", None)], scene_id, [".pth", ".pt"])
    if spp_file is None:
        raise FileNotFoundError(f"Missing point-wise superpoint labels for {scene_id}.")
    spp = loader.read_spp(spp_file, device=str(device)).long()
    _, spp = torch.unique(spp, return_inverse=True)
    n_spp = int(spp.max().item() + 1)

    point_feature_path = os.path.join(cfg.data.point_features_path, f"{scene_id}.pth")
    if not os.path.exists(point_feature_path):
        raise FileNotFoundError(f"Missing Utonia point features for {scene_id}: {point_feature_path}")
    point_features = loader.read_feature(point_feature_path, device=str(device)).to(torch.float32)
    point_features = F.normalize(point_features, dim=1, p=2, eps=1e-6)
    if point_features.shape[0] != n_points:
        raise ValueError(
            f"Point feature count mismatch for {scene_id}: points={n_points}, features={point_features.shape[0]}"
        )

    spp_features, _, spp_counts = aggregate_spp_features(point_features, spp, points, n_spp)
    spp_members = build_spp_members(spp, n_spp)
    spp_neighbors = build_spp_adjacency_point_knn(
        points,
        spp,
        n_spp,
        k=int(getattr(cfg.cluster, "spp_graph_k", 8)),
        max_neighbor_dist=getattr(cfg.cluster, "spp_graph_max_neighbor_dist", None),
    )

    groundedsam_data_dict = _torch_load_local(mask2d_path, map_location="cpu")
    proposal_mode = resolve_lifted_part_proposal_mode(cfg.cluster)
    collect_start = time.perf_counter()
    part_support, part_features, part_conf, part_point_indices, collect_stats = _collect_lifted_parts(
        scene_id,
        cfg,
        loader=loader,
        pointcloud_mapper=pointcloud_mapper,
        points=points,
        spp=spp,
        spp_counts=spp_counts.to(device=device, dtype=torch.float32),
        point_features=point_features,
        groundedsam_data_dict=groundedsam_data_dict,
        collect_point_indices=lifted_part_mode_requires_point_indices(cfg.cluster),
    )
    collect_elapsed = time.perf_counter() - collect_start

    if part_support.shape[0] == 0:
        print(f"[LiftedPart] scene={scene_id} no valid lifted parts after filtering.")
        return None, None, {}

    groups = _build_part_groups(part_support, part_features, part_conf)
    thresholds = list(getattr(cfg.cluster, "part_affinity_thresholds", [0.75, 0.65, 0.55, 0.45]))
    support_metric = getattr(cfg.cluster, "part_support_metric", "cosine").lower()
    affinity_alpha = float(getattr(cfg.cluster, "part_affinity_alpha", 0.7))

    merge_start = time.perf_counter()
    groups, stage_stats = _merge_groups_progressively(
        groups,
        thresholds=thresholds,
        affinity_alpha=affinity_alpha,
        support_metric=support_metric,
    )
    merge_elapsed = time.perf_counter() - merge_start

    proposal_context = LiftedPartProposalContext(
        cluster_cfg=cfg.cluster,
        scene_points=points,
        n_points=n_points,
        spp_members=spp_members,
        spp_neighbors=spp_neighbors,
        spp_features=spp_features,
        part_support=part_support,
        part_conf=part_conf,
        part_point_indices=part_point_indices if part_point_indices else None,
    )
    proposals, confidence, proposal_stats = build_lifted_part_proposals(groups, proposal_context)

    if not proposals:
        print(f"[LiftedPart] scene={scene_id} no proposals remained after reassignment and cleanup.")
        return None, None, {}

    proposals3d = torch.stack(proposals, dim=0)
    confidence_t = torch.tensor(confidence, dtype=torch.float32, device=device)

    stats = {
        "raw_parts": int(collect_stats["raw_parts"]),
        "valid_parts": int(collect_stats["valid_parts"]),
        "merged_groups": int(len(groups)),
        "final_proposals": int(proposals3d.shape[0]),
        "avg_spp_per_proposal": float(proposal_stats.get("avg_spp_per_proposal", 0.0)),
        "num_points": int(n_points),
        "num_spp": int(n_spp),
        "collect_time": float(collect_elapsed),
        "merge_time": float(merge_elapsed),
        "device": str(device),
        "stage_stats": stage_stats,
        "proposal_mode": proposal_mode,
        **proposal_stats,
    }
    return proposals3d, confidence_t, stats
