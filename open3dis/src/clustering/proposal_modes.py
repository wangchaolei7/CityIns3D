from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import torch

from open3dis.src.clustering.geometry_grow_stpls3d import (
    grow_split_clusters_with_utonia,
    split_projected_clusters,
)


@dataclass(frozen=True)
class ProposalModeContext:
    n_points: int
    scene_points: torch.Tensor
    spp: torch.Tensor
    spp_members: Sequence[torch.Tensor]
    spp_neighbors: Sequence[Sequence[int]]
    spp_features: torch.Tensor
    spp_centroids: torch.Tensor
    spp_counts: torch.Tensor
    cluster_cfg: object
    update_seed_stats: bool = True


_MODE_ALIASES = {
    "baseline": "split_grow",
    "default": "split_grow",
    "split_grow": "split_grow",
    "lift_only": "lift_only",
    "no_grow": "lift_only",
    "raw_lift": "lift_only",
    "split_only": "split_only",
    "grow_only": "grow_only",
    "no_split": "grow_only",
}


def normalize_proposal_mode(mode: str) -> str:
    mode_key = str(mode).strip().lower()
    if mode_key not in _MODE_ALIASES:
        valid_modes = ", ".join(sorted(set(_MODE_ALIASES.values())))
        raise ValueError(f"Unknown proposal_mode={mode!r}. Expected one of: {valid_modes}")
    return _MODE_ALIASES[mode_key]


def resolve_proposal_mode(cluster_cfg) -> str:
    explicit_mode = getattr(cluster_cfg, "proposal_mode", None) if cluster_cfg is not None else None
    if explicit_mode is not None:
        return normalize_proposal_mode(explicit_mode)

    enable_splitting = bool(
        getattr(cluster_cfg, "enable_splitting", True) if cluster_cfg is not None else True
    )
    return "split_grow" if enable_splitting else "grow_only"


def available_proposal_modes() -> List[str]:
    return sorted(set(_MODE_ALIASES.values()))


def _clusters_to_point_masks(
    split_clusters: Sequence[torch.Tensor],
    *,
    n_points: int,
    device: torch.device,
) -> List[torch.Tensor]:
    masks = []
    for cluster_indices in split_clusters:
        if cluster_indices is None or cluster_indices.numel() == 0:
            continue
        group_mask = torch.zeros(n_points, dtype=torch.int8, device=device)
        group_mask[cluster_indices] = 1
        masks.append(group_mask)
    return masks


def _split_clusters(
    highlight_points: torch.Tensor,
    context: ProposalModeContext,
) -> List[torch.Tensor]:
    cfg = context.cluster_cfg
    return split_projected_clusters(
        highlight_points,
        context.scene_points,
        method=getattr(cfg, "split_method", "dbscan"),
        dbscan_eps=float(getattr(cfg, "split_dbscan_eps", 0.5)),
        dbscan_min_samples=int(getattr(cfg, "split_dbscan_min_samples", 50)),
        hdbscan_min_cluster_size=int(getattr(cfg, "split_hdbscan_min_cluster_size", 50)),
        hdbscan_min_samples=int(getattr(cfg, "split_hdbscan_min_samples", 20)),
        cluster_min_points=int(getattr(cfg, "split_cluster_min_points", 50)),
    )


def _grow_clusters(
    split_clusters: Sequence[torch.Tensor],
    context: ProposalModeContext,
) -> List[torch.Tensor]:
    cfg = context.cluster_cfg
    return grow_split_clusters_with_utonia(
        split_clusters,
        n_points=context.n_points,
        spp=context.spp,
        spp_members=context.spp_members,
        spp_neighbors=context.spp_neighbors,
        spp_features=context.spp_features,
        spp_centroids=context.spp_centroids,
        spp_counts=context.spp_counts,
        grow_feature_threshold=float(getattr(cfg, "grow_feature_threshold", getattr(cfg, "simi", 0.6))),
        grow_min_seed_overlap=float(getattr(cfg, "grow_min_seed_overlap", 0.5)),
        grow_use_geometry=bool(getattr(cfg, "grow_use_geometry", False)),
        grow_centroid_dist_threshold=float(getattr(cfg, "grow_centroid_dist_threshold", 1.5)),
        grow_update_region_stats=bool(context.update_seed_stats),
    )


def _build_split_grow(
    highlight_points: torch.Tensor,
    context: ProposalModeContext,
) -> List[torch.Tensor]:
    return _grow_clusters(_split_clusters(highlight_points, context), context)


def _build_lift_only(
    highlight_points: torch.Tensor,
    context: ProposalModeContext,
) -> List[torch.Tensor]:
    return _clusters_to_point_masks(
        [highlight_points],
        n_points=context.n_points,
        device=context.spp.device,
    )


def _build_split_only(
    highlight_points: torch.Tensor,
    context: ProposalModeContext,
) -> List[torch.Tensor]:
    return _clusters_to_point_masks(
        _split_clusters(highlight_points, context),
        n_points=context.n_points,
        device=context.spp.device,
    )


def _build_grow_only(
    highlight_points: torch.Tensor,
    context: ProposalModeContext,
) -> List[torch.Tensor]:
    return _grow_clusters([highlight_points], context)


_BUILDERS: Dict[str, Callable[[torch.Tensor, ProposalModeContext], List[torch.Tensor]]] = {
    "split_grow": _build_split_grow,
    "lift_only": _build_lift_only,
    "split_only": _build_split_only,
    "grow_only": _build_grow_only,
}


def build_seed_proposals_for_mask(
    highlight_points: torch.Tensor,
    context: ProposalModeContext,
) -> List[torch.Tensor]:
    mode = resolve_proposal_mode(context.cluster_cfg)
    return _BUILDERS[mode](highlight_points, context)
