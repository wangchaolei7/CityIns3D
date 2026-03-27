from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from open3dis.src.clustering.geometry_grow_stpls3d import split_projected_clusters


@dataclass
class PartGroup:
    member_parts: List[int]
    support_raw: torch.Tensor
    feature_raw: torch.Tensor
    conf_sum: float

    @property
    def support(self) -> torch.Tensor:
        return F.normalize(self.support_raw.unsqueeze(0), dim=1, p=2, eps=1e-6)[0]

    @property
    def feature(self) -> torch.Tensor:
        return F.normalize(self.feature_raw.unsqueeze(0), dim=1, p=2, eps=1e-6)[0]

    def merged_with(self, other: "PartGroup") -> "PartGroup":
        return PartGroup(
            member_parts=self.member_parts + other.member_parts,
            support_raw=self.support_raw + other.support_raw,
            feature_raw=self.feature_raw + other.feature_raw,
            conf_sum=float(self.conf_sum + other.conf_sum),
        )


@dataclass(frozen=True)
class LiftedPartProposalContext:
    cluster_cfg: object
    scene_points: torch.Tensor
    n_points: int
    spp_members: Sequence[torch.Tensor]
    spp_neighbors: Sequence[Sequence[int]]
    spp_features: torch.Tensor
    part_support: torch.Tensor
    part_conf: torch.Tensor
    part_point_indices: Optional[Sequence[torch.Tensor]] = None


_MODE_ALIASES = {
    "baseline": "spp_completion",
    "default": "spp_completion",
    "spp_completion": "spp_completion",
    "spp_complete": "spp_completion",
    "point_union": "point_union",
    "mask_union": "point_union",
    "direct_merge": "point_union",
    "raw_union": "point_union",
    "no_grow": "point_union",
}


def _lifted_part_cfg_value(cluster_cfg, name: str, default, fallback_attr: Optional[str] = None):
    lifted_part_cfg = getattr(cluster_cfg, "lifted_part", None) if cluster_cfg is not None else None
    if lifted_part_cfg is not None and hasattr(lifted_part_cfg, name):
        return getattr(lifted_part_cfg, name)

    flat_name = f"lifted_part_{name}"
    if cluster_cfg is not None and hasattr(cluster_cfg, flat_name):
        return getattr(cluster_cfg, flat_name)

    if fallback_attr and cluster_cfg is not None and hasattr(cluster_cfg, fallback_attr):
        return getattr(cluster_cfg, fallback_attr)

    return default


def normalize_lifted_part_proposal_mode(mode: str) -> str:
    mode_key = str(mode).strip().lower()
    if mode_key not in _MODE_ALIASES:
        valid_modes = ", ".join(sorted(set(_MODE_ALIASES.values())))
        raise ValueError(
            f"Unknown lifted-part proposal mode={mode!r}. Expected one of: {valid_modes}"
        )
    return _MODE_ALIASES[mode_key]


def resolve_lifted_part_proposal_mode(cluster_cfg) -> str:
    explicit_mode = _lifted_part_cfg_value(cluster_cfg, "proposal_mode", "spp_completion")
    return normalize_lifted_part_proposal_mode(explicit_mode)


def lifted_part_mode_requires_point_indices(cluster_cfg) -> bool:
    return resolve_lifted_part_proposal_mode(cluster_cfg) == "point_union"


def _connected_components_from_adj(
    neighbors: Sequence[Sequence[int]], node_ids: Sequence[int]
) -> List[List[int]]:
    allowed = set(int(x) for x in node_ids)
    visited = set()
    components = []

    for start in node_ids:
        start = int(start)
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        component = []

        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in neighbors[current]:
                neighbor = int(neighbor)
                if neighbor not in allowed or neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)

        components.append(component)

    return components


def _assign_spp_ownership(
    groups: List[PartGroup],
    *,
    spp_features: torch.Tensor,
    tie_break: bool,
    tie_margin: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not groups:
        empty_owner = torch.empty((0,), dtype=torch.long, device=spp_features.device)
        empty_scores = torch.empty((0,), dtype=torch.float32, device=spp_features.device)
        return empty_owner, empty_scores

    group_support = torch.stack([group.support_raw for group in groups], dim=0)
    support_scores, owner = torch.max(group_support, dim=0)

    if tie_break and len(groups) > 1:
        group_feature = torch.stack([group.feature for group in groups], dim=0)
        feature_scores = group_feature @ spp_features.T
        tie_mask = (group_support >= (support_scores.unsqueeze(0) - tie_margin)) & (group_support > 0)
        tie_count = tie_mask.sum(dim=0)
        if torch.any(tie_count > 1):
            tie_feature = feature_scores.masked_fill(~tie_mask, -1e6)
            tie_owner = torch.argmax(tie_feature, dim=0)
            owner = torch.where(tie_count > 1, tie_owner, owner)

    owner = torch.where(support_scores > 0, owner, torch.full_like(owner, -1))
    return owner, support_scores


def _proposal_confidence_from_parts(
    component_spp: Sequence[int],
    group: PartGroup,
    part_support: torch.Tensor,
    part_conf: torch.Tensor,
) -> float:
    if len(component_spp) == 0:
        return 0.0
    component_spp_tensor = torch.tensor(component_spp, device=part_support.device, dtype=torch.long)
    member_parts = torch.tensor(group.member_parts, device=part_support.device, dtype=torch.long)
    support_weight = part_support[member_parts][:, component_spp_tensor].sum(dim=1)
    if float(support_weight.sum().item()) <= 0:
        return float(part_conf[member_parts].mean().item())
    score = (part_conf[member_parts] * support_weight).sum() / support_weight.sum().clamp_min(1e-6)
    return float(score.item())


def _proposal_confidence_from_point_overlap(
    component_points: torch.Tensor,
    group: PartGroup,
    part_point_indices: Sequence[torch.Tensor],
    part_conf: torch.Tensor,
    *,
    n_points: int,
) -> float:
    if component_points.numel() == 0:
        return 0.0

    member_parts = torch.tensor(group.member_parts, device=part_conf.device, dtype=torch.long)
    default_score = float(part_conf[member_parts].mean().item())

    component_mask = torch.zeros((n_points,), dtype=torch.bool, device=component_points.device)
    component_mask[component_points] = True

    overlaps = []
    scores = []
    for part_idx in group.member_parts:
        points = part_point_indices[part_idx]
        if points.numel() == 0:
            continue
        overlap = component_mask[points].sum()
        if int(overlap.item()) <= 0:
            continue
        overlaps.append(overlap.to(torch.float32))
        scores.append(part_conf[part_idx])

    if not overlaps:
        return default_score

    overlap_tensor = torch.stack(overlaps)
    score_tensor = torch.stack(scores)
    weighted = (overlap_tensor * score_tensor).sum() / overlap_tensor.sum().clamp_min(1e-6)
    return float(weighted.item())


def _final_component_min_spp(cluster_cfg) -> int:
    return int(
        _lifted_part_cfg_value(
            cluster_cfg,
            "final_component_min_spp",
            1,
            fallback_attr="part_final_component_min_spp",
        )
    )


def _final_component_min_points(cluster_cfg) -> int:
    default_points = int(getattr(cluster_cfg, "valid_points", 50)) if cluster_cfg is not None else 50
    return int(
        _lifted_part_cfg_value(
            cluster_cfg,
            "final_component_min_points",
            default_points,
            fallback_attr="part_final_component_min_points",
        )
    )


def _build_spp_completion_proposals(
    groups: List[PartGroup],
    context: LiftedPartProposalContext,
) -> Tuple[List[torch.Tensor], List[float], Dict[str, float]]:
    tie_break = bool(
        _lifted_part_cfg_value(
            context.cluster_cfg,
            "reassign_use_feature_tiebreak",
            True,
            fallback_attr="part_reassign_use_feature_tiebreak",
        )
    )
    tie_margin = float(
        _lifted_part_cfg_value(
            context.cluster_cfg,
            "reassign_tie_margin",
            1e-3,
            fallback_attr="part_reassign_tie_margin",
        )
    )
    cleanup_min_spp = _final_component_min_spp(context.cluster_cfg)
    cleanup_min_points = _final_component_min_points(context.cluster_cfg)

    owner, owner_scores = _assign_spp_ownership(
        groups,
        spp_features=context.spp_features,
        tie_break=tie_break,
        tie_margin=tie_margin,
    )

    proposals = []
    confidence = []
    spp_per_proposal = []

    for group_id, group in enumerate(groups):
        group_spp = torch.nonzero(owner == group_id, as_tuple=True)[0].detach().cpu().tolist()
        if len(group_spp) < cleanup_min_spp:
            continue

        components = _connected_components_from_adj(context.spp_neighbors, group_spp)
        for component in components:
            if len(component) < cleanup_min_spp:
                continue

            point_ids = [
                context.spp_members[spp_idx]
                for spp_idx in component
                if context.spp_members[spp_idx].numel() > 0
            ]
            if not point_ids:
                continue
            component_points = torch.cat(point_ids, dim=0)
            if component_points.numel() < cleanup_min_points:
                continue

            proposal = torch.zeros((context.n_points,), dtype=torch.bool, device=context.scene_points.device)
            proposal[component_points] = True
            proposals.append(proposal)
            confidence.append(
                _proposal_confidence_from_parts(
                    component,
                    group,
                    context.part_support,
                    context.part_conf,
                )
            )
            spp_per_proposal.append(len(component))

    return proposals, confidence, {
        "ownership_nonzero": int((owner_scores > 0).sum().item()),
        "avg_spp_per_proposal": float(sum(spp_per_proposal) / len(spp_per_proposal))
        if spp_per_proposal
        else 0.0,
    }


def _build_point_union_proposals(
    groups: List[PartGroup],
    context: LiftedPartProposalContext,
) -> Tuple[List[torch.Tensor], List[float], Dict[str, float]]:
    if context.part_point_indices is None:
        raise ValueError("point_union mode requires raw lifted point indices for each part.")

    component_split = bool(_lifted_part_cfg_value(context.cluster_cfg, "component_split", True))
    component_method = str(
        _lifted_part_cfg_value(
            context.cluster_cfg,
            "component_split_method",
            "dbscan",
            fallback_attr="split_method",
        )
    )
    component_dbscan_eps = float(
        _lifted_part_cfg_value(
            context.cluster_cfg,
            "component_dbscan_eps",
            0.5,
            fallback_attr="split_dbscan_eps",
        )
    )
    component_dbscan_min_samples = int(
        _lifted_part_cfg_value(
            context.cluster_cfg,
            "component_dbscan_min_samples",
            50,
            fallback_attr="split_dbscan_min_samples",
        )
    )
    component_hdbscan_min_cluster_size = int(
        _lifted_part_cfg_value(
            context.cluster_cfg,
            "component_hdbscan_min_cluster_size",
            50,
            fallback_attr="split_hdbscan_min_cluster_size",
        )
    )
    component_hdbscan_min_samples = int(
        _lifted_part_cfg_value(
            context.cluster_cfg,
            "component_hdbscan_min_samples",
            20,
            fallback_attr="split_hdbscan_min_samples",
        )
    )
    component_min_points = int(
        _lifted_part_cfg_value(
            context.cluster_cfg,
            "component_min_points",
            _final_component_min_points(context.cluster_cfg),
        )
    )

    proposals = []
    confidence = []

    for group in groups:
        group_point_ids = [
            context.part_point_indices[part_idx]
            for part_idx in group.member_parts
            if context.part_point_indices[part_idx].numel() > 0
        ]
        if not group_point_ids:
            continue

        merged_points = torch.unique(torch.cat(group_point_ids, dim=0), sorted=False)
        if merged_points.numel() < component_min_points:
            continue

        if component_split:
            components = split_projected_clusters(
                merged_points,
                context.scene_points,
                method=component_method,
                dbscan_eps=component_dbscan_eps,
                dbscan_min_samples=component_dbscan_min_samples,
                hdbscan_min_cluster_size=component_hdbscan_min_cluster_size,
                hdbscan_min_samples=component_hdbscan_min_samples,
                cluster_min_points=component_min_points,
            )
        else:
            components = [merged_points]

        for component_points in components:
            if component_points.numel() < component_min_points:
                continue
            proposal = torch.zeros((context.n_points,), dtype=torch.bool, device=context.scene_points.device)
            proposal[component_points] = True
            proposals.append(proposal)
            confidence.append(
                _proposal_confidence_from_point_overlap(
                    component_points,
                    group,
                    context.part_point_indices,
                    context.part_conf,
                    n_points=context.n_points,
                )
            )

    return proposals, confidence, {
        "component_split": int(component_split),
        "avg_spp_per_proposal": 0.0,
    }


_BUILDERS: Dict[
    str, Callable[[List[PartGroup], LiftedPartProposalContext], Tuple[List[torch.Tensor], List[float], Dict[str, float]]]
] = {
    "spp_completion": _build_spp_completion_proposals,
    "point_union": _build_point_union_proposals,
}


def build_lifted_part_proposals(
    groups: List[PartGroup],
    context: LiftedPartProposalContext,
) -> Tuple[List[torch.Tensor], List[float], Dict[str, float]]:
    mode = resolve_lifted_part_proposal_mode(context.cluster_cfg)
    return _BUILDERS[mode](groups, context)
