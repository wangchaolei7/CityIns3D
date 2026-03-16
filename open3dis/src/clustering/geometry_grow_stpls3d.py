import numpy as np
import torch
import torch_scatter
import torch.nn.functional as F
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from collections import defaultdict
from collections import deque
from typing import List, Optional, Sequence, Tuple
import time
import os
import pickle
import hdbscan
from hdbscan import HDBSCAN as CpuHDBSCAN  # type: ignore


_HAS_GPU_HDBSCAN = False
CuHDBSCAN = None

# Global resolution parameter
total_dbscan_time = 0.0
total_hdbscan_time = 0.0

def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor (0-1 or 0-255 range) to Lab."""
    if rgb is None:
        return None
    if rgb.numel() == 0:
        return rgb
    if rgb.dtype not in (torch.float32, torch.float64):
        rgb = rgb.float()
    scale = 255.0 if rgb.max() > 1.001 else 1.0
    rgb = rgb.clamp(0.0, 255.0) / scale

    mask = rgb > 0.04045
    rgb_linear = torch.where(mask, torch.pow((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)

    xyz_transform = torch.tensor(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        dtype=rgb_linear.dtype,
        device=rgb_linear.device,
    )
    xyz = torch.matmul(rgb_linear, xyz_transform.transpose(0, 1))

    white = torch.tensor([0.95047, 1.0, 1.08883], dtype=xyz.dtype, device=xyz.device)
    xyz_scaled = xyz / white

    eps = 1e-6
    mask = xyz_scaled > 0.008856
    f = torch.where(mask, torch.pow(xyz_scaled.clamp_min(eps), 1.0 / 3.0), 7.787 * xyz_scaled + 16.0 / 116.0)

    L = (116.0 * f[:, 1] - 16.0).unsqueeze(-1)
    a = (500.0 * (f[:, 0] - f[:, 1])).unsqueeze(-1)
    b = (200.0 * (f[:, 1] - f[:, 2])).unsqueeze(-1)
    return torch.cat([L, a, b], dim=-1)


def build_spp_adjacency_point_knn(
    scene_points: torch.Tensor,
    spp: torch.Tensor,
    n_spp: int,
    k: int = 8,
    max_neighbor_dist: Optional[float] = None,
    chunk_size: int = 50000,
) -> List[Tuple[int, ...]]:
    """Approximate local superpoint adjacency with point-level KNN contacts."""
    coords = scene_points.detach().cpu().numpy()
    spp_np = spp.detach().cpu().numpy().astype(np.int64, copy=False)
    if coords.shape[0] == 0:
        return [tuple() for _ in range(n_spp)]

    k = max(1, min(int(k), coords.shape[0] - 1 if coords.shape[0] > 1 else 1))
    tree = KDTree(coords)
    adjacency = [set() for _ in range(n_spp)]

    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])
        dists, neighbors = tree.query(coords[start:end], k=k + 1)
        src_indices = np.arange(start, end, dtype=np.int64)

        for row, src_idx in enumerate(src_indices):
            src_spp = int(spp_np[src_idx])
            for dist, dst_idx in zip(dists[row, 1:], neighbors[row, 1:]):
                if max_neighbor_dist is not None and dist > max_neighbor_dist:
                    continue
                dst_spp = int(spp_np[dst_idx])
                if src_spp == dst_spp:
                    continue
                adjacency[src_spp].add(dst_spp)
                adjacency[dst_spp].add(src_spp)

    return [tuple(sorted(neighbors)) for neighbors in adjacency]


def build_spp_members(spp: torch.Tensor, n_spp: int) -> List[torch.Tensor]:
    """Precompute point indices for each superpoint."""
    order = torch.argsort(spp)
    sorted_spp = spp[order]
    counts = torch.bincount(sorted_spp, minlength=n_spp)
    offsets = torch.cumsum(counts, dim=0)

    members = []
    start = 0
    for end in offsets.tolist():
        members.append(order[start:end])
        start = end
    return members


def aggregate_spp_features(
    point_features: torch.Tensor,
    spp: torch.Tensor,
    scene_points: torch.Tensor,
    n_spp: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate point-wise Utonia features and coordinates to superpoints."""
    spp_counts = torch.bincount(spp, minlength=n_spp).to(torch.float32)
    spp_feat = torch_scatter.scatter(
        point_features.to(torch.float32),
        spp,
        dim=0,
        dim_size=n_spp,
        reduce="mean",
    )
    spp_feat = F.normalize(spp_feat, dim=1, p=2)
    spp_centroid = torch_scatter.scatter(
        scene_points.to(torch.float32),
        spp,
        dim=0,
        dim_size=n_spp,
        reduce="mean",
    )
    return spp_feat, spp_centroid, spp_counts


def split_projected_clusters(
    highlight_points: torch.Tensor,
    scene_points: torch.Tensor,
    *,
    method: str = "dbscan",
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 50,
    hdbscan_min_cluster_size: int = 50,
    hdbscan_min_samples: int = 20,
    cluster_min_points: int = 50,
) -> List[torch.Tensor]:
    """Split lifted mask points into disconnected 3D clusters."""
    if highlight_points.numel() == 0:
        return []

    coords = scene_points[highlight_points].detach().cpu().numpy()
    if coords.shape[0] < max(2, cluster_min_points):
        return [highlight_points] if highlight_points.numel() > 0 else []

    method = method.lower()
    if method == "hdbscan":
        clusterer = CpuHDBSCAN(
            min_cluster_size=max(2, int(hdbscan_min_cluster_size)),
            min_samples=max(1, int(hdbscan_min_samples)),
        )
        labels_np = clusterer.fit_predict(coords)
    else:
        labels_np = DBSCAN(
            eps=float(dbscan_eps),
            min_samples=max(1, int(dbscan_min_samples)),
            algorithm="ball_tree",
        ).fit(coords).labels_

    labels = torch.from_numpy(labels_np).to(highlight_points.device)
    cluster_indices = []
    for label in labels.unique().tolist():
        if label == -1:
            continue
        indices = highlight_points[labels == label]
        if indices.numel() >= cluster_min_points:
            cluster_indices.append(indices)

    if cluster_indices:
        return cluster_indices
    return [highlight_points]


def grow_split_clusters_with_utonia(
    split_clusters: Sequence[torch.Tensor],
    *,
    n_points: int,
    spp: torch.Tensor,
    spp_members: Sequence[torch.Tensor],
    spp_neighbors: Sequence[Sequence[int]],
    spp_features: torch.Tensor,
    spp_centroids: torch.Tensor,
    spp_counts: torch.Tensor,
    grow_feature_threshold: float = 0.6,
    grow_min_seed_overlap: float = 0.5,
    grow_use_geometry: bool = False,
    grow_centroid_dist_threshold: float = 1.5,
    grow_update_region_stats: bool = False,
) -> List[torch.Tensor]:
    """Grow split clusters only along local superpoint adjacency using Utonia features."""
    results = []
    device = spp.device

    for cluster_indices in split_clusters:
        if cluster_indices.numel() == 0:
            continue

        group_mask = torch.zeros(n_points, dtype=torch.int8, device=device)
        group_mask[cluster_indices] = 1

        cluster_spp = spp[cluster_indices]
        unique_spp, counts = torch.unique(cluster_spp, return_counts=True)
        if unique_spp.numel() == 0:
            continue

        overlap = counts.to(torch.float32) / spp_counts[unique_spp].clamp_min(1.0)
        seed_mask = overlap >= grow_min_seed_overlap
        if torch.any(seed_mask):
            seed_spp = unique_spp[seed_mask]
        else:
            seed_spp = unique_spp[torch.argmax(overlap)].view(1)

        seed_spp_list = [int(x) for x in seed_spp.detach().cpu().tolist()]
        visited = set(seed_spp_list)
        queue = deque(seed_spp_list)

        for spp_idx in seed_spp_list:
            member_points = spp_members[spp_idx]
            if member_points.numel() > 0:
                group_mask[member_points] = 1

        region_spp = set(seed_spp_list)

        def refresh_region_stats():
            indices = torch.tensor(sorted(region_spp), dtype=torch.long, device=device)
            region_feat = F.normalize(spp_features[indices].mean(dim=0, keepdim=True), dim=1)[0]
            region_centroid = spp_centroids[indices].mean(dim=0)
            return region_feat, region_centroid

        region_feat, region_centroid = refresh_region_stats()

        while queue:
            current = queue.popleft()
            for neighbor in spp_neighbors[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                feat_sim = torch.dot(region_feat, spp_features[neighbor]).item()
                if feat_sim < grow_feature_threshold:
                    continue

                if grow_use_geometry:
                    centroid_dist = torch.norm(region_centroid - spp_centroids[neighbor]).item()
                    if centroid_dist > grow_centroid_dist_threshold:
                        continue

                member_points = spp_members[neighbor]
                if member_points.numel() == 0:
                    continue

                group_mask[member_points] = 1
                region_spp.add(neighbor)
                queue.append(neighbor)
                if grow_update_region_stats:
                    region_feat, region_centroid = refresh_region_stats()

        results.append(group_mask)

    return results


def compute_proposal_features_from_spp(
    proposals: torch.Tensor,
    spp: torch.Tensor,
    n_spp: int,
    spp_features: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Pool proposal masks to superpoints, then aggregate normalized Utonia features."""
    if proposals.numel() == 0:
        return torch.zeros((0, spp_features.shape[1]), device=spp_features.device, dtype=spp_features.dtype)

    spp = spp.to(proposals.device)
    results = []
    for start in range(0, proposals.shape[0], chunk_size):
        end = min(start + chunk_size, proposals.shape[0])
        chunk = proposals[start:end].to(torch.float32)
        spp_expanded = spp.unsqueeze(0).expand(chunk.shape[0], -1)
        spp_weights = torch_scatter.scatter(
            chunk,
            spp_expanded,
            dim=1,
            dim_size=n_spp,
            reduce="mean",
        )
        denom = spp_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = spp_weights @ spp_features
        pooled = F.normalize(pooled / denom, dim=1, p=2)
        results.append(pooled)
    return torch.cat(results, dim=0)

### dbscan 加速
def vccs_grow_spp(highlight_points, scene_points, spp, target_spp, dc_feature_spp, dc_feature):
    global total_dbscan_time

    device = highlight_points.device
    original_indices = highlight_points.clone()
    highlight_points = scene_points[highlight_points]

    if len(highlight_points) == 0:
        return []

    dbscan_start = time.time()
    # db = DBSCAN(eps=0.1, min_samples=10).fit(highlight_points.cpu().numpy())
    db = DBSCAN(eps=0.5, min_samples=15, algorithm='ball_tree').fit(highlight_points.cpu().numpy())
    labels = db.labels_
    current_time = time.time() - dbscan_start
    total_dbscan_time += current_time
    # print(f"[DBSCAN] 本次耗时: {current_time:.4f}s | 累计总耗时: {total_dbscan_time:.4f}s")

    cluster_permask = []
    for label in set(labels):
        if label == -1:
            continue
        cluster_indices = original_indices[labels == label]
        if len(cluster_indices) < 150: ### 10
            continue

        # 向量化计算特征相似度
        cluster_dc_feature = dc_feature[cluster_indices].mean(dim=0).unsqueeze(0)
        spp_features = dc_feature_spp[target_spp]
        similarities = F.cosine_similarity(cluster_dc_feature, spp_features, dim=1)
        valid_spp = target_spp[similarities > 0.6]

        group_tmp = torch.zeros(len(scene_points), dtype=torch.int8, device=device)
        group_tmp[cluster_indices] = 1

        for spp_idx in valid_spp:
            spp_point_indices = torch.where(spp == spp_idx)[0]
            overlap_ratio = group_tmp[spp_point_indices].sum().float() / len(spp_point_indices)
            if overlap_ratio > 0.1:
                group_tmp[spp_point_indices] = 1

        cluster_permask.append(group_tmp)

    return cluster_permask


def vccs_growspp_dbscan(
    highlight_points,
    scene_points,
    spp,
    target_spp,
    *,
    edge_index=None,
    node_features=None,
    rgb_mean=None,
    spp_counts=None,
    eps=0.5,
    min_samples=50,
    cluster_min_points=50,
    overlap_threshold=0.5, # 重叠阈值,大于此阈值的点被视为种子点
    normal_similarity_threshold=0.7,
    color_similarity_threshold=0.7,
    geometric_similarity_threshold=0.6, # 相似度阈值
    update_seed_stats=False,
    tau_normal=0.4,
    tau_radiometric=0.6,
    tau_geometric=1, # 高斯核函数权重
    weight_normal=1.0,
    weight_radiometric=1.0,
    weight_geometric=1.0, # 加权几何平均权重
):
    """使用DBSCAN聚类mask反投影的点，并基于超点图扩展聚类区域。

    Args:
        update_seed_stats: 为False时，不在吸收邻居后刷新平均法线与颜色。
    """
    global total_dbscan_time

    device = highlight_points.device
    n_points = scene_points.shape[0]
    if highlight_points.numel() == 0:
        return []

    coords = scene_points[highlight_points].detach().cpu().numpy()
    dbscan_start = time.time()
    labels = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree').fit(coords).labels_
    total_dbscan_time += time.time() - dbscan_start
    labels = torch.from_numpy(labels).to(device)

    unique_labels = labels.unique()
    unique_labels = unique_labels[unique_labels != -1]
    if unique_labels.numel() == 0:
        return []

    n_spp = int(spp.max().item() + 1)
    if spp_counts is not None:
        spp_counts_tensor = spp_counts.to(device=device, dtype=torch.float32)
    else:
        spp_counts_tensor = torch.zeros(n_spp, dtype=torch.float32, device=device)
        ones = torch.ones_like(spp, dtype=torch.float32)
        spp_counts_tensor.index_add_(0, spp, ones)

    target_spp_set = set()
    if target_spp is not None and target_spp.numel() > 0:
        target_spp_set.update(int(x) for x in target_spp.detach().cpu().tolist())

    normals = None
    if node_features is not None and "normal" in node_features:
        normals = F.normalize(node_features["normal"], dim=1)

    rgb_lab_tensor = _rgb_to_lab(rgb_mean.float()) if rgb_mean is not None else None

    geom_keys = [
        key
        for key in ("linearity", "planarity", "scattering", "verticality", "curvature")
        if node_features is not None and key in node_features
    ]
    geom_feature = None
    if geom_keys:
        geom_values = []
        for key in geom_keys:
            val = node_features[key].float()
            if val.dim() > 1 and val.shape[-1] == 1:
                val = val.squeeze(-1)
            geom_values.append(val.view(-1))
        geom_stack = torch.stack(geom_values, dim=-1)
        geom_mean = geom_stack.mean(dim=0)
        geom_std = geom_stack.std(dim=0, unbiased=False).clamp_min(1e-6)
        geom_feature = (geom_stack - geom_mean) / geom_std

    adjacency = defaultdict(set)
    if edge_index is not None:
        edge_np = edge_index.detach().cpu().numpy()
        for src, dst in edge_np.T:
            adjacency[int(src)].add(int(dst))
            adjacency[int(dst)].add(int(src))

    spp_indices_cache = {}

    def get_spp_indices(idx: int) -> torch.Tensor:
        cached = spp_indices_cache.get(idx)
        if cached is None:
            cached = torch.nonzero(spp == idx, as_tuple=True)[0]
            spp_indices_cache[idx] = cached
        return cached

    cluster_results = []
    for label in unique_labels.tolist():
        cluster_mask = labels == label
        cluster_indices = highlight_points[cluster_mask]
        if cluster_indices.numel() < cluster_min_points:
            continue

        seed_mask = torch.zeros(n_points, dtype=torch.int8, device=device)
        seed_mask[cluster_indices] = 1
        group_mask = seed_mask.clone()

        cluster_spp = spp[cluster_indices]
        cluster_unique_spp, cluster_counts = torch.unique(cluster_spp, return_counts=True)
        if cluster_unique_spp.numel() == 0:
            cluster_results.append(group_mask)
            continue

        total_counts = spp_counts_tensor[cluster_unique_spp].clamp_min(1.0)
        overlap = cluster_counts.float() / total_counts
        source_mask = overlap >= overlap_threshold
        if torch.any(source_mask):
            source_spp_tensor = cluster_unique_spp[source_mask]
        else:
            max_idx = torch.argmax(overlap)
            source_spp_tensor = cluster_unique_spp[max_idx].view(1)

        source_spp_set = set(int(x) for x in source_spp_tensor.detach().cpu().tolist())
        if target_spp_set:
            filtered = {idx for idx in source_spp_set if idx in target_spp_set}
            if filtered:
                source_spp_set = filtered

        for spp_idx in source_spp_set:
            group_mask[get_spp_indices(spp_idx)] = 1

        visited = set(source_spp_set)
        queue = list(source_spp_set)

        def refresh_source_stats():
            if not source_spp_set:
                return None, None, None
            indices = torch.tensor(sorted(source_spp_set), dtype=torch.long, device=device)
            normal_mean = None
            color_mean = None
            geom_stats = None
            if normals is not None:
                avg = normals[indices].mean(dim=0, keepdim=True)
                normal_mean = F.normalize(avg, dim=1)[0]
            if rgb_lab_tensor is not None:
                color_mean = rgb_lab_tensor[indices].mean(dim=0)
            if geom_feature is not None:
                geom_subset = geom_feature[indices]
                geom_mean_val = geom_subset.mean(dim=0)
                if geom_subset.shape[0] > 1:
                    geom_var_val = geom_subset.var(dim=0, unbiased=False)
                else:
                    geom_var_val = torch.ones_like(geom_mean_val)
                geom_stats = (geom_mean_val, geom_var_val.clamp_min(1e-6))
            return normal_mean, color_mean, geom_stats

        normal_mean, color_mean, geom_stats = refresh_source_stats()
        limit_to_target = len(target_spp_set) > 0

        while queue:
            current = queue.pop(0)
            for neighbor in adjacency.get(current, []):
                if neighbor in visited:
                    continue
                if limit_to_target and neighbor not in target_spp_set:
                    continue
                neighbor_points = get_spp_indices(neighbor)
                if neighbor_points.numel() == 0:
                    visited.add(neighbor)
                    continue

                sims = []
                weights = []

                thresholds = []

                if normal_mean is not None and normals is not None:
                    neighbor_normal = normals[neighbor]
                    normal_cos = torch.clamp(torch.dot(neighbor_normal, normal_mean), -1.0, 1.0)
                    d_normal = 1.0 - torch.abs(normal_cos)
                    s_normal = torch.exp(- (d_normal ** 2) / (tau_normal ** 2))
                    sims.append(s_normal)
                    weights.append(weight_normal)
                    thresholds.append(normal_similarity_threshold)

                if color_mean is not None and rgb_lab_tensor is not None:
                    neighbor_color = rgb_lab_tensor[neighbor]
                    d_color = torch.norm(color_mean - neighbor_color) / 50.0
                    s_color = torch.exp(- (d_color ** 2) / (tau_radiometric ** 2))
                    sims.append(s_color)
                    weights.append(weight_radiometric)
                    thresholds.append(color_similarity_threshold)

                if geom_stats is not None and geom_feature is not None:
                    neighbor_geom = geom_feature[neighbor]
                    geom_mean_val, geom_var_val = geom_stats
                    diff_geom = geom_mean_val - neighbor_geom
                    d_geom = torch.sqrt(torch.sum((diff_geom ** 2) / geom_var_val))
                    d_geom = d_geom / 2 # 抑制过大
                    s_geom = torch.exp(- (d_geom ** 2) / (tau_geometric ** 2))
                    sims.append(s_geom)
                    weights.append(weight_geometric)
                    thresholds.append(geometric_similarity_threshold)
                # print("sims", sims)
                if sims:
                    total_weight = sum(weights) if sum(weights) > 0 else len(weights)
                    log_score = torch.zeros(1, device=device)
                    for s_val, w_val in zip(sims, weights):
                        log_score += (w_val / total_weight) * torch.log(s_val.clamp(min=1e-6))
                    score_value = torch.exp(log_score).item()
                else:
                    score_value = 1.0
                # print("score_value", score_value)
                score_threshold = min(thresholds) if thresholds else 0.0
                if score_value < score_threshold:
                    visited.add(neighbor)
                    continue

                group_mask[neighbor_points] = 1
                source_spp_set.add(neighbor)
                visited.add(neighbor)
                queue.append(neighbor)
                if update_seed_stats:
                    normal_mean, color_mean, geom_stats = refresh_source_stats()

        cluster_results.append(group_mask)

    return cluster_results


def vccs_growspp_hdbscan(
    highlight_points,
    scene_points,
    spp,
    target_spp,
    *,
    edge_index=None,
    node_features=None,
    rgb_mean=None,
    spp_counts=None,
    min_cluster_size=5, # hdbnscan 参数
    min_samples=5,
    cluster_min_points=50,
    overlap_threshold=0.3,
    normal_similarity_threshold=0.7,
    color_similarity_threshold=0.7,
    geometric_similarity_threshold=0.6,
    update_seed_stats=False,
    tau_normal=0.4,
    tau_radiometric=0.6,
    tau_geometric=1,
    weight_normal=1.0,
    weight_radiometric=1.0,
    weight_geometric=1.0,
):
    """基于HDBSCAN（优先使用GPU实现）进行聚类后执行同样的超点扩展。"""
    global total_hdbscan_time

    device = highlight_points.device
    n_points = scene_points.shape[0]
    if highlight_points.numel() == 0:
        return []

    coords_np = scene_points[highlight_points].detach().cpu().numpy()

    start = time.time()
    labels_np = None
    if CpuHDBSCAN is not None:
        clusterer = CpuHDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels_np = clusterer.fit_predict(coords_np)
    else:
        raise RuntimeError("HDBSCAN模块未正确安装，无法调用vccs_growspp_hdbscan。")

    total_hdbscan_time += time.time() - start
    labels = torch.from_numpy(labels_np).to(device)

    unique_labels = labels.unique()
    unique_labels = unique_labels[unique_labels != -1]
    if unique_labels.numel() == 0:
        return []

    if spp_counts is not None:
        spp_counts_tensor = spp_counts.to(device=device, dtype=torch.float32)
    else:
        n_spp = int(spp.max().item() + 1)
        spp_counts_tensor = torch.zeros(n_spp, dtype=torch.float32, device=device)
        ones = torch.ones_like(spp, dtype=torch.float32)
        spp_counts_tensor.index_add_(0, spp, ones)

    target_spp_set = set()
    if target_spp is not None and target_spp.numel() > 0:
        target_spp_set.update(int(x) for x in target_spp.detach().cpu().tolist())

    normals = None
    if node_features is not None and "normal" in node_features:
        normals = F.normalize(node_features["normal"], dim=1)

    rgb_lab_tensor = _rgb_to_lab(rgb_mean.float()) if rgb_mean is not None else None

    geom_keys = [
        key
        for key in ("linearity", "planarity", "scattering", "verticality", "curvature")
        if node_features is not None and key in node_features
    ]
    geom_feature = None
    if geom_keys:
        geom_values = []
        for key in geom_keys:
            val = node_features[key].float()
            if val.dim() > 1 and val.shape[-1] == 1:
                val = val.squeeze(-1)
            geom_values.append(val.view(-1))
        geom_stack = torch.stack(geom_values, dim=-1)
        geom_mean = geom_stack.mean(dim=0)
        geom_std = geom_stack.std(dim=0, unbiased=False).clamp_min(1e-6)
        geom_feature = (geom_stack - geom_mean) / geom_std

    adjacency = defaultdict(set)
    if edge_index is not None:
        edge_np = edge_index.detach().cpu().numpy()
        for src, dst in edge_np.T:
            adjacency[int(src)].add(int(dst))
            adjacency[int(dst)].add(int(src))

    spp_indices_cache = {}

    def get_spp_indices(idx: int) -> torch.Tensor:
        cached = spp_indices_cache.get(idx)
        if cached is None:
            cached = torch.nonzero(spp == idx, as_tuple=True)[0]
            spp_indices_cache[idx] = cached
        return cached

    cluster_results = []
    for label in unique_labels.tolist():
        cluster_mask = labels == label
        cluster_indices = highlight_points[cluster_mask]
        if cluster_indices.numel() < cluster_min_points:
            continue

        seed_mask = torch.zeros(n_points, dtype=torch.int8, device=device)
        seed_mask[cluster_indices] = 1
        group_mask = seed_mask.clone()

        cluster_spp = spp[cluster_indices]
        cluster_unique_spp, cluster_counts = torch.unique(cluster_spp, return_counts=True)
        if cluster_unique_spp.numel() == 0:
            cluster_results.append(group_mask)
            continue

        total_counts = spp_counts_tensor[cluster_unique_spp].clamp_min(1.0)
        overlap = cluster_counts.float() / total_counts
        source_mask = overlap >= overlap_threshold
        if torch.any(source_mask):
            source_spp_tensor = cluster_unique_spp[source_mask]
        else:
            max_idx = torch.argmax(overlap)
            source_spp_tensor = cluster_unique_spp[max_idx].view(1)

        source_spp_set = set(int(x) for x in source_spp_tensor.detach().cpu().tolist())
        if target_spp_set:
            filtered = {idx for idx in source_spp_set if idx in target_spp_set}
            if filtered:
                source_spp_set = filtered

        for spp_idx in source_spp_set:
            group_mask[get_spp_indices(spp_idx)] = 1

        visited = set(source_spp_set)
        queue = list(source_spp_set)

        def refresh_source_stats():
            if not source_spp_set:
                return None, None, None
            indices = torch.tensor(sorted(source_spp_set), dtype=torch.long, device=device)
            normal_mean = None
            color_mean = None
            geom_stats = None
            if normals is not None:
                avg = normals[indices].mean(dim=0, keepdim=True)
                normal_mean = F.normalize(avg, dim=1)[0]
            if rgb_lab_tensor is not None:
                color_mean = rgb_lab_tensor[indices].mean(dim=0)
            if geom_feature is not None:
                geom_subset = geom_feature[indices]
                geom_mean_val = geom_subset.mean(dim=0)
                if geom_subset.shape[0] > 1:
                    geom_var_val = geom_subset.var(dim=0, unbiased=False)
                else:
                    geom_var_val = torch.ones_like(geom_mean_val)
                geom_stats = (geom_mean_val, geom_var_val.clamp_min(1e-6))
            return normal_mean, color_mean, geom_stats

        normal_mean, color_mean, geom_stats = refresh_source_stats()
        limit_to_target = len(target_spp_set) > 0

        while queue:
            current = queue.pop(0)
            for neighbor in adjacency.get(current, []):
                if neighbor in visited:
                    continue
                if limit_to_target and neighbor not in target_spp_set:
                    continue
                neighbor_points = get_spp_indices(neighbor)
                if neighbor_points.numel() == 0:
                    visited.add(neighbor)
                    continue

                sims = []
                weights = []
                thresholds = []

                if normal_mean is not None and normals is not None:
                    neighbor_normal = normals[neighbor]
                    normal_cos = torch.clamp(torch.dot(neighbor_normal, normal_mean), -1.0, 1.0)
                    d_normal = 1.0 - torch.abs(normal_cos)
                    s_normal = torch.exp(- (d_normal ** 2) / (tau_normal ** 2))
                    sims.append(s_normal)
                    weights.append(weight_normal)
                    thresholds.append(normal_similarity_threshold)

                if color_mean is not None and rgb_lab_tensor is not None:
                    neighbor_color = rgb_lab_tensor[neighbor]
                    d_color = torch.norm(color_mean - neighbor_color) / 50.0
                    s_color = torch.exp(- (d_color ** 2) / (tau_radiometric ** 2))
                    sims.append(s_color)
                    weights.append(weight_radiometric)
                    thresholds.append(color_similarity_threshold)

                if geom_stats is not None and geom_feature is not None:
                    neighbor_geom = geom_feature[neighbor]
                    geom_mean_val, geom_var_val = geom_stats
                    diff_geom = geom_mean_val - neighbor_geom
                    d_geom = torch.sqrt(torch.sum((diff_geom ** 2) / geom_var_val))
                    s_geom = torch.exp(- (d_geom ** 2) / (tau_geometric ** 2))
                    sims.append(s_geom)
                    weights.append(weight_geometric)
                    thresholds.append(geometric_similarity_threshold)

                if sims:
                    total_weight = sum(weights) if sum(weights) > 0 else len(weights)
                    log_score = torch.zeros(1, device=device)
                    for s_val, w_val in zip(sims, weights):
                        log_score += (w_val / total_weight) * torch.log(s_val.clamp(min=1e-6))
                    score_value = torch.exp(log_score).item()
                else:
                    score_value = 1.0

                score_threshold = min(thresholds) if thresholds else 0.0
                if score_value < score_threshold:
                    visited.add(neighbor)
                    continue

                group_mask[neighbor_points] = 1
                source_spp_set.add(neighbor)
                visited.add(neighbor)
                queue.append(neighbor)
                if update_seed_stats:
                    normal_mean, color_mean, geom_stats = refresh_source_stats()

        cluster_results.append(group_mask)

    return cluster_results
