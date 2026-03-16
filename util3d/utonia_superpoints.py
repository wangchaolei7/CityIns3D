import os
from collections import deque
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA


def _row_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.clip(norm, a_min=1e-6, a_max=None)
    return x / norm


def _standardize(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return labels.astype(np.int64, copy=False)
    unique = np.unique(labels)
    remapped = np.searchsorted(unique, labels)
    return remapped.astype(np.int64, copy=False)


def _torch_load_local(path: str, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _neighbor_offsets(connectivity: int) -> Tuple[Tuple[int, int, int], ...]:
    if connectivity == 6:
        return (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        )
    if connectivity == 26:
        return tuple(
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        )
    raise ValueError(f"Unsupported connectivity: {connectivity}")


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros((n,), dtype=np.int8)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1

    def labels(self) -> np.ndarray:
        labels = np.empty_like(self.parent)
        for idx in range(self.parent.shape[0]):
            labels[idx] = self.find(idx)
        return _compact_labels(labels)


class UtoniaSuperpointGenerator:
    def __init__(self, cfg):
        if not hasattr(cfg, "superpoint"):
            raise ValueError("superpoint config is required for Utonia superpoint generation.")

        self.cfg = cfg
        self.point_feature_dir = cfg.data.point_features_path
        self.original_ply_root = cfg.data.original_ply
        self.save_dir = getattr(cfg.superpoint, "save_dir", cfg.data.spp_path)
        self.method = getattr(cfg.superpoint, "method", "utonia_voxel_graph_overseg")
        self.proto_grid_size = float(getattr(cfg.superpoint, "proto_grid_size", 0.2))

        # New default: local voxel-graph over-segmentation.
        self.graph_connectivity = int(getattr(cfg.superpoint, "graph_connectivity", 6))
        self.graph_feature_threshold = float(
            getattr(cfg.superpoint, "graph_feature_threshold", 0.78)
        )
        self.graph_merge_small_components = bool(
            getattr(cfg.superpoint, "graph_merge_small_components", True)
        )
        self.graph_min_component_voxels = int(
            getattr(cfg.superpoint, "graph_min_component_voxels", 3)
        )
        self.graph_min_component_points = int(
            getattr(cfg.superpoint, "graph_min_component_points", 64)
        )
        self.graph_merge_sim_threshold = float(
            getattr(cfg.superpoint, "graph_merge_sim_threshold", 0.72)
        )
        self.graph_edge_chunk_size = int(
            getattr(cfg.superpoint, "graph_edge_chunk_size", 262144)
        )
        self.component_connectivity = int(
            getattr(cfg.superpoint, "component_connectivity", self.graph_connectivity)
        )

        # Legacy global clustering path, kept optional.
        self.cluster_method = getattr(cfg.superpoint, "cluster_method", "hdbscan")
        self.hdbscan_min_cluster_size = int(
            getattr(cfg.superpoint, "hdbscan_min_cluster_size", 20)
        )
        self.hdbscan_min_samples = int(getattr(cfg.superpoint, "hdbscan_min_samples", 10))
        self.dbscan_eps = float(getattr(cfg.superpoint, "dbscan_eps", 1.0))
        self.dbscan_min_samples = int(getattr(cfg.superpoint, "dbscan_min_samples", 8))
        self.prototype_feature_normalize = bool(
            getattr(cfg.superpoint, "prototype_feature_normalize", True)
        )
        self.feature_weight = float(getattr(cfg.superpoint, "feature_weight", 1.0))
        self.coord_weight = float(getattr(cfg.superpoint, "coord_weight", 0.35))
        self.pca_dim = int(getattr(cfg.superpoint, "pca_dim", 16))
        self.assignment_chunk_size = int(
            getattr(cfg.superpoint, "assignment_chunk_size", 32768)
        )
        self.fallback_num_prototypes = int(
            getattr(cfg.superpoint, "fallback_num_prototypes", 256)
        )

        requested_device = getattr(
            cfg.superpoint,
            "assignment_device",
            getattr(cfg.foundation_model, "device", "cuda"),
        )
        self.assignment_device = self._resolve_device(requested_device)

    def generate_scene(self, scene_id: str):
        coord = self._load_point_coord(scene_id)
        point_feat, feature_meta = self._load_point_features(scene_id)
        if coord.shape[0] != point_feat.shape[0]:
            raise ValueError(
                f"Point count mismatch for {scene_id}: coord={coord.shape[0]} feat={point_feat.shape[0]}"
            )

        voxel_pack = self._build_voxel_prototypes(coord, point_feat)

        if self.method == "utonia_voxel_graph_overseg":
            voxel_labels, method_stats = self._graph_oversegment(voxel_pack)
            cluster_mode = "graph_overseg"
            noise_count = 0
        elif self.method == "utonia_voxel_prototype":
            voxel_labels, cluster_mode, noise_count, method_stats = self._global_cluster(voxel_pack)
        else:
            raise ValueError(f"Unsupported superpoint method: {self.method}")

        point_labels = voxel_labels[voxel_pack["point_to_voxel"]]

        stats = {
            "scene_id": scene_id,
            "num_points": int(point_feat.shape[0]),
            "num_voxels": int(voxel_pack["coord"].shape[0]),
            "num_clusters": int(np.unique(voxel_labels).shape[0]),
            "num_superpoints": int(np.unique(voxel_labels).shape[0]),
            "num_noise_points": int(noise_count),
            "cluster_mode": cluster_mode,
            "used_saved_spp_feat": bool(feature_meta["used_saved_spp_feat"]),
        }
        stats.update(method_stats)
        return point_labels, stats

    def save_scene(self, scene_id: str, point_labels: np.ndarray):
        os.makedirs(self.save_dir, exist_ok=True)
        output_path = os.path.join(self.save_dir, f"{scene_id}.pth")
        torch.save(torch.from_numpy(point_labels.astype(np.int64)), output_path)
        return output_path

    def _resolve_device(self, requested_device: str) -> torch.device:
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device

    def _load_point_coord(self, scene_id: str) -> np.ndarray:
        ply_path = os.path.join(self.original_ply_root, f"{scene_id}.ply")
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"Missing point cloud file: {ply_path}")

        with open(ply_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip() == "end_header":
                    break
            coord = np.loadtxt(handle, usecols=(0, 1, 2), dtype=np.float32)

        if coord.ndim != 2 or coord.shape[1] != 3:
            raise ValueError(f"Unexpected coord shape in '{ply_path}': {coord.shape}")
        return coord

    def _load_point_features(self, scene_id: str) -> Tuple[np.ndarray, Dict[str, bool]]:
        feature_path = os.path.join(self.point_feature_dir, f"{scene_id}.pth")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(
                f"Missing point features for {scene_id}: {feature_path}. "
                "Run scripts/generate_spp_feat.sh first."
            )

        data = _torch_load_local(feature_path, map_location="cpu")
        if isinstance(data, dict):
            point_feat = data.get("spp_feat", data.get("feat"))
            used_saved_spp_feat = "spp_feat" in data
        else:
            point_feat = data
            used_saved_spp_feat = True

        if point_feat is None:
            raise ValueError(f"Feature file {feature_path} does not contain point features.")
        if not isinstance(point_feat, torch.Tensor):
            point_feat = torch.as_tensor(point_feat)
        point_feat = point_feat.to(torch.float32).cpu().numpy()

        if point_feat.ndim != 2:
            raise ValueError(f"Unexpected point feature shape for {scene_id}: {tuple(point_feat.shape)}")

        return point_feat, {"used_saved_spp_feat": used_saved_spp_feat}

    def _build_voxel_prototypes(self, coord: np.ndarray, point_feat: np.ndarray):
        scaled = np.floor(coord / self.proto_grid_size).astype(np.int64)
        scaled -= scaled.min(axis=0, keepdims=True)
        grid_coord, point_to_voxel = np.unique(scaled, axis=0, return_inverse=True)
        num_voxels = grid_coord.shape[0]

        counts = np.bincount(point_to_voxel, minlength=num_voxels).astype(np.float32)
        voxel_coord = np.zeros((num_voxels, 3), dtype=np.float32)
        voxel_feat = np.zeros((num_voxels, point_feat.shape[1]), dtype=np.float32)

        np.add.at(voxel_coord, point_to_voxel, coord)
        np.add.at(voxel_feat, point_to_voxel, point_feat)

        voxel_coord /= counts[:, None]
        voxel_feat /= counts[:, None]

        return {
            "grid_coord": grid_coord,
            "point_to_voxel": point_to_voxel,
            "coord": voxel_coord,
            "feat": voxel_feat,
            "counts": counts,
        }

    def _graph_oversegment(self, voxel_pack) -> Tuple[np.ndarray, Dict[str, int]]:
        num_voxels = voxel_pack["coord"].shape[0]
        if num_voxels == 0:
            return np.zeros((0,), dtype=np.int64), {
                "num_graph_edges": 0,
                "num_kept_edges": 0,
                "num_small_merged": 0,
            }
        if num_voxels == 1:
            return np.zeros((1,), dtype=np.int64), {
                "num_graph_edges": 0,
                "num_kept_edges": 0,
                "num_small_merged": 0,
            }

        edges = self._build_local_graph_edges(
            voxel_pack["grid_coord"],
            connectivity=self.graph_connectivity,
        )
        if edges.shape[0] == 0:
            labels = np.arange(num_voxels, dtype=np.int64)
            return labels, {
                "num_graph_edges": 0,
                "num_kept_edges": 0,
                "num_small_merged": 0,
            }

        edge_sim = self._compute_edge_similarity(voxel_pack["feat"], edges)
        keep_mask = edge_sim >= self.graph_feature_threshold
        labels = self._connected_components_from_edges(num_voxels, edges[keep_mask])

        merged_small = 0
        if self.graph_merge_small_components:
            labels, merged_small = self._merge_small_components(
                labels,
                voxel_pack["counts"],
                edges,
                edge_sim,
            )

        labels = self._split_connected_components(
            voxel_pack["grid_coord"],
            labels,
            connectivity=self.component_connectivity,
        )
        labels = _compact_labels(labels)

        return labels, {
            "num_graph_edges": int(edges.shape[0]),
            "num_kept_edges": int(keep_mask.sum()),
            "num_small_merged": int(merged_small),
        }

    def _build_local_graph_edges(
        self,
        grid_coord: np.ndarray,
        *,
        connectivity: int,
    ) -> np.ndarray:
        offsets = _neighbor_offsets(connectivity)
        index_map = {tuple(coord.tolist()): idx for idx, coord in enumerate(grid_coord)}
        edges = []

        for idx, coord in enumerate(grid_coord):
            x, y, z = coord.tolist()
            for dx, dy, dz in offsets:
                nbr_idx = index_map.get((x + dx, y + dy, z + dz))
                if nbr_idx is None or nbr_idx <= idx:
                    continue
                edges.append((idx, nbr_idx))

        if not edges:
            return np.zeros((0, 2), dtype=np.int64)
        return np.asarray(edges, dtype=np.int64)

    def _compute_edge_similarity(self, voxel_feat: np.ndarray, edges: np.ndarray) -> np.ndarray:
        feat = voxel_feat.astype(np.float32, copy=False)
        feat = _row_normalize(feat)

        sims = np.empty((edges.shape[0],), dtype=np.float32)
        for start_idx in range(0, edges.shape[0], self.graph_edge_chunk_size):
            end_idx = min(edges.shape[0], start_idx + self.graph_edge_chunk_size)
            edge_chunk = edges[start_idx:end_idx]
            lhs = torch.from_numpy(feat[edge_chunk[:, 0]]).to(
                self.assignment_device,
                dtype=torch.float32,
            )
            rhs = torch.from_numpy(feat[edge_chunk[:, 1]]).to(
                self.assignment_device,
                dtype=torch.float32,
            )
            sims[start_idx:end_idx] = (lhs * rhs).sum(dim=1).cpu().numpy()
        return sims

    def _connected_components_from_edges(self, num_nodes: int, edges: np.ndarray) -> np.ndarray:
        if edges.shape[0] == 0:
            return np.arange(num_nodes, dtype=np.int64)

        uf = _UnionFind(num_nodes)
        for u, v in edges:
            uf.union(int(u), int(v))
        return uf.labels()

    def _merge_small_components(
        self,
        labels: np.ndarray,
        voxel_counts: np.ndarray,
        edges: np.ndarray,
        edge_sim: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        labels = labels.astype(np.int64, copy=True)
        merged = 0

        while True:
            unique, inverse = np.unique(labels, return_inverse=True)
            comp_voxel_counts = np.bincount(inverse)
            comp_point_counts = np.bincount(
                inverse,
                weights=voxel_counts.astype(np.float64, copy=False),
            )
            small_components = unique[
                (comp_voxel_counts < self.graph_min_component_voxels)
                | (comp_point_counts < self.graph_min_component_points)
            ]
            if small_components.size == 0:
                break

            changed = False
            for comp in small_components.tolist():
                label_u = labels[edges[:, 0]]
                label_v = labels[edges[:, 1]]
                edge_mask = ((label_u == comp) & (label_v != comp)) | ((label_v == comp) & (label_u != comp))
                if not np.any(edge_mask):
                    continue

                nbr_labels = np.where(label_u[edge_mask] == comp, label_v[edge_mask], label_u[edge_mask])
                nbr_sims = edge_sim[edge_mask]
                if nbr_labels.size == 0:
                    continue

                uniq_nbr, inv_nbr = np.unique(nbr_labels, return_inverse=True)
                best_scores = np.full((uniq_nbr.shape[0],), -np.inf, dtype=np.float32)
                np.maximum.at(best_scores, inv_nbr, nbr_sims)
                best_idx = int(np.argmax(best_scores))
                if best_scores[best_idx] < self.graph_merge_sim_threshold:
                    continue

                labels[labels == comp] = uniq_nbr[best_idx]
                merged += 1
                changed = True

            if not changed:
                break

            labels = _compact_labels(labels)

        return labels, merged

    def _global_cluster(self, voxel_pack):
        voxel_embedding = self._build_cluster_embedding(
            voxel_pack["coord"],
            voxel_pack["feat"],
        )

        voxel_labels, cluster_mode = self._cluster_voxels(voxel_embedding)
        voxel_labels, noise_count = self._assign_noise_voxels(voxel_embedding, voxel_labels)
        voxel_labels = self._split_connected_components(
            voxel_pack["grid_coord"],
            voxel_labels,
            connectivity=self.component_connectivity,
        )
        voxel_labels = _compact_labels(voxel_labels)
        return voxel_labels, cluster_mode, noise_count, {
            "num_graph_edges": 0,
            "num_kept_edges": 0,
            "num_small_merged": 0,
        }

    def _build_cluster_embedding(self, voxel_coord: np.ndarray, voxel_feat: np.ndarray) -> np.ndarray:
        feat_embed = voxel_feat.astype(np.float32, copy=False)
        if self.prototype_feature_normalize:
            feat_embed = _row_normalize(feat_embed)
        if (
            self.pca_dim > 0
            and feat_embed.shape[1] > self.pca_dim
            and feat_embed.shape[0] > self.pca_dim
        ):
            pca = PCA(
                n_components=self.pca_dim,
                svd_solver="randomized",
                random_state=0,
            )
            feat_embed = pca.fit_transform(feat_embed)
        feat_embed = _standardize(feat_embed.astype(np.float32, copy=False))
        coord_embed = _standardize(voxel_coord.astype(np.float32, copy=False))
        return np.concatenate(
            [
                self.feature_weight * feat_embed,
                self.coord_weight * coord_embed,
            ],
            axis=1,
        )

    def _cluster_voxels(self, voxel_embedding: np.ndarray) -> Tuple[np.ndarray, str]:
        if voxel_embedding.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64), "empty"
        if voxel_embedding.shape[0] == 1:
            return np.zeros((1,), dtype=np.int64), "single"

        if self.cluster_method == "hdbscan":
            labels = self._run_hdbscan(voxel_embedding)
            cluster_mode = "hdbscan"
        elif self.cluster_method == "dbscan":
            clusterer = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
                metric="euclidean",
            )
            labels = clusterer.fit_predict(voxel_embedding)
            cluster_mode = "dbscan"
        else:
            raise ValueError(f"Unsupported cluster method: {self.cluster_method}")

        if np.any(labels >= 0):
            return labels.astype(np.int64), cluster_mode

        fallback_labels = self._run_fallback_kmeans(voxel_embedding)
        return fallback_labels.astype(np.int64), "fallback_kmeans"

    def _run_hdbscan(self, voxel_embedding: np.ndarray) -> np.ndarray:
        try:
            import hdbscan
        except ImportError:
            try:
                from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
            except ImportError as exc:
                raise ImportError(
                    "HDBSCAN requested but neither 'hdbscan' nor sklearn.cluster.HDBSCAN is available."
                ) from exc

            clusterer = SklearnHDBSCAN(
                min_cluster_size=self.hdbscan_min_cluster_size,
                min_samples=self.hdbscan_min_samples,
            )
            return clusterer.fit_predict(voxel_embedding)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            min_samples=self.hdbscan_min_samples,
            metric="euclidean",
        )
        return clusterer.fit_predict(voxel_embedding)

    def _run_fallback_kmeans(self, voxel_embedding: np.ndarray) -> np.ndarray:
        num_points = voxel_embedding.shape[0]
        heuristic = max(1, int(round(np.sqrt(float(num_points)))))
        if self.fallback_num_prototypes > 0:
            num_clusters = min(num_points, min(self.fallback_num_prototypes, heuristic))
        else:
            num_clusters = min(num_points, heuristic)

        clusterer = MiniBatchKMeans(
            n_clusters=max(1, num_clusters),
            batch_size=min(8192, max(256, num_points)),
            n_init=10,
            random_state=0,
        )
        return clusterer.fit_predict(voxel_embedding)

    def _assign_noise_voxels(
        self,
        voxel_embedding: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        labels = labels.astype(np.int64, copy=True)
        noise_mask = labels < 0
        noise_count = int(noise_mask.sum())
        if noise_count == 0:
            return labels, 0

        valid_labels = np.unique(labels[labels >= 0])
        if valid_labels.size == 0:
            raise RuntimeError("Noise reassignment requires at least one valid cluster.")

        centroids = np.zeros((valid_labels.size, voxel_embedding.shape[1]), dtype=np.float32)
        counts = np.zeros((valid_labels.size,), dtype=np.float32)
        label_map = {int(label): idx for idx, label in enumerate(valid_labels.tolist())}
        mapped = np.vectorize(label_map.get, otypes=[np.int64])(labels[~noise_mask])
        np.add.at(centroids, mapped, voxel_embedding[~noise_mask])
        np.add.at(counts, mapped, 1.0)
        centroids /= counts[:, None]

        assigned = self._assign_by_l2(voxel_embedding[noise_mask], centroids)
        labels[noise_mask] = valid_labels[assigned]
        return labels, noise_count

    def _assign_by_l2(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        if data.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)

        centroid_tensor = torch.from_numpy(centroids).to(
            self.assignment_device,
            dtype=torch.float32,
        )
        assigned = np.empty((data.shape[0],), dtype=np.int64)
        centroid_sq = (centroid_tensor * centroid_tensor).sum(dim=1)

        for start_idx in range(0, data.shape[0], self.assignment_chunk_size):
            end_idx = min(data.shape[0], start_idx + self.assignment_chunk_size)
            chunk = torch.from_numpy(data[start_idx:end_idx]).to(
                self.assignment_device,
                dtype=torch.float32,
            )
            chunk_sq = (chunk * chunk).sum(dim=1, keepdim=True)
            dist = chunk_sq + centroid_sq.unsqueeze(0) - 2.0 * (chunk @ centroid_tensor.t())
            assigned[start_idx:end_idx] = dist.argmin(dim=1).cpu().numpy()
        return assigned

    def _split_connected_components(
        self,
        grid_coord: np.ndarray,
        labels: np.ndarray,
        *,
        connectivity: int,
    ) -> np.ndarray:
        if grid_coord.shape[0] != labels.shape[0]:
            raise ValueError(
                f"grid_coord/labels size mismatch: {grid_coord.shape[0]} != {labels.shape[0]}"
            )
        if labels.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)

        index_map = {tuple(coord.tolist()): idx for idx, coord in enumerate(grid_coord)}
        offsets = _neighbor_offsets(connectivity)
        final_labels = -np.ones_like(labels, dtype=np.int64)
        next_label = 0

        for base_label in np.unique(labels):
            member_indices = np.where(labels == base_label)[0]
            if member_indices.size == 0:
                continue
            visited = set()
            for seed in member_indices:
                if seed in visited:
                    continue
                queue = deque([seed])
                visited.add(seed)
                final_labels[seed] = next_label
                while queue:
                    current = queue.popleft()
                    x, y, z = grid_coord[current].tolist()
                    for dx, dy, dz in offsets:
                        nbr_idx = index_map.get((x + dx, y + dy, z + dz))
                        if nbr_idx is None:
                            continue
                        if nbr_idx in visited or labels[nbr_idx] != base_label:
                            continue
                        visited.add(nbr_idx)
                        final_labels[nbr_idx] = next_label
                        queue.append(nbr_idx)
                next_label += 1

        if np.any(final_labels < 0):
            raise RuntimeError("Connected-component split produced unlabeled voxels.")
        return final_labels
