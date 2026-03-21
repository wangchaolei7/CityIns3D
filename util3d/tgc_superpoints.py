import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from open3dis.dataset_outdoor.stpls3d_io import load_pointcloud_xyz_rgb, resolve_scene_path

def _torch_load_local(path: str, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


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
    return np.searchsorted(unique, labels).astype(np.int64, copy=False)


def _neighbor_offsets(connectivity: int):
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
    raise ValueError(f"Unsupported graph connectivity: {connectivity}")


class TgcSuperpointGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.original_ply_root = cfg.data.original_ply
        self.point_feature_dir = cfg.data.point_features_path
        self.device = self._resolve_device(getattr(cfg.foundation_model, "device", "cuda"))

        tgc_repo_path = os.path.abspath(
            os.path.expanduser(
                getattr(
                    cfg.foundation_model,
                    "tgc_repo_path",
                    "./segmenter3d/torch-graph-components/src",
                )
            )
        )
        if tgc_repo_path not in sys.path:
            sys.path.insert(0, tgc_repo_path)

        from torch_graph_components import merge_components_by_contour_prior

        self.merge_components_by_contour_prior = merge_components_by_contour_prior

        sp_cfg = cfg.superpoint
        self.native_output_dir = getattr(
            sp_cfg,
            "tgc_native_output_dir",
            os.path.join(os.path.dirname(cfg.data.spp_path.rstrip("/")), "tgc_superpoints_native"),
        )
        self.utonia_output_dir = getattr(
            sp_cfg,
            "tgc_utonia_output_dir",
            os.path.join(os.path.dirname(cfg.data.spp_path.rstrip("/")), "tgc_superpoints_utonia"),
        )
        shared_voxel_size = float(getattr(sp_cfg, "tgc_voxel_size", 0.1))
        self.native_voxel_size = float(getattr(sp_cfg, "tgc_native_voxel_size", max(0.25, shared_voxel_size)))
        self.utonia_voxel_size = float(getattr(sp_cfg, "tgc_utonia_voxel_size", max(0.2, shared_voxel_size)))
        self.graph_connectivity = int(getattr(sp_cfg, "tgc_graph_connectivity", 6))
        shared_reg = float(getattr(sp_cfg, "tgc_reg", 1.0))
        self.native_reg = float(getattr(sp_cfg, "tgc_native_reg", max(4.0, shared_reg)))
        self.utonia_reg = float(getattr(sp_cfg, "tgc_utonia_reg", shared_reg))
        shared_min_size = int(getattr(sp_cfg, "tgc_min_size", 64))
        self.native_min_size = int(getattr(sp_cfg, "tgc_native_min_size", max(256, shared_min_size)))
        self.utonia_min_size = int(getattr(sp_cfg, "tgc_utonia_min_size", max(128, shared_min_size)))
        self.merge_only_small = bool(getattr(sp_cfg, "tgc_merge_only_small", False))
        shared_k = int(getattr(sp_cfg, "tgc_k", -1))
        self.native_k = int(getattr(sp_cfg, "tgc_native_k", max(4, shared_k)))
        self.utonia_k = int(getattr(sp_cfg, "tgc_utonia_k", shared_k))
        shared_w_adjacency = float(getattr(sp_cfg, "tgc_w_adjacency", -1.0))
        self.native_w_adjacency = float(getattr(sp_cfg, "tgc_native_w_adjacency", 1.0 if shared_w_adjacency <= 0 else shared_w_adjacency))
        self.utonia_w_adjacency = float(getattr(sp_cfg, "tgc_utonia_w_adjacency", shared_w_adjacency))
        self.sharding = getattr(sp_cfg, "tgc_sharding", 0.25)
        self.reduce = getattr(sp_cfg, "tgc_reduce", "add")

        self.native_color_weight = float(getattr(sp_cfg, "tgc_native_color_weight", 1.0))
        self.native_coord_weight = float(getattr(sp_cfg, "tgc_native_coord_weight", 0.25))

        self.utonia_pca_dim = int(getattr(sp_cfg, "tgc_utonia_pca_dim", 32))
        self.utonia_pca_fit_samples = int(getattr(sp_cfg, "tgc_utonia_pca_fit_samples", 20000))
        self.utonia_edge_weight_mode = getattr(sp_cfg, "tgc_utonia_edge_weight_mode", "similarity")
        self.utonia_edge_weight_floor = float(getattr(sp_cfg, "tgc_utonia_edge_weight_floor", 0.05))
        self.auto_sparse_txt = bool(getattr(sp_cfg, "tgc_auto_sparse_txt", True))
        self.txt_utonia_voxel_size = float(getattr(sp_cfg, "tgc_txt_utonia_voxel_size", 1.0))
        self.txt_graph_connectivity = int(getattr(sp_cfg, "tgc_txt_graph_connectivity", 26))
        self.txt_utonia_reg = float(getattr(sp_cfg, "tgc_txt_utonia_reg", 120.0))
        self.txt_utonia_min_size = int(getattr(sp_cfg, "tgc_txt_utonia_min_size", self.utonia_min_size))
        self.txt_utonia_k = int(getattr(sp_cfg, "tgc_txt_utonia_k", 8))
        self.txt_utonia_w_adjacency = float(getattr(sp_cfg, "tgc_txt_utonia_w_adjacency", 1.0))
        self.txt_utonia_edge_weight_floor = float(
            getattr(sp_cfg, "tgc_txt_utonia_edge_weight_floor", 0.25)
        )

    def generate_scene(self, scene_id: str, method: str):
        method = method.lower()
        if method not in {"tgc_native", "tgc_utonia"}:
            raise ValueError(f"Unsupported TGC method: {method}")

        stage_start = time.perf_counter()
        coord, color, source_path = self._load_point_cloud(scene_id)
        load_time = time.perf_counter() - stage_start

        runtime_cfg = self._resolve_runtime_cfg(method, source_path)

        stage_start = time.perf_counter()
        point_feat = self._load_point_features(scene_id) if method == "tgc_utonia" else None
        voxel_size = runtime_cfg["voxel_size"]
        voxel_pack = self._voxelize(coord, color, point_feat, voxel_size=voxel_size)
        voxel_time = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        X, S, E, W, P = self._build_tgc_inputs(voxel_pack, method, runtime_cfg)
        build_time = time.perf_counter() - stage_start

        reg = runtime_cfg["reg"]
        min_size = runtime_cfg["min_size"]
        k = runtime_cfg["k"]
        w_adjacency = runtime_cfg["w_adjacency"]

        stage_start = time.perf_counter()
        if E.shape[1] == 0:
            voxel_labels = torch.arange(X.shape[0], device=X.device, dtype=torch.long)
            depth = 0
        else:
            voxel_labels, depth, _ = self.merge_components_by_contour_prior(
                X,
                S,
                E,
                W,
                reg,
                min_size,
                merge_only_small=self.merge_only_small,
                P=P,
                k=k,
                w_adjacency=w_adjacency,
                sharding=self.sharding,
                reduce=self.reduce,
                verbose=False,
            )
        partition_time = time.perf_counter() - stage_start

        voxel_labels = voxel_labels.detach().cpu().numpy().astype(np.int64, copy=False)
        voxel_labels = _compact_labels(voxel_labels)
        point_labels = voxel_labels[voxel_pack["point_to_voxel"]]

        stats = {
            "scene_id": scene_id,
            "method": method,
            "num_points": int(coord.shape[0]),
            "num_voxels": int(voxel_pack["coord"].shape[0]),
            "num_edges": int(E.shape[1]),
            "num_superpoints": int(np.unique(voxel_labels).shape[0]),
            "depth": int(depth),
            "used_utonia_feature": method == "tgc_utonia",
            "device": str(self.device),
            "source_format": os.path.splitext(source_path)[1].lstrip(".").lower(),
            "voxel_size": float(voxel_size),
            "reg": float(reg),
            "min_size": int(min_size),
            "graph_connectivity": int(runtime_cfg["graph_connectivity"]),
            "k": int(k),
            "w_adjacency": float(w_adjacency),
            "load_time": float(load_time),
            "voxel_time": float(voxel_time),
            "build_time": float(build_time),
            "partition_time": float(partition_time),
        }
        return point_labels, stats

    def save_scene(self, scene_id: str, point_labels: np.ndarray, method: str, output_dir: str = None):
        target_dir = output_dir or self._resolve_output_dir(method)
        os.makedirs(target_dir, exist_ok=True)
        output_path = os.path.join(target_dir, f"{scene_id}.pth")
        torch.save(torch.from_numpy(point_labels.astype(np.int64, copy=False)), output_path)
        return output_path

    def _resolve_output_dir(self, method: str) -> str:
        return self.utonia_output_dir if method == "tgc_utonia" else self.native_output_dir

    def _resolve_device(self, requested_device: str) -> torch.device:
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device

    def _load_point_cloud(self, scene_id: str) -> Tuple[np.ndarray, np.ndarray, str]:
        scene_path = resolve_scene_path(self.original_ply_root, scene_id)
        coord, color = load_pointcloud_xyz_rgb(scene_path)
        if coord.ndim != 2 or coord.shape[1] != 3:
            raise ValueError(f"Unexpected point cloud shape in '{scene_path}': {coord.shape}")
        if color.size > 0 and color.max() <= 1.0 + 1e-6:
            color = color * 255.0
        return coord, color.astype(np.float32, copy=False), scene_path

    def _resolve_runtime_cfg(self, method: str, source_path: str):
        if method == "tgc_native":
            return {
                "voxel_size": self.native_voxel_size,
                "graph_connectivity": self.graph_connectivity,
                "reg": self.native_reg,
                "min_size": self.native_min_size,
                "k": self.native_k,
                "w_adjacency": self.native_w_adjacency,
                "edge_weight_floor": self.utonia_edge_weight_floor,
            }

        source_ext = os.path.splitext(source_path)[1].lower()
        if self.auto_sparse_txt and source_ext == ".txt":
            return {
                "voxel_size": self.txt_utonia_voxel_size,
                "graph_connectivity": self.txt_graph_connectivity,
                "reg": self.txt_utonia_reg,
                "min_size": self.txt_utonia_min_size,
                "k": self.txt_utonia_k,
                "w_adjacency": self.txt_utonia_w_adjacency,
                "edge_weight_floor": self.txt_utonia_edge_weight_floor,
            }

        return {
            "voxel_size": self.utonia_voxel_size,
            "graph_connectivity": self.graph_connectivity,
            "reg": self.utonia_reg,
            "min_size": self.utonia_min_size,
            "k": self.utonia_k,
            "w_adjacency": self.utonia_w_adjacency,
            "edge_weight_floor": self.utonia_edge_weight_floor,
        }

    def _load_point_features(self, scene_id: str) -> np.ndarray:
        feature_path = os.path.join(self.point_feature_dir, f"{scene_id}.pth")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(
                f"Missing point features for {scene_id}: {feature_path}. "
                "Run scripts/generate_spp_feat.sh first."
            )

        data = _torch_load_local(feature_path, map_location="cpu")
        if isinstance(data, dict):
            point_feat = data.get("spp_feat", data.get("feat"))
        else:
            point_feat = data

        if point_feat is None:
            raise ValueError(f"Feature file {feature_path} does not contain usable point features.")
        if not isinstance(point_feat, torch.Tensor):
            point_feat = torch.as_tensor(point_feat)
        point_feat = point_feat.to(torch.float32).cpu().numpy()
        if point_feat.ndim != 2:
            raise ValueError(f"Unexpected point feature shape for {scene_id}: {tuple(point_feat.shape)}")
        return point_feat

    def _voxelize(
        self,
        coord: np.ndarray,
        color: np.ndarray,
        point_feat: np.ndarray = None,
        *,
        voxel_size: float,
    ):
        scaled = np.floor(coord / voxel_size).astype(np.int64)
        scaled -= scaled.min(axis=0, keepdims=True)
        grid_coord, point_to_voxel = np.unique(scaled, axis=0, return_inverse=True)
        num_voxels = grid_coord.shape[0]

        counts = np.bincount(point_to_voxel, minlength=num_voxels).astype(np.float32)
        voxel_coord = np.zeros((num_voxels, 3), dtype=np.float32)
        voxel_color = np.zeros((num_voxels, 3), dtype=np.float32)
        np.add.at(voxel_coord, point_to_voxel, coord)
        np.add.at(voxel_color, point_to_voxel, color)
        voxel_coord /= counts[:, None]
        voxel_color /= counts[:, None]

        voxel_feat = None
        if point_feat is not None:
            voxel_feat = np.zeros((num_voxels, point_feat.shape[1]), dtype=np.float32)
            np.add.at(voxel_feat, point_to_voxel, point_feat)
            voxel_feat /= counts[:, None]

        return {
            "grid_coord": grid_coord,
            "point_to_voxel": point_to_voxel,
            "coord": voxel_coord,
            "color": voxel_color,
            "counts": counts,
            "feat": voxel_feat,
        }

    def _build_graph(self, grid_coord: np.ndarray, connectivity: int):
        offsets = _neighbor_offsets(connectivity)
        index_map = {tuple(coord.tolist()): idx for idx, coord in enumerate(grid_coord)}
        edges = []
        weights = []

        for idx, coord in enumerate(grid_coord):
            x, y, z = coord.tolist()
            for dx, dy, dz in offsets:
                nbr_idx = index_map.get((x + dx, y + dy, z + dz))
                if nbr_idx is None or nbr_idx <= idx:
                    continue
                edges.append((idx, nbr_idx))
                weights.append(1.0 / float(np.sqrt(dx * dx + dy * dy + dz * dz)))

        if not edges:
            return (
                torch.zeros((2, 0), dtype=torch.long, device=self.device),
                torch.zeros((0,), dtype=torch.float32, device=self.device),
            )

        edge_index = torch.as_tensor(np.asarray(edges, dtype=np.int64).T, device=self.device)
        edge_weight = torch.as_tensor(np.asarray(weights, dtype=np.float32), device=self.device)
        return edge_index, edge_weight

    def _build_native_node_features(self, voxel_coord: np.ndarray, voxel_color: np.ndarray) -> np.ndarray:
        color_embed = _standardize((voxel_color / 255.0).astype(np.float32, copy=False))
        coord_embed = _standardize(voxel_coord.astype(np.float32, copy=False))
        return np.concatenate(
            [
                self.native_color_weight * color_embed,
                self.native_coord_weight * coord_embed,
            ],
            axis=1,
        ).astype(np.float32, copy=False)

    def _build_utonia_node_features(self, voxel_feat: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        if voxel_feat is None:
            raise ValueError("Utonia voxel features are required for method 'tgc_utonia'.")

        feat = voxel_feat.astype(np.float32, copy=False)
        feat_norm = _row_normalize(feat)
        feat_tensor = torch.from_numpy(feat_norm).to(
            self.device,
            dtype=torch.float32,
        )

        if (
            self.utonia_pca_dim > 0
            and feat_norm.shape[1] > self.utonia_pca_dim
            and feat_norm.shape[0] > self.utonia_pca_dim
        ):
            sample_count = min(self.utonia_pca_fit_samples, feat_norm.shape[0])
            if sample_count < feat_norm.shape[0]:
                sample_idx = np.random.RandomState(0).choice(feat_norm.shape[0], size=sample_count, replace=False)
                feat_fit = feat_norm[sample_idx]
            else:
                feat_fit = feat_norm

            from sklearn.decomposition import PCA

            pca = PCA(
                n_components=self.utonia_pca_dim,
                svd_solver="randomized",
                random_state=0,
            )
            pca.fit(feat_fit)
            feat = pca.transform(feat_norm)
        else:
            feat = feat_norm

        feat = _standardize(feat.astype(np.float32, copy=False))
        return feat, feat_tensor

    def _build_tgc_inputs(self, voxel_pack: Dict[str, np.ndarray], method: str, runtime_cfg: Dict[str, float]):
        edge_index, edge_weight = self._build_graph(
            voxel_pack["grid_coord"],
            connectivity=int(runtime_cfg["graph_connectivity"]),
        )

        if method == "tgc_native":
            node_feat = self._build_native_node_features(voxel_pack["coord"], voxel_pack["color"])
            node_feat_t = torch.from_numpy(node_feat).to(self.device, dtype=torch.float32)
            edge_weight_t = edge_weight
        else:
            node_feat, feat_tensor = self._build_utonia_node_features(voxel_pack["feat"])
            node_feat_t = torch.from_numpy(node_feat).to(self.device, dtype=torch.float32)
            edge_weight_t = edge_weight
            if edge_index.shape[1] > 0 and self.utonia_edge_weight_mode == "similarity":
                sim = F.cosine_similarity(feat_tensor[edge_index[0]], feat_tensor[edge_index[1]], dim=1)
                sim = sim.clamp(min=0.0)
                sim = sim.clamp_min(float(runtime_cfg["edge_weight_floor"]))
                edge_weight_t = edge_weight_t * sim

        size_t = torch.from_numpy(voxel_pack["counts"]).to(self.device, dtype=torch.float32)
        pos_t = torch.from_numpy(voxel_pack["coord"]).to(self.device, dtype=torch.float32)
        return node_feat_t, size_t, edge_index, edge_weight_t, pos_t
