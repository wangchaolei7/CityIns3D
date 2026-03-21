import os
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
from plyfile import PlyData


def resolve_scene_path(root: str, scene_id: str, exts: Tuple[str, ...] = (".txt", ".ply")) -> str:
    if os.path.isabs(scene_id) and os.path.exists(scene_id):
        return scene_id

    stem, ext = os.path.splitext(scene_id)
    if ext:
        candidate = os.path.join(root, scene_id)
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"Missing scene file: {candidate}")

    for suffix in exts:
        candidate = os.path.join(root, f"{scene_id}{suffix}")
        if os.path.exists(candidate):
            return candidate

    tried = ", ".join(os.path.join(root, f"{scene_id}{suffix}") for suffix in exts)
    raise FileNotFoundError(f"Missing scene file for '{scene_id}'. Tried: {tried}")


def _load_txt_matrix(path: str) -> np.ndarray:
    try:
        data = np.genfromtxt(path, delimiter=",", dtype=np.float32)
    except ValueError:
        data = np.genfromtxt(path, dtype=np.float32)

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Unexpected txt point cloud shape for '{path}': {data.shape}")
    return data


def load_pointcloud_xyz_rgb(path: str) -> Tuple[np.ndarray, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        data = _load_txt_matrix(path)
        xyz = data[:, :3].astype(np.float32, copy=False)
        if data.shape[1] >= 6:
            rgb = data[:, 3:6].astype(np.float32, copy=False)
            if rgb.size > 0 and rgb.max() > 1.0 + 1e-6:
                rgb = rgb / 255.0
        else:
            rgb = np.zeros_like(xyz, dtype=np.float32)
        return xyz, rgb

    if ext == ".ply":
        scene_pcd = o3d.io.read_point_cloud(str(path))
        xyz = np.asarray(scene_pcd.points, dtype=np.float32)
        rgb = np.asarray(scene_pcd.colors, dtype=np.float32)
        if rgb.size == 0:
            rgb = np.zeros_like(xyz, dtype=np.float32)
        return xyz, rgb

    raise ValueError(f"Unsupported point cloud extension: {path}")


def load_pointcloud_xyz(path: str) -> np.ndarray:
    xyz, _ = load_pointcloud_xyz_rgb(path)
    return xyz


def load_semantic_instance(path: str) -> Tuple[np.ndarray, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        data = _load_txt_matrix(path)
        if data.shape[1] < 8:
            raise ValueError(f"TXT GT requires at least 8 columns (xyzrgbsemanticinstance): {path}")
        sem = data[:, 6].astype(np.int32, copy=False)
        inst = data[:, 7].astype(np.int32, copy=False)
        return sem, inst

    if ext == ".ply":
        plydata = PlyData.read(path)
        vertex = plydata["vertex"]
        sem = np.asarray(vertex["semantic"], dtype=np.int32)
        inst = np.asarray(vertex["instance"], dtype=np.int32)
        return sem, inst

    raise ValueError(f"Unsupported GT extension: {path}")


def load_xyz_semantic_instance(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        data = _load_txt_matrix(path)
        if data.shape[1] < 8:
            raise ValueError(f"TXT GT requires at least 8 columns (xyzrgbsemanticinstance): {path}")
        xyz = data[:, :3].astype(np.float32, copy=False)
        sem = data[:, 6].astype(np.int32, copy=False)
        inst = data[:, 7].astype(np.int32, copy=False)
        return xyz, sem, inst

    if ext == ".ply":
        plydata = PlyData.read(path)
        vertex = plydata["vertex"]
        xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
        sem = np.asarray(vertex["semantic"], dtype=np.int32)
        inst = np.asarray(vertex["instance"], dtype=np.int32)
        return xyz, sem, inst

    raise ValueError(f"Unsupported GT extension: {path}")


def save_txt_scene_copy(src_path: str, dst_path: str):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        return
    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        dst.write(src.read())
