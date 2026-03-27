#!/usr/bin/env python3
"""
功能说明：
1. 读取 STPLS3D 场景 05_points_GTv3_83 的 2D mask 结果（默认使用 stage12 融合结果）。
2. 对指定帧的“当前帧所有 masks”取并集，并将其反投影到 3D 点云上。
3. 为每一帧分别导出一份整场景 xyzrgb 文本：
   - 落在该帧任意 mask 投影内的点，用指定高亮色赋色。
   - 场景其余点，用与 vivid 视觉风格兼容的暖灰色赋色。
4. 默认保存 0001 / 0004 / 0000 三帧，输出格式均为 xyzrgb 的 txt 文件。

默认输入：
- 2D masks: /data1/wangcl/project/CityIns3D/stpls3d/version_SAM3/mask_sam3_stage12/05_points_GTv3_83.pth
- 点云场景: /data1/wangcl/dataset/open_3d/STPLS3D_Open3DIS/3D/groundtruth/05_points_GTv3_83.txt
- 2D 场景目录: /data1/wangcl/dataset/open_3d/STPLS3D_Open3DIS/2D/05_points_GTv3_83
- 输出目录: /data1/wangcl/project/CityIns3D/graduation

说明：
- 该脚本是独立可视化工具，不修改主代码逻辑。
- 需要在带 torch + pycocotools 的环境中运行，例如 SAM3 环境。
- 当前使用与主流程一致的深度可见性判断：dynamic threshold，scale=0.01，min=0.05。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch


DEFAULT_SCENE_ID = '05_points_GTv3_83'
DEFAULT_MASK_PATH = Path('/data1/wangcl/project/CityIns3D/stpls3d/version_SAM3/mask_sam3_stage12/05_points_GTv3_83.pth')
DEFAULT_SCENE_TXT = Path('/data1/wangcl/dataset/open_3d/STPLS3D_Open3DIS/3D/groundtruth/05_points_GTv3_83.txt')
DEFAULT_SCENE_2D_ROOT = Path('/data1/wangcl/dataset/open_3d/STPLS3D_Open3DIS/2D/05_points_GTv3_83')
DEFAULT_OUTPUT_DIR = Path('/data1/wangcl/project/CityIns3D/graduation')
BACKGROUND_RGB = np.array([178, 168, 146], dtype=np.uint8)
FRAME_HIGHLIGHT_HEX = {
    '0001': '#F079FF',
    '0004': '#FFEB94',
    '0000': '#FFB071',
}
VIS_SCALE = 0.01
VIS_MIN = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Back-project all masks of selected STPLS3D frames to the scene point cloud and export xyzrgb txt files.'
    )
    parser.add_argument('--scene-id', type=str, default=DEFAULT_SCENE_ID)
    parser.add_argument('--mask-path', type=Path, default=DEFAULT_MASK_PATH)
    parser.add_argument('--scene-txt', type=Path, default=DEFAULT_SCENE_TXT)
    parser.add_argument('--scene-2d-root', type=Path, default=DEFAULT_SCENE_2D_ROOT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--frames', nargs='+', default=['0001', '0004', '0000'])
    return parser.parse_args()


def hex_to_rgb(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip('#')
    return np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)


def load_xyz(scene_txt: Path) -> np.ndarray:
    data = np.loadtxt(scene_txt, delimiter=',', dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f'Expected at least 3 columns in {scene_txt}, got shape {data.shape}')
    return data[:, :3]


def load_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise FileNotFoundError(f'Failed to read depth image: {path}')
    return depth.astype(np.float32)


def load_valid_mask(path: Path) -> np.ndarray:
    valid = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if valid is None:
        raise FileNotFoundError(f'Failed to read valid mask: {path}')
    return valid > 0


def load_pose(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def load_intrinsic(path: Path) -> np.ndarray:
    intrinsic = np.load(path).astype(np.float32)
    if intrinsic.shape == (4, 4):
        intrinsic = intrinsic[:3, :3]
    if intrinsic.shape != (3, 3):
        raise ValueError(f'Unexpected intrinsic shape: {intrinsic.shape}')
    return intrinsic


def decode_union_mask(mask_entries: list[dict], image_shape: tuple[int, int]) -> np.ndarray:
    if not mask_entries:
        return np.zeros(image_shape, dtype=np.bool_)
    union_mask = np.zeros(image_shape, dtype=np.bool_)
    for entry in mask_entries:
        decoded = mask_util.decode(entry)
        if decoded.ndim == 3:
            decoded = decoded[..., 0]
        decoded = decoded.astype(bool)
        if decoded.shape != image_shape:
            decoded = cv2.resize(decoded.astype(np.uint8), (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        union_mask |= decoded
    return union_mask


def project_points_to_frame(
    xyz: np.ndarray,
    pose: np.ndarray,
    intrinsic: np.ndarray,
    depth: np.ndarray,
    valid_region: np.ndarray,
    union_mask: np.ndarray,
) -> np.ndarray:
    n_points = xyz.shape[0]
    xyz_h = np.concatenate([xyz, np.ones((n_points, 1), dtype=np.float32)], axis=1).T
    world_to_camera = np.linalg.inv(pose)
    proj = world_to_camera @ xyz_h

    z_cam = proj[2]
    front_mask = z_cam > 1e-6

    px = np.round((proj[0] * intrinsic[0, 0]) / np.maximum(z_cam, 1e-6) + intrinsic[0, 2]).astype(np.int64)
    py = np.round((proj[1] * intrinsic[1, 1]) / np.maximum(z_cam, 1e-6) + intrinsic[1, 2]).astype(np.int64)

    h, w = depth.shape
    inside = front_mask & (px >= 0) & (py >= 0) & (px < w) & (py < h)
    if not np.any(inside):
        return np.zeros((n_points,), dtype=np.bool_)

    visible = np.zeros((n_points,), dtype=np.bool_)
    idx = np.flatnonzero(inside)
    sampled_depth = depth[py[idx], px[idx]]
    tol = np.maximum(VIS_SCALE * z_cam[idx], VIS_MIN)
    depth_ok = np.abs(sampled_depth - z_cam[idx]) <= tol
    valid_ok = valid_region[py[idx], px[idx]]
    mask_ok = union_mask[py[idx], px[idx]]
    visible[idx] = depth_ok & valid_ok & mask_ok
    return visible


def colorize_scene(xyz: np.ndarray, highlight_mask: np.ndarray, highlight_rgb: np.ndarray) -> np.ndarray:
    rgb = np.repeat(BACKGROUND_RGB[None, :], repeats=xyz.shape[0], axis=0)
    rgb[highlight_mask] = highlight_rgb
    return np.concatenate([xyz, rgb.astype(np.float64)], axis=1)


def save_xyzrgb(path: Path, xyzrgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        xyzrgb,
        fmt=['%.6f', '%.6f', '%.6f', '%.0f', '%.0f', '%.0f'],
        delimiter=',',
    )


def frame_file(scene_2d_root: Path, subdir: str, frame_id: str, ext: str) -> Path:
    return scene_2d_root / subdir / f'{frame_id}{ext}'


def main() -> None:
    args = parse_args()
    xyz = load_xyz(args.scene_txt)
    mask_pack = torch.load(args.mask_path, map_location='cpu', weights_only=False)

    print(f'[info] scene_id={args.scene_id}')
    print(f'[info] scene_points={xyz.shape[0]}')
    print(f'[info] mask_path={args.mask_path}')

    for frame_id in args.frames:
        if frame_id not in mask_pack:
            raise KeyError(f'Frame {frame_id} not found in {args.mask_path}')
        if frame_id not in FRAME_HIGHLIGHT_HEX:
            raise KeyError(f'No highlight color configured for frame {frame_id}')

        frame_masks = mask_pack[frame_id]['masks']
        depth = load_depth(frame_file(args.scene_2d_root, 'depth', frame_id, '.tif'))
        pose = load_pose(frame_file(args.scene_2d_root, 'pose', frame_id, '.npy'))
        intrinsic = load_intrinsic(frame_file(args.scene_2d_root, 'intrinsic', frame_id, '.npy'))
        valid_region = load_valid_mask(frame_file(args.scene_2d_root, 'valid_mask', frame_id, '.png'))
        union_mask = decode_union_mask(frame_masks, depth.shape)
        highlight_mask = project_points_to_frame(xyz, pose, intrinsic, depth, valid_region, union_mask)
        highlight_rgb = hex_to_rgb(FRAME_HIGHLIGHT_HEX[frame_id])
        xyzrgb = colorize_scene(xyz, highlight_mask, highlight_rgb)

        out_path = args.output_dir / f'{args.scene_id}_frame{frame_id}_allmasks_backproject_xyzrgb.txt'
        save_xyzrgb(out_path, xyzrgb)
        print(
            f'[saved] frame={frame_id} masks={len(frame_masks)} highlighted_points={int(highlight_mask.sum())} path={out_path}'
        )


if __name__ == '__main__':
    main()
