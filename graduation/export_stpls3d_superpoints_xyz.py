#!/usr/bin/env python3
"""
功能说明：
1. 读取 STPLS3D 的超点文件（.pth），该文件存储每个点对应的 superpoint id。
2. 自动匹配同名场景的点云 txt，并从中提取 xyz 坐标。
3. 按要求导出一份 xyz 文本文件。
4. 额外导出一份基于 vivid 调色盘的 xyzrgb 文本，便于直接可视化超点分布。

默认输入超点文件：
/data1/wangcl/dataset/open_3d/STPLS3D_Open3DIS/3D/tgc_superpoints_utonia/05_points_GTv3_83.pth

默认坐标来源：
/data1/wangcl/dataset/open_3d/STPLS3D_Open3DIS/3D/groundtruth/05_points_GTv3_83.txt

默认输出目录：
/data1/wangcl/project/CityIns3D/graduation

说明：
- 需要在带 torch 的 Python 环境中运行，例如 SSP 环境。
- xyz 主输出只包含 3 列坐标。
- vivid 可视化输出包含 6 列：xyzrgb。
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import numpy as np
import torch


DEFAULT_SPP_PATH = Path(
    '/data1/wangcl/dataset/open_3d/STPLS3D_Open3DIS/3D/tgc_superpoints_utonia/05_points_GTv3_83.pth'
)
DEFAULT_XYZ_ROOT = Path('/data1/wangcl/dataset/open_3d/STPLS3D_Open3DIS/3D/groundtruth')
DEFAULT_OUTPUT_DIR = Path('/data1/wangcl/project/CityIns3D/graduation')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export STPLS3D superpoint tensor to xyz txt and vivid xyzrgb txt.'
    )
    parser.add_argument('--input', type=Path, default=DEFAULT_SPP_PATH, help='Input superpoint .pth file.')
    parser.add_argument('--xyz-root', type=Path, default=DEFAULT_XYZ_ROOT, help='Root directory of matching STPLS3D txt scenes.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Directory to save exported txt files.')
    parser.add_argument('--save-vivid-xyzrgb', action='store_true', default=True, help='Also save a vivid-colored xyzrgb companion file.')
    return parser.parse_args()


def golden_hues(n: int, offset: float = 0.0) -> np.ndarray:
    if n <= 0:
        return np.empty((0,), dtype=np.float64)
    golden_ratio = 0.6180339887498949
    return (offset + np.arange(n, dtype=np.float64) * golden_ratio) % 1.0


def vivid_colors(n: int) -> np.ndarray:
    hues = golden_hues(n)
    sat = np.full(n, 0.82)
    val = np.full(n, 0.96)
    rgb = np.zeros((n, 3), dtype=np.uint8)
    for idx, (h, s, v) in enumerate(zip(hues, sat, val)):
        r, g, b = colorsys.hsv_to_rgb(float(h), float(s), float(v))
        rgb[idx] = np.round(np.array([r, g, b]) * 255.0).astype(np.uint8)
    return rgb


def resolve_xyz_path(spp_path: Path, xyz_root: Path) -> Path:
    candidate = xyz_root / f'{spp_path.stem}.txt'
    if not candidate.exists():
        raise FileNotFoundError(f'Matching xyz txt not found: {candidate}')
    return candidate


def load_xyz(txt_path: Path) -> np.ndarray:
    data = np.loadtxt(txt_path, delimiter=',', dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f'Expected at least 3 columns in {txt_path}, got {data.shape}')
    return data[:, :3]


def load_superpoints(spp_path: Path) -> np.ndarray:
    spp = torch.load(spp_path, map_location='cpu', weights_only=False)
    if not isinstance(spp, torch.Tensor):
        raise TypeError(f'Expected torch.Tensor in {spp_path}, got {type(spp)}')
    if spp.ndim != 1:
        raise ValueError(f'Expected 1D superpoint tensor, got shape {tuple(spp.shape)}')
    return spp.cpu().numpy().astype(np.int64, copy=False)


def build_vivid_xyzrgb(xyz: np.ndarray, spp_ids: np.ndarray) -> np.ndarray:
    unique_ids = np.unique(spp_ids)
    colors = vivid_colors(len(unique_ids))
    rgb = np.zeros((xyz.shape[0], 3), dtype=np.uint8)
    for color, sp_id in zip(colors, unique_ids.tolist()):
        rgb[spp_ids == sp_id] = color
    return np.concatenate([xyz, rgb.astype(np.float64)], axis=1)


def save_txt(path: Path, array: np.ndarray, fmt: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, array, fmt=fmt, delimiter=',')


def main() -> None:
    args = parse_args()
    xyz_path = resolve_xyz_path(args.input, args.xyz_root)
    xyz = load_xyz(xyz_path)
    spp_ids = load_superpoints(args.input)

    if xyz.shape[0] != spp_ids.shape[0]:
        raise ValueError(
            f'Point count mismatch: xyz has {xyz.shape[0]} rows, superpoint tensor has {spp_ids.shape[0]} entries.'
        )

    stem = args.input.stem
    xyz_output = args.output_dir / f'{stem}_superpoints_xyz.txt'
    save_txt(xyz_output, xyz, fmt=['%.6f', '%.6f', '%.6f'])
    print(f'[saved] {xyz_output}')

    if args.save_vivid_xyzrgb:
        xyzrgb = build_vivid_xyzrgb(xyz, spp_ids)
        xyzrgb_output = args.output_dir / f'{stem}_superpoints_vivid_xyzrgb.txt'
        save_txt(xyzrgb_output, xyzrgb, fmt=['%.6f', '%.6f', '%.6f', '%.0f', '%.0f', '%.0f'])
        print(f'[saved] {xyzrgb_output}')


if __name__ == '__main__':
    main()
