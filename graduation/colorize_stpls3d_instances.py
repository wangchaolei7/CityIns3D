#!/usr/bin/env python3
"""
功能说明：
1. 读取 STPLS3D 的 txt 点云文件，默认格式为 xyzrgbsemanticinstance。
2. 按实例 ID 对点云重新赋色，不同实例使用不同颜色。
3. 内置多种调色盘色系，可一次性导出多份 xyzrgb 文本结果。
4. 输出文件只保留 xyzrgb 六列，采用逗号分隔，方便后续可视化或导入其他工具。

默认输入：
/data1/wangcl/dataset/open_3d/STPLS3D_Open3DIS/3D/Synthetic_v3_InstanceSegmentation/5_points_GTv3.txt

默认输出目录：
/data1/wangcl/project/CityIns3D/graduation

说明：
- 输入文件默认第 1~3 列为 xyz，第 8 列为 instance id（0-based 下标分别为 0:3 和 7）。
- instance id < 0 的点视为 ignore/background，统一赋为暖灰卡其色。
- 同一个实例在同一种调色盘下会获得稳定一致的颜色。
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import numpy as np


DEFAULT_INPUT = Path(
    "/data1/wangcl/dataset/open_3d/STPLS3D_Open3DIS/3D/Synthetic_v3_InstanceSegmentation/5_points_GTv3.txt"
)
DEFAULT_OUTPUT_DIR = Path("/data1/wangcl/project/CityIns3D/graduation")
DEFAULT_PALETTES = ["vivid", "pastel", "earth", "ocean", "sunset"]
IGNORE_COLOR = np.array([178, 168, 146], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Colorize STPLS3D instances with several palette styles and export xyzrgb txt files."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input STPLS3D txt file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for xyzrgb txt files.")
    parser.add_argument(
        "--palettes",
        nargs="+",
        default=DEFAULT_PALETTES,
        choices=DEFAULT_PALETTES,
        help="Palette styles to export.",
    )
    parser.add_argument(
        "--instance-col",
        type=int,
        default=7,
        help="0-based column index for instance id in the input txt.",
    )
    return parser.parse_args()


def load_scene(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] <= 7:
        raise ValueError(f"Expected at least 8 columns (xyzrgbsemanticinstance), got shape {data.shape}.")
    return data


def golden_hues(n: int, offset: float = 0.0) -> np.ndarray:
    if n <= 0:
        return np.empty((0,), dtype=np.float64)
    golden_ratio = 0.6180339887498949
    hues = (offset + np.arange(n, dtype=np.float64) * golden_ratio) % 1.0
    return hues


def palette_colors(n: int, palette_name: str) -> np.ndarray:
    hues = golden_hues(n)

    if palette_name == "vivid":
        sat = np.full(n, 0.82)
        val = np.full(n, 0.96)
    elif palette_name == "pastel":
        sat = np.full(n, 0.38)
        val = np.full(n, 0.98)
    elif palette_name == "earth":
        hues = (0.08 + hues * 0.28) % 1.0
        sat = np.full(n, 0.60)
        val = np.linspace(0.52, 0.86, n, endpoint=True) if n > 1 else np.array([0.70])
    elif palette_name == "ocean":
        hues = (0.48 + hues * 0.22) % 1.0
        sat = np.full(n, 0.68)
        val = np.linspace(0.62, 0.96, n, endpoint=True) if n > 1 else np.array([0.82])
    elif palette_name == "sunset":
        hues = (0.92 + hues * 0.16) % 1.0
        sat = np.full(n, 0.72)
        val = np.linspace(0.72, 0.98, n, endpoint=True) if n > 1 else np.array([0.88])
    else:
        raise ValueError(f"Unsupported palette: {palette_name}")

    rgb = np.zeros((n, 3), dtype=np.uint8)
    for idx, (h, s, v) in enumerate(zip(hues, sat, val)):
        r, g, b = colorsys.hsv_to_rgb(float(h), float(s), float(v))
        rgb[idx] = np.round(np.array([r, g, b]) * 255.0).astype(np.uint8)
    return rgb


def colorize_instances(data: np.ndarray, instance_col: int, palette_name: str) -> np.ndarray:
    xyz = data[:, :3].astype(np.float64, copy=False)
    instance_ids = data[:, instance_col].astype(np.int64, copy=False)

    rgb = np.repeat(IGNORE_COLOR[None, :], repeats=data.shape[0], axis=0)
    valid_mask = instance_ids >= 0
    valid_instance_ids = np.unique(instance_ids[valid_mask])
    colors = palette_colors(len(valid_instance_ids), palette_name)

    for color, instance_id in zip(colors, valid_instance_ids.tolist()):
        rgb[instance_ids == instance_id] = color

    return np.concatenate([xyz, rgb.astype(np.float64)], axis=1)


def save_xyzrgb(path: Path, xyzrgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        xyzrgb,
        fmt=["%.6f", "%.6f", "%.6f", "%.0f", "%.0f", "%.0f"],
        delimiter=",",
    )


def main() -> None:
    args = parse_args()
    data = load_scene(args.input)
    stem = args.input.stem

    print(f"[info] input={args.input}")
    print(f"[info] points={data.shape[0]}")
    print(f"[info] output_dir={args.output_dir}")

    for palette_name in args.palettes:
        xyzrgb = colorize_instances(data, instance_col=args.instance_col, palette_name=palette_name)
        output_path = args.output_dir / f"{stem}_instance_palette_{palette_name}.txt"
        save_xyzrgb(output_path, xyzrgb)
        print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
