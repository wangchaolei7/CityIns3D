import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import pyviz3d.visualizer as viz
from plyfile import PlyData
from os.path import join
import open3d as o3d
from munch import Munch
from open3dis.dataset_outdoor.stpls3d_io import (
    load_pointcloud_xyz_rgb,
    load_xyz_semantic_instance,
    resolve_scene_path,
)

# Global visualization controls.
# Edit these values directly to change default behavior.
POINT_SIZE = 8
POINT_SIZE_MODE = "fixed"  # "fixed" or "attenuated"
POINT_SIZE_DISTANCE_SCALE = 200.0
SHOW_PCL_BY_DEFAULT = True
MANUAL_SHOW_SPP = True
MANUAL_SHOW_LIFTED = False
MANUAL_SHOW_FINAL = False
MANUAL_SHOW_GT = True
MANUAL_SHOW_FEATURE = True

def generate_palette(n):
    palette = []
    for _ in range(n):
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        palette.append((red, green, blue))
    return palette

def rle_decode(rle):
    length = rle["length"]
    s = rle["counts"]

    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask


def decode_rle_masks(mask_pack):
    masks = mask_pack["ins"]
    if isinstance(masks, torch.Tensor):
        if masks.ndim == 1:
            masks = masks.unsqueeze(0)
        return masks.to(torch.bool)
    if len(masks) == 0:
        return torch.empty((0, mask_pack.get("length", 0)), dtype=torch.bool)
    return torch.stack([torch.tensor(rle_decode(ins), dtype=torch.bool) for ins in masks], dim=0)


def torch_load_local(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_feature_tensor(feature_path):
    data = torch_load_local(feature_path, map_location="cpu")
    if isinstance(data, dict):
        feat = data.get("feat", data.get("spp_feat"))
    else:
        feat = data
    if feat is None:
        raise ValueError(f"Feature file does not contain usable features: {feature_path}")
    if not isinstance(feat, torch.Tensor):
        feat = torch.as_tensor(feat)
    if feat.ndim != 2:
        raise ValueError(f"Expected 2D feature tensor, got {tuple(feat.shape)} from {feature_path}")
    return feat.to(torch.float32).cpu()

def read_pointcloud(pcd_path):
    return load_pointcloud_xyz_rgb(pcd_path)


def patch_saved_viewer(output_dir):
    """Patch the generated PyViz3D shader so point size is actually controllable."""
    index_path = Path(output_dir) / "index.html"
    if not index_path.exists():
        return
    text = index_path.read_text(encoding="utf-8")
    old = "gl_PointSize = pointSize / length(vertex_position_camera);"
    if POINT_SIZE_MODE == "fixed":
        new = "gl_PointSize = max(pointSize, 1.0);"
    else:
        new = (
            "gl_PointSize = max(pointSize * "
            f"{float(POINT_SIZE_DISTANCE_SCALE):.6f}"
            " / max(length(vertex_position_camera), 1e-6), 1.0);"
        )
    if old in text:
        text = text.replace(old, new)
    elif "gl_PointSize = max(pointSize, 1.0);" in text or "gl_PointSize = max(pointSize *" in text:
        # Already patched by a previous run; rewrite to the current mode.
        import re

        text = re.sub(
            r"gl_PointSize = .*?;",
            new,
            text,
            count=1,
        )
    index_path.write_text(text, encoding="utf-8")

SCANNET200 = 'wall.floor.cabinet.bed.chair.sofa.table.door.window.bookshelf.picture.counter.desk.curtrain.refridgerator.shower_toilet.sink.bathtub.otherfurniture'
class_names = SCANNET200.split('.')

class VisualizationScannet200:
    def __init__(self, point, color):
        self.point = point.astype(np.float32)
        self.color = color
        self.point_size = POINT_SIZE
        self.vis = viz.Visualizer()
        self._center_point_cloud()
        self.vis.add_points(
            'pcl',
            self.point,
            self.color.astype(np.float32),
            point_size=self.point_size,
            visible=SHOW_PCL_BY_DEFAULT,
        )

    def _center_point_cloud(self):
        """让点云以几何中心为原点，便于浏览器初始视角对准十字中心。"""
        if self.point.size == 0:
            return
        centroid = self.point.mean(axis=0, keepdims=True)
        self.point = self.point - centroid
    
    def save(self, path):
        self.vis.save(path)
        patch_saved_viewer(path)
    
    def superpointviz(self, spp_path):
        print('...Visualizing Superpoints...')
        spp = torch_load_local(spp_path, map_location='cpu')
        if isinstance(spp, np.ndarray):
            spp = torch.from_numpy(spp)
        spp = spp.to(dtype=torch.int64)
        unique_spp, spp, num_point = torch.unique(spp, return_inverse=True, return_counts=True)
        n_spp = unique_spp.shape[0]
        pallete =  generate_palette(n_spp + 1)
        uniqueness = torch.unique(spp).clone()
        tt_col = self.color.copy()
        for idx in range(uniqueness.shape[0]):
            label_value = int(uniqueness[idx].item())
            if label_value < 0:
                continue
            ss = torch.where(spp == label_value)[0]
            color_idx = label_value % len(pallete)
            for ind in ss.tolist():
                tt_col[ind, :] = pallete[color_idx]
        self.vis.add_points(
            f'superpoint: {n_spp}',
            self.point,
            tt_col,
            point_size=self.point_size,
            visible=True,
        )
        print('---Done---')
    
    def gtviz(self, gt_data, specific = False):
        print('...Visualizing Groundtruth...')
        if gt_data.endswith('.ply') or gt_data.endswith('.txt'):
            _, sem_label, ins_label = load_xyz_semantic_instance(gt_data)
        else:
            gt_pack = torch_load_local(gt_data, map_location='cpu')
            if isinstance(gt_pack, (list, tuple)) and len(gt_pack) == 4:
                _, _, sem_label, ins_label = gt_pack
            else:
                raise ValueError(f"Unsupported GT format: {gt_data}")
            if isinstance(sem_label, torch.Tensor):
                sem_label = sem_label.cpu().numpy()
            if isinstance(ins_label, torch.Tensor):
                ins_label = ins_label.cpu().numpy()
        pallete =  generate_palette(int(2e3 + 1))
        n_label = np.unique(ins_label)
        tt_col = self.color.copy()
        for i in range(0, n_label.shape[0]):
            sem_value = sem_label[np.where(ins_label==n_label[i])][0]
            # if sem_value in (0, 15, 17, 18, 19):  # 忽略地面
            if sem_value in (0, 15, 18, 19):  # 忽略地面
                continue
            tt_col[np.where(ins_label==n_label[i])] = pallete[i]
            if specific: # be more specific
                tt_col_specific = self.color.copy()
                tt_col_specific[np.where(ins_label==n_label[i])] = pallete[i]
                self.vis.add_points(
                    f'GT instance: {i}_{sem_value}',
                    self.point,
                    tt_col_specific,
                    point_size=self.point_size,
                    visible=True,
                )

        self.vis.add_points(
            f'GT instance: ' + str(i),
            self.point,
            tt_col,
            point_size=self.point_size,
            visible=True,
        )
        print('---Done---')

    def vizmask3d(self, mask3d_path, specific = False):
        print('...Visualizing 3D backbone mask...')
        dic = torch_load_local(mask3d_path, map_location='cpu')
        instance = decode_rle_masks(dic)
        conf3d = dic['conf']
        if instance.shape[0] == 0:
            self.vis.add_points(
                '3D backbone mask: 0',
                self.point,
                self.color.copy(),
                point_size=self.point_size,
                visible=True,
            )
            print('---Done (empty)---')
            return
        pallete =  generate_palette(int(2e3 + 1))
        tt_col = self.color.copy()
        limit = 10
        for i in range(0, instance.shape[0]):
            tt_col[instance[i] == 1] = pallete[i]
            if specific and limit > 0: # be more specific but limit 10 masks (avoiding lag)
                limit -= 1
                tt_col_specific = self.color.copy()
                tt_col_specific[instance[i] == 1] = pallete[i]
                self.vis.add_points(
                    f'3D backbone mask: ' + str(i) + '_' + str(conf3d[i]),
                    self.point,
                    tt_col_specific,
                    point_size=self.point_size,
                    visible=True,
                )

        self.vis.add_points(
            f'3D backbone mask: ' + str(i),
            self.point,
            tt_col,
            point_size=self.point_size,
            visible=True,
        )
        print('---Done---')

    def vizmask2d(self, mask2d_path, specific = False):
        print('...Visualizing 2D lifted mask...')
        dic = torch_load_local(mask2d_path, map_location='cpu')
        instance = decode_rle_masks(dic)
        conf2d = dic['conf'] # confidence really doesn't affect much (large mask -> small conf)
        if instance.shape[0] == 0:
            self.vis.add_points(
                '2D lifted mask: 0',
                self.point,
                self.color.copy(),
                point_size=self.point_size,
                visible=True,
            )
            print('---Done (empty)---')
            return
        pallete =  generate_palette(int(5e3 + 1))
        tt_col = self.color.copy()
        limit = 10
        for i in range(0, instance.shape[0]):
            tt_col[instance[i] == 1] = pallete[i]
            if specific and limit > 0: # be more specific but limit 10 masks (avoiding lag)
                limit -= 1
                tt_col_specific = self.color.copy()
                tt_col_specific[instance[i] == 1] = pallete[i]
                self.vis.add_points(
                    f'2D lifted mask: ' + str(i) + '_' + str(conf2d[i].item())[:5],
                    self.point,
                    tt_col_specific,
                    point_size=self.point_size,
                    visible=True,
                )

        self.vis.add_points(
            f'2D lifted mask: ' + str(i),
            self.point,
            tt_col,
            point_size=self.point_size,
            visible=True,
        )
        print('---Done---')        
        
    def finalviz(self, agnostic_path, specific = False, vocab = False):
        print('...Visualizing final class agnostic mask...')
        dic = torch_load_local(agnostic_path, map_location='cpu')
        instance = decode_rle_masks(dic)
        conf2d = dic['conf'] # confidence really doesn't affect much (large mask -> small conf)
        if instance.shape[0] == 0:
            self.vis.add_points(
                'final mask: 0',
                self.point,
                self.color.copy(),
                point_size=self.point_size,
                visible=True,
            )
            print('---Done (empty)---')
            return

        if vocab == True:
            label = dic['final_class']
        pallete =  generate_palette(int(2e3 + 1))
        tt_col = self.color.copy()
        limit = 5
        for i in range(0, instance.shape[0]):
            tt_col[instance[i] == 1] = pallete[i]
            if specific and limit > 0: # be more specific but limit 10 masks (avoiding lag)
                limit -= 1
                tt_col_specific = self.color.copy()
                tt_col_specific[instance[i] == 1] = pallete[i]
                if vocab == True:
                    self.vis.add_points(
                        f'final mask: ' + str(i) + '_' + class_names[label[i]],
                        self.point,
                        tt_col_specific,
                        point_size=self.point_size,
                        visible=True,
                    )
                else:
                    self.vis.add_points(
                        f'final mask: ' + str(i) + '_' + str(conf2d[i].item())[:5],
                        self.point,
                        tt_col_specific,
                        point_size=self.point_size,
                        visible=True,
                    )

        self.vis.add_points(
            f'final mask: ' + str(i),
            self.point,
            tt_col,
            point_size=self.point_size,
            visible=True,
        )
        print('---Done---')  

    def featureviz(self, feature_path):
        print('...Visualizing point features...')
        feat = load_feature_tensor(feature_path)
        if feat.shape[0] != self.point.shape[0]:
            raise ValueError(
                f"Feature/point count mismatch: feat={feat.shape[0]} point={self.point.shape[0]}"
            )

        feat = F.normalize(feat, dim=1, p=2)
        if feat.shape[1] >= 3 and feat.shape[0] >= 3:
            q = min(6, feat.shape[1], feat.shape[0] - 1)
            u, s, _ = torch.pca_lowrank(feat, q=q, center=True)
            rgb = (u[:, :3] * s[:3]).cpu().numpy()
        else:
            rgb = feat[:, : min(3, feat.shape[1])].cpu().numpy()
            if rgb.shape[1] < 3:
                rgb = np.pad(rgb, ((0, 0), (0, 3 - rgb.shape[1])), mode="constant")

        rgb_min = rgb.min(axis=0, keepdims=True)
        rgb_max = rgb.max(axis=0, keepdims=True)
        denom = np.maximum(rgb_max - rgb_min, 1e-6)
        rgb = (rgb - rgb_min) / denom
        rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.float32)

        self.vis.add_points(
            'point feature (PCA)',
            self.point,
            rgb,
            point_size=self.point_size,
            visible=True,
        )
        print('---Done---')  

def get_parser():
    parser = argparse.ArgumentParser(description="Visualize STPLS3D superpoints and masks.")
    parser.add_argument("--config", type=str, default="configs/stpls3d.yaml", help="Path to config yaml.")
    parser.add_argument("--scene", type=str, required=True, help="Scene id to visualize.")
    parser.add_argument("--spp-path", type=str, default=None, help="Optional explicit superpoint label path.")
    parser.add_argument("--lifted-path", type=str, default=None, help="Optional explicit 2D-lifted proposal path.")
    parser.add_argument("--final-path", type=str, default=None, help="Optional explicit final proposal path.")
    parser.add_argument("--ply-root", type=str, default=None, help="Optional explicit original ply directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="PyViz3D output directory.")
    parser.add_argument("--show-spp", action="store_true", help="Visualize superpoint labels.")
    parser.add_argument("--show-lifted", action="store_true", help="Visualize 2D-lifted 3D proposals.")
    parser.add_argument("--show-final", action="store_true", help="Visualize final merged masks.")
    parser.add_argument("--show-feature", action="store_true", help="Visualize point-level features with PCA colors.")
    parser.add_argument("--show-gt", action="store_true", help="Also visualize GT instances.")
    parser.add_argument("--gt-path", type=str, default=None, help="Optional explicit GT file path.")
    parser.add_argument("--feature-path", type=str, default=None, help="Optional explicit feature file path.")
    parser.add_argument("--specific-lifted", action="store_true", help="Show individual 2D-lifted proposals as separate layers.")
    parser.add_argument("--specific-final", action="store_true", help="Show individual final proposals as separate layers.")
    return parser


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return Munch.fromDict(yaml.safe_load(handle.read()))


def resolve_superpoint_path(cfg, scene_id, explicit_path=None):
    if explicit_path is not None:
        return explicit_path
    spp_root = None
    # if hasattr(cfg, "superpoint"):
    #     spp_root = getattr(cfg.superpoint, "save_dir", None)
    #     if spp_root is None:
    #         spp_root = getattr(cfg.superpoint, "label_dir", None)
    if spp_root is None:
        spp_root = cfg.data.spp_path
    pth_path = os.path.join(spp_root, f"{scene_id}.pth")
    if os.path.exists(pth_path):
        return pth_path
    return os.path.join(spp_root, f"{scene_id}.pt")


def resolve_gt_path(cfg, scene_id, explicit_path=None):
    if explicit_path is not None:
        return explicit_path
    gt_root = getattr(cfg.data, "gt_pth", None)
    if gt_root is None:
        return None
    try:
        return resolve_scene_path(gt_root, scene_id)
    except FileNotFoundError:
        pass
    pth_path = os.path.join(gt_root, f"{scene_id}.pth")
    if os.path.exists(pth_path):
        return pth_path
    return None


def resolve_cluster_output_path(cfg, scene_id, explicit_path=None):
    if explicit_path is not None:
        return explicit_path
    # output_root = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
    output_root = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output_lifted_part)
    return os.path.join(output_root, f"{scene_id}.pth")


def resolve_final_output_path(cfg, scene_id, explicit_path=None):
    if explicit_path is not None:
        return explicit_path
    output_root = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.final_output)
    return os.path.join(output_root, f"{scene_id}.pth")


def resolve_feature_path(cfg, scene_id, explicit_path=None):
    if explicit_path is not None:
        return explicit_path
    feature_root = getattr(cfg.data, "point_features_path", None)
    if feature_root is None:
        return None
    feature_path = os.path.join(feature_root, f"{scene_id}.pth")
    return feature_path if os.path.exists(feature_path) else None


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = load_config(args.config)

    scene_id = args.scene
    spp_path = resolve_superpoint_path(cfg, scene_id, args.spp_path)
    lifted_path = resolve_cluster_output_path(cfg, scene_id, args.lifted_path)
    final_path = resolve_final_output_path(cfg, scene_id, args.final_path)
    feature_path = resolve_feature_path(cfg, scene_id, args.feature_path)
    ply_root = args.ply_root or cfg.data.original_ply
    gt_path = resolve_gt_path(cfg, scene_id, args.gt_path)
    pyviz3d_dir = args.output_dir or os.path.join("viz", scene_id)

    point_path = resolve_scene_path(ply_root, scene_id)
    point, color = read_pointcloud(point_path)
    color = color * 127.5

    VIZ = VisualizationScannet200(point, color)

    if args.show_spp or MANUAL_SHOW_SPP:
        VIZ.superpointviz(spp_path)
    if args.show_lifted or MANUAL_SHOW_LIFTED:
        if not os.path.exists(lifted_path):
            raise FileNotFoundError(f"Cannot resolve 2D-lifted path for scene {scene_id}: {lifted_path}")
        VIZ.vizmask2d(lifted_path, specific=args.specific_lifted)
    if args.show_final or MANUAL_SHOW_FINAL:
        if not os.path.exists(final_path):
            raise FileNotFoundError(f"Cannot resolve final mask path for scene {scene_id}: {final_path}")
        VIZ.finalviz(final_path, specific=args.specific_final)
    if args.show_gt or MANUAL_SHOW_GT:
        if gt_path is None:
            raise FileNotFoundError(f"Cannot resolve GT path for scene {scene_id}")
        VIZ.gtviz(gt_path, specific=False)
    if args.show_feature or MANUAL_SHOW_FEATURE:
        if feature_path is None:
            raise FileNotFoundError(f"Cannot resolve feature path for scene {scene_id}")
        VIZ.featureviz(feature_path)
    VIZ.save(pyviz3d_dir)
