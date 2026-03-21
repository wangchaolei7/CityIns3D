import argparse
import torch
import numpy as np
from pytorch3d.io import IO
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    AmbientLights,
    HardPhongShader,
    BlendParams,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)
from tqdm import tqdm
import pytorch3d
from PIL import Image
import os
import random
from pytorch3d.structures import Pointclouds
import plyfile
import pandas as pd
import glob
import sys
import cv2 # <--- 确保导入cv2

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from utils import mask_lable_location, mask_rasterization

class Snap:
    def __init__(self, image_size, adjust_camera, save_folder, device=None):
        self.image_width, self.image_height = image_size
        self.lift_cam, self.zoomout, self.remove_lip = adjust_camera
        self.save_folder = save_folder
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def read_npy_point_cloud(self, filepath):
        data = np.load(filepath)
        if data.shape[1] < 6:
            print(f"Error: npy文件的点云数据至少需要6列（xyzrgb） in {filepath}")
            return None
        return data[:, :6]
    
    def read_txt_point_cloud(self, filepath):
        # 支持逗号分隔或空白分隔的 txt 点云，假设至少包含 xyzrgb
        try:
            data = np.loadtxt(filepath, delimiter=",")
        except ValueError:
            data = np.loadtxt(filepath)
        if data.shape[1] < 6:
            print(f"Error: txt文件的点云数据至少需要6列（xyzrgb） in {filepath}")
            return None
        return data[:, :6]  # 只返回xyzrgb部分
    
    def get3d_box_from_pcs(self, pc):
        w = pc[:, 0].max() - pc[:, 0].min()
        l = pc[:, 1].max() - pc[:, 1].min()
        h = pc[:, 2].max() - pc[:, 2].min()
        scene_center = np.array([
            pc[:, 0].max() - w / 2,
            pc[:, 1].max() - l / 2,
            pc[:, 2].max() - h / 2,
        ])
        return w, l, h, scene_center

    def _compute_valid_bbox(self, valid_mask):
        height, width = valid_mask.shape
        if not np.any(valid_mask):
            return np.array([0, 0, width, height], dtype=np.int32)

        ys, xs = np.where(valid_mask)
        return np.array(
            [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            dtype=np.int32,
        )

    def render_pcd(
        self,
        pose,
        intrinsic,
        image_width,
        image_height,
        pcd,
        name,
        depth_name,
        valid_mask_name,
        coverage_name,
        meta_name,
    ):
        device = self.device
        intrinsic_matrix = torch.zeros([4, 4], dtype=torch.float32, device=device)
        intrinsic_matrix[3, 3] = 1
        pcd = pcd.to(device=device, dtype=torch.float32)
        point_cloud = Pointclouds(points=[pcd[:, :3]], features=[pcd[:, 3:6]])
        intrinsic_matrix_torch = torch.from_numpy(intrinsic).to(device=device, dtype=torch.float32)
        intrinsic_matrix[:3, :3] = intrinsic_matrix_torch
        camera_to_world = torch.from_numpy(pose).to(device=device, dtype=torch.float32)
        world_to_camera = torch.inverse(camera_to_world)
        fx, fy, cx, cy = (
            intrinsic_matrix[0, 0],
            intrinsic_matrix[1, 1],
            intrinsic_matrix[0, 2],
            intrinsic_matrix[1, 2],
        )
        width, height = image_width, image_height
        rotation_matrix = world_to_camera[:3, :3].permute(1, 0).unsqueeze(0)
        translation_vector = world_to_camera[:3, 3].reshape(-1, 1).permute(1, 0)
        focal_length = -torch.stack([fx, fy], dim=0).reshape(1, 2)
        principal_point = torch.stack([cx, cy], dim=0).reshape(1, 2)
        camera = PerspectiveCameras(focal_length=focal_length,
                                        principal_point=principal_point,
                                        R=rotation_matrix,
                                        T=translation_vector,
                                        image_size=torch.tensor([[height, width]], device=device),
                                        in_ndc=False,
                                        device=device)
            
        raster_settings = PointsRasterizationSettings(
                image_size=(height, width), 
                radius = 0.007,
                points_per_pixel = 10
                )

        rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings) 
        renderer = PointsRenderer(
                rasterizer=rasterizer,
                compositor=AlphaCompositor(background_color = 255)
            )
        rendered_image = renderer(point_cloud)
        rendered_image = rendered_image[0].cpu().numpy().astype(np.uint8)
        color = rendered_image[..., :3]
        color_image = Image.fromarray(color)
        color_image.save(name)

        fragments = rasterizer(point_cloud, cameras=camera)
        zbuf = fragments.zbuf[0, ..., 0].cpu().numpy()
        idx = fragments.idx[0].cpu().numpy()
        valid_mask = np.any(idx >= 0, axis=2) & (zbuf > 0)
        coverage = np.sum(idx >= 0, axis=2).astype(np.uint16)
        valid_bbox = self._compute_valid_bbox(valid_mask)
        valid_pixel_count = int(valid_mask.sum())
        valid_ratio = float(valid_pixel_count) / float(valid_mask.size)

        zbuf[zbuf < 0] = 0
        depth_image_path = depth_name.replace(".npy", ".tif")
        cv2.imwrite(depth_image_path, zbuf)
        cv2.imwrite(valid_mask_name, (valid_mask.astype(np.uint8) * 255))
        cv2.imwrite(coverage_name, coverage)
        np.savez_compressed(
            meta_name,
            valid_bbox=valid_bbox,
            valid_pixel_count=np.int64(valid_pixel_count),
            valid_ratio=np.float32(valid_ratio),
            coverage_max=np.int32(int(coverage.max()) if coverage.size > 0 else 0),
            image_size=np.array([height, width], dtype=np.int32),
        )
        # 彩色深度图
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # np.save(depth_name, zbuf)  # 不保存深度的npy文件
        zbuf_vis = zbuf.copy()
        zbuf_vis[zbuf_vis == 0] = np.nan 
        zbuf_norm = (zbuf_vis - np.nanmin(zbuf_vis)) / (np.nanmax(zbuf_vis) - np.nanmin(zbuf_vis))

        colormap = cm.get_cmap('Spectral_r')  
        color_img = colormap(zbuf_norm)   
        color_img = (color_img[:, :, :3] * 255).astype(np.uint8)  
        Image.fromarray(color_img).save(depth_name.replace(".npy", "_color.png"))


    def scene_image_rendering(self, scan_pc_raw, scene_name, mode = ["global"]):
        self.scene_name = scene_name
        data_type = "pcd"
        
        z_max = scan_pc_raw[:, 2].max()
        idx_remained = scan_pc_raw[:, 2] <= (z_max - self.remove_lip)
        scan_pc_np = scan_pc_raw[idx_remained, :]
        scan_pc = torch.tensor(scan_pc_np, dtype=torch.float32)

        color_folder = os.path.join(self.save_folder, "color")
        depth_folder = os.path.join(self.save_folder, "depth")
        intrinsic_folder = os.path.join(self.save_folder, "intrinsic")
        pose_folder = os.path.join(self.save_folder, "pose")
        valid_mask_folder = os.path.join(self.save_folder, "valid_mask")
        coverage_folder = os.path.join(self.save_folder, "coverage")
        meta_folder = os.path.join(self.save_folder, "meta")
        
        for folder in [
            color_folder,
            depth_folder,
            intrinsic_folder,
            pose_folder,
            valid_mask_folder,
            coverage_folder,
            meta_folder,
        ]:
            os.makedirs(folder, exist_ok=True)

        w_raw, l_raw, h_raw, scene_center = self.get3d_box_from_pcs(scan_pc_raw)
        scale_factor = self.zoomout + 2
        w, l, h = w_raw * scale_factor, l_raw * scale_factor, h_raw * scale_factor

        extrinsic_list = []
        intrinsic_list = []
        generation_functions = {
            "global": self.global_level_camera_generation,
            "corner": self.corner_angle_level_camera_generation,
        }
        for mode_key, func in generation_functions.items():
            if mode_key in mode:
                extrinsic, intrinsic = func(w, l, h, scene_center, scan_pc_raw)
                extrinsic_list.extend(extrinsic)
                intrinsic_list.extend(intrinsic)

        print(f"***** Start to render snap images for {scene_name} ({len(extrinsic_list)} views) *****")

        for i in range(len(extrinsic_list)):
            sys.stdout.write(f"\r- Rendering view {i + 1}/{len(extrinsic_list)}")
            sys.stdout.flush()

            extrinsic = extrinsic_list[i]
            intrinsic = intrinsic_list[i]
            
            # 统一文件名格式
            file_base_name = f"{i:04d}"

            intrinsic_path = os.path.join(intrinsic_folder, f"{file_base_name}.npy")
            pose_path = os.path.join(pose_folder, f"{file_base_name}.npy")
            color_path = os.path.join(color_folder, f"{file_base_name}.png")
            depth_path = os.path.join(depth_folder, f"{file_base_name}.npy") # 基础名，后续会改为.tif
            valid_mask_path = os.path.join(valid_mask_folder, f"{file_base_name}.png")
            coverage_path = os.path.join(coverage_folder, f"{file_base_name}.png")
            meta_path = os.path.join(meta_folder, f"{file_base_name}.npz")

            np.save(intrinsic_path, intrinsic)
            np.save(pose_path, extrinsic)
            
            self.render_pcd(
                extrinsic,
                intrinsic[:3, :3],
                self.image_width,
                self.image_height,
                scan_pc.to(self.device),
                color_path,
                depth_path,
                valid_mask_path,
                coverage_path,
                meta_path,
            )
        print("\n-> Rendering complete.")
        return extrinsic, intrinsic

    # ===================================================================
    # 以下函数保持您提供的原始版本，未做任何修改
    # ===================================================================
    
    def global_level_camera_generation(self, w, l, h, scene_center, scan_pc):
        # 对齐 snap_openins3d.py 的 global 模式：
        # 1. 使用原始场景 bbox，而不是当前文件里额外放大的 +2 版本
        # 2. 使用 num_split=5 生成 16 个外围全局视角
        w_raw, l_raw, h_raw, scene_center_raw = self.get3d_box_from_pcs(scan_pc)
        scale_factor = self.zoomout + 1
        w = w_raw * scale_factor
        l = l_raw * scale_factor
        h = h_raw * scale_factor

        extrinsic_list = []
        intrinsic_list = []
        camera_locations = self.generate_camera_locations(scene_center_raw, w, l, h, 5)
        for camera_location in camera_locations:
            camera_location[-1] += self.lift_cam
            pose_matrix = self.lookat(camera_location, scene_center_raw, np.array([0, 0, -1]))
            pose_to_RBT = np.transpose(np.linalg.inv(np.transpose(pose_matrix)))
            intrinsic_calibrated = self.intrinsic_calibration(
                scan_pc, pose_to_RBT, self.image_width, self.image_height
            )
            extrinsic_list.append(pose_to_RBT)
            intrinsic_list.append(intrinsic_calibrated)
        return extrinsic_list, intrinsic_list

    def _corner_target_positions(self, half_width, half_length, bottom_z, scene_center, num_views=8):
        target_positions = []
        angles = np.linspace(0.0, 2.0 * np.pi, num_views, endpoint=False)
        eps = 1e-8
        for angle in angles:
            dx = float(np.cos(angle))
            dy = float(np.sin(angle))
            tx = half_width / max(abs(dx), eps)
            ty = half_length / max(abs(dy), eps)
            t = min(tx, ty)
            target_positions.append(
                np.array(
                    [
                        scene_center[0] + dx * t,
                        scene_center[1] + dy * t,
                        bottom_z,
                    ],
                    dtype=np.float32,
                )
            )
        return target_positions

    def _select_corner_calibration_points(self, scan_pc, scene_center, target_position, sector_half_angle_deg=75.0):
        pts = np.asarray(scan_pc)
        rel_xy = pts[:, :2] - scene_center[:2]
        dir_xy = target_position[:2] - scene_center[:2]
        dir_norm = np.linalg.norm(dir_xy)
        if dir_norm < 1e-8:
            return pts
        dir_xy = dir_xy / dir_norm

        rel_norm = np.linalg.norm(rel_xy, axis=1)
        keep_center = rel_norm < 1e-3
        valid = rel_norm > 1e-8
        cosine = np.full(rel_norm.shape, -1.0, dtype=np.float32)
        cosine[valid] = (rel_xy[valid] @ dir_xy).astype(np.float32) / rel_norm[valid]
        cosine_thresh = np.cos(np.deg2rad(sector_half_angle_deg))
        mask = keep_center | (cosine >= cosine_thresh)
        selected = pts[mask]
        return selected if selected.shape[0] > 32 else pts

    def corner_angle_level_camera_generation(self, w, l, h, scene_center, scan_pc):
        half_width, half_length, half_height = w / 2, l / 2, h / 2
        top_z = scene_center[2] + half_height
        bottom_z = scene_center[2] - half_height

        camera_location = scene_center.copy()
        corner_lift = getattr(self, "corner_lift_cam", self.lift_cam)
        camera_location[-1] = top_z + corner_lift
        target_positions = self._corner_target_positions(
            half_width=half_width,
            half_length=half_length,
            bottom_z=bottom_z,
            scene_center=scene_center,
            num_views=8,
        )

        extrinsic_list, intrinsic_list = [], []
        for target_position in target_positions:
            pose_matrix = self.lookat(camera_location.copy(), target_position, np.array([0, 0, -1]))
            pose_to_RBT = np.transpose(np.linalg.inv(np.transpose(pose_matrix)))
            calibration_points = self._select_corner_calibration_points(
                scan_pc=scan_pc,
                scene_center=scene_center,
                target_position=target_position,
            )
            intrinsic_calibrated = self.intrinsic_calibration(
                calibration_points, pose_to_RBT, self.image_width, self.image_height
            )
            extrinsic_list.append(pose_to_RBT)
            intrinsic_list.append(intrinsic_calibrated)

        return extrinsic_list, intrinsic_list

    def intrinsic_calibration(self, point_cloud, pose, width, height):
        depth_intrinsic = np.array([
            [577.590698, 0.000000, 318.905426, 0.000000],
            [0.000000, 578.729797, 242.683609, 0.000000],
            [0.000000, 0.000000, 1.000000, 0.000000],
            [0.000000, 0.000000, 0.000000, 1.000000],
        ])
        fx, fy = depth_intrinsic[0, 0], depth_intrinsic[1, 1]
        cx, cy = depth_intrinsic[0, 2], depth_intrinsic[1, 2]
        points = np.hstack([point_cloud[:, :3], np.ones((point_cloud.shape[0], 1))])
        inv_pose = np.linalg.inv(pose.T)
        points_new = points @ inv_pose
        front_mask = points_new[:, 2] > 1e-4
        if np.any(front_mask):
            points_new = points_new[front_mask]
        if points_new.shape[0] == 0:
            return depth_intrinsic.copy()
        point_projected = np.zeros((points_new.shape[0], 2))
        point_projected[:, 0] = points_new[:, 0] * fx / points_new[:, 2] + cx
        point_projected[:, 1] = points_new[:, 1] * fy / points_new[:, 2] + cy
        cx_new = cx - point_projected[:, 0].min()
        cy_new = cy - point_projected[:, 1].min()
        point_projected[:, 0] = points_new[:, 0] * fx / points_new[:, 2] + cx_new
        point_projected[:, 1] = points_new[:, 1] * fy / points_new[:, 2] + cy_new
        scale_1 = width / (point_projected[:, 0].max() + 1e-8) # 避免除以0
        scale_2 = height / (point_projected[:, 1].max() + 1e-8) # 避免除以0
        scale = min(scale_1, scale_2)
        fx_new = depth_intrinsic[0, 0] * scale
        fy_new = depth_intrinsic[1, 1] * scale
        cx_new *= scale
        cy_new *= scale
        new_intrinsic = depth_intrinsic.copy()
        new_intrinsic[0, 0] = fx_new
        new_intrinsic[1, 1] = fy_new
        new_intrinsic[0, 2] = cx_new
        new_intrinsic[1, 2] = cy_new
        return new_intrinsic

    def generate_camera_locations(self, center, width, length, height, num_split=3):
        half_width, half_length, half_height = width / 2, length / 2, height / 2
        top_height = center[2] + half_height
        top_coord = np.linspace(center[0] - half_width, center[0] + half_width, num_split)
        ver_coord = np.linspace(center[1] - half_length, center[1] + half_length, num_split)
        camera_pos_from = []
        for x_coord in top_coord:
            camera_pos_from.append([x_coord, ver_coord[0], top_height])
            camera_pos_from.append([x_coord, ver_coord[-1], top_height])
        for y_coord in ver_coord[1:-1]:
            camera_pos_from.append([top_coord[0], y_coord, top_height])
            camera_pos_from.append([top_coord[-1], y_coord, top_height])
        return camera_pos_from

    def lookat(self, center, target, up):
        f = target - center
        f = f / np.linalg.norm(f)
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)
        u = u / np.linalg.norm(u)
        m = np.zeros((4, 4))
        m[0, :-1] = -s
        m[1, :-1] = u
        m[2, :-1] = f
        m[-1, -1] = 1.0
        t = np.matmul(-m[:3, :3], center)
        m[:3, 3] = t
        return m

def main():
    parser = argparse.ArgumentParser(description="Render STPLS3D multi-view snapshots with optional scene sharding.")
    parser.add_argument(
        "--input-root",
        default="/data1/wangcl/dataset/open_3d/STPLS3D_Open3DIS/3D/stpls3d_render_npy",
    )
    parser.add_argument(
        "--output-root",
        default="/data1/wangcl/dataset/open_3d/STPLS3D_Open3DIS/2D",
    )
    parser.add_argument("--image-width", type=int, default=2000)
    parser.add_argument("--image-height", type=int, default=2000)
    parser.add_argument("--lift-cam", type=float, default=3.0)
    parser.add_argument(
        "--corner-lift-cam",
        type=float,
        default=None,
        help="Extra z offset used only by corner mode. Defaults to --lift-cam if not set.",
    )
    parser.add_argument("--zoomout", type=float, default=1)
    parser.add_argument("--remove-lip", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--scene", action="append", default=None, help="Only render the specified scene id. Repeatable.")
    parser.add_argument("--scene-list", default=None, help="Text file with one scene id per line.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["global"],
        help="Rendering modes. Current file supports global and corner.",
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "npy", "txt"],
        default="auto",
        help="Input point cloud format. auto searches txt first, then npy.",
    )
    args = parser.parse_args()

    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError("--worker-id must satisfy 0 <= worker-id < num-workers")
    supported_modes = {"global", "corner"}
    unsupported_modes = [mode for mode in args.modes if mode not in supported_modes]
    if unsupported_modes:
        print(f"警告: snap_open3dis_ring_3h.py 当前仅支持 {sorted(supported_modes)}，以下模式将被忽略: {unsupported_modes}")
        args.modes = [mode for mode in args.modes if mode in supported_modes] or ["global"]

    adjust_camera = [args.lift_cam, args.zoomout, args.remove_lip]
    image_size = [args.image_width, args.image_height]

    if args.input_format == "txt":
        input_files = sorted(glob.glob(os.path.join(args.input_root, "*_points_GTv3_*.txt")))
    elif args.input_format == "npy":
        input_files = sorted(glob.glob(os.path.join(args.input_root, "*_points_GTv3_*.npy")))
    else:
        input_files = sorted(glob.glob(os.path.join(args.input_root, "*_points_GTv3_*.txt")))
        if not input_files:
            input_files = sorted(glob.glob(os.path.join(args.input_root, "*_points_GTv3_*.npy")))

    if not input_files:
        print(f"错误: 未在 '{args.input_root}' 中找到匹配的 txt/npy 场景文件。")
        return

    selected_scenes = set(args.scene or [])
    if args.scene_list:
        with open(args.scene_list, "r", encoding="utf-8") as f:
            selected_scenes.update(line.strip() for line in f if line.strip())

    if selected_scenes:
        input_files = [
            path for path in input_files
            if os.path.splitext(os.path.basename(path))[0] in selected_scenes
        ]

    if not input_files:
        print("错误: 经过 scene 过滤后没有待处理场景。")
        return

    sharded_files = input_files[args.worker_id::args.num_workers]
    if not sharded_files:
        print(
            f"worker {args.worker_id}/{args.num_workers} 没有分到场景。"
        )
        return

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"总场景数={len(input_files)} 当前worker场景数={len(sharded_files)} "
        f"worker={args.worker_id}/{args.num_workers} device={device_name} "
        f"modes={args.modes} input_format={args.input_format}"
    )

    for input_path in tqdm(sharded_files, desc=f"worker{args.worker_id}"):
        scene_name = os.path.splitext(os.path.basename(input_path))[0]
        tqdm.write(f"\n--- 开始处理场景: {scene_name} ---")

        scene_output_dir = os.path.join(args.output_root, scene_name)
        
        snap_module = Snap(image_size, adjust_camera, scene_output_dir, device=device_name)
        snap_module.corner_lift_cam = args.corner_lift_cam if args.corner_lift_cam is not None else args.lift_cam

        if input_path.endswith(".txt"):
            pcd_rgb = snap_module.read_txt_point_cloud(input_path)
        else:
            pcd_rgb = snap_module.read_npy_point_cloud(input_path)
        if pcd_rgb is None:
            tqdm.write(f"!! 跳过场景 {scene_name} 因为点云加载失败。")
            continue
        
        snap_module.scene_image_rendering(pcd_rgb[:, :6], scene_name, mode=args.modes)

    tqdm.write("\n\n所有场景处理完毕！")

if __name__ == "__main__":
    main()
