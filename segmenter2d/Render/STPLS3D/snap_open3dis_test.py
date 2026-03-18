"""
SNAP (Synthetic Navigation and Acquisition of Perspectives) 点云多视角渲染模块

版本: 2.0 (点云专用批处理版)
作者: 基于 Zhening Huang 的原始代码修改

功能描述:
  本模块专为批量处理3D点云数据设计，能够自动为每个点云场景生成多视角的2D渲染图像、
  深度图、相机内参和外参矩阵。输出格式标准化，适用于计算机视觉和深度学习任务。

主要特性:
  - 专为点云数据优化，移除网格相关功能
  - 支持批量处理多个场景，自动遍历输入目录
  - 生成标准化的颜色图像、深度图(TIFF格式)和相机参数
  - 提供彩色深度可视化图像，便于直观检查
  - 使用tqdm进度条，提供友好的处理进度反馈

输入格式:
  - 点云数据: .npy 或 .txt 格式，每点包含至少6个属性 (x, y, z, r, g, b)

输出结构:
  对于每个场景，创建以下子目录:
    color/      - 渲染的颜色图像 (PNG格式)
    depth/      - 深度图 (TIFF格式) 和彩色可视化深度图
    intrinsic/  - 相机内参矩阵 (NPY格式)
    pose/       - 相机外参(姿态)矩阵 (NPY格式)

关键参数:
  - image_size: 输出图像尺寸 [宽度, 高度]
  - adjust_camera: [lift_cam, zoomout, remove_lip]
    * lift_cam: 相机抬升高度，避免与场景碰撞
    * zoomout: 缩放系数，控制相机距离场景中心的远近
    * remove_lip: 移除顶部点的距离，用于裁剪天花板
  - save_folder: 输出根目录

使用示例:
  参见 main() 函数中的批量处理流程

注意:
  - 需要安装 pytorch3d, opencv-python, tqdm 等依赖库
  - 深度图使用TIFF格式保存以保证精度，同时提供彩色可视化版本
  - 颜色值会自动归一化(除以255)以适应渲染器的输入要求

修改记录:
  - 移除网格渲染功能，专注于点云处理
  - 添加批量处理功能和进度显示
  - 改进深度图输出格式和质量
  - 规范化输出文件命名和目录结构

修改记录: 相比于snap_open3dis.py
    - 2025-09-22: 添加多种视角生成模式, lift_cam高度升到20m
"""
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

os.environ['CUDA_VISIBLE_DEVICES']='0'

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from utils import mask_lable_location, mask_rasterization

class Snap:
    def __init__(self, image_size, adjust_camera, save_folder):
        self.image_width, self.image_height = image_size
        self.lift_cam, self.zoomout, self.remove_lip = adjust_camera
        self.save_folder = save_folder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bird_lift_cam = 5.0  # 默认鸟瞰高度，可在渲染时覆盖
        self.corner_lift_cam = 20.0  # corner角点视角单独抬升高度

    def read_npy_point_cloud(self, filepath):
        data = np.load(filepath)
        if data.shape[1] < 6:
            print(f"Error: npy文件的点云数据至少需要6列（xyzrgb） in {filepath}")
            return None
        return data[:, :6]
    
    def read_txt_point_cloud(self, filepath):
        # 读取txt格式的点云，假设每行格式为 x y z r g b semantic
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

    def render_pcd(self, pose, intrinsic, image_width, image_height, pcd, name, depth_name):
        device = self.device
        intrinsic_matrix = torch.zeros([4, 4])
        intrinsic_matrix[3, 3] = 1
        point_cloud = Pointclouds(points=[pcd[:,:3]], features=[pcd[:,3:6]/255.0]) # 颜色值需归一化
        intrinsic_matrix_torch = torch.from_numpy(intrinsic)
        intrinsic_matrix[:3, :3] = intrinsic_matrix_torch
        camera_to_world = torch.from_numpy(pose)
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
        focal_length = -torch.tensor([[fx, fy]])
        principal_point = torch.tensor([[cx, cy]])
        camera = PerspectiveCameras(focal_length=focal_length,
                                        principal_point=principal_point,
                                        R=rotation_matrix,
                                        T=translation_vector,
                                        image_size=torch.tensor([[height, width]]),
                                        in_ndc=False,
                                        device=device)
            
        raster_settings = PointsRasterizationSettings(
                image_size=(height, width), 
                radius = 0.007, # 0.007
                points_per_pixel = 10
                )

        rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings) 
        renderer = PointsRenderer(
                rasterizer=rasterizer,
                compositor=AlphaCompositor(background_color = (1.0, 1.0, 1.0)) # 背景色归一化
            )
        rendered_image = renderer(point_cloud)
        rendered_image = (rendered_image[0].cpu().numpy() * 255).astype(np.uint8)
        color = rendered_image[..., :3]
        color_image = Image.fromarray(color)
        color_image.save(name)

        fragments = rasterizer(point_cloud, cameras=camera)
        zbuf =fragments.zbuf[0, ..., 0].cpu().numpy()

        zbuf[zbuf < 0] = 0
        depth_image_path = depth_name.replace(".npy", ".tif")
        cv2.imwrite(depth_image_path, zbuf)
        # 彩色深度图
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # np.save(depth_name, zbuf)
        zbuf_vis = zbuf.copy()
        zbuf_vis[zbuf_vis == 0] = np.nan 
        zbuf_norm = (zbuf_vis - np.nanmin(zbuf_vis)) / (np.nanmax(zbuf_vis) - np.nanmin(zbuf_vis))

        colormap = cm.get_cmap('Spectral_r')  
        color_img = colormap(zbuf_norm)   
        color_img = (color_img[:, :, :3] * 255).astype(np.uint8)  
        Image.fromarray(color_img).save(depth_name.replace(".npy", "_color.png"))


    def scene_image_rendering(self, scan_pc_raw, scene_name, mode = ["global"], bird_lift_cam=None):
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
        
        for folder in [color_folder, depth_folder, intrinsic_folder, pose_folder]:
            os.makedirs(folder, exist_ok=True)

        w_raw, l_raw, h_raw, scene_center = self.get3d_box_from_pcs(scan_pc_raw)
        scale_factor = self.zoomout + 1
        w, l, h = w_raw * scale_factor, l_raw * scale_factor, h_raw * scale_factor

        extrinsic_list = []
        intrinsic_list = []
        pc_for_generation = scan_pc_np
        bird_height = bird_lift_cam if bird_lift_cam is not None else self.bird_lift_cam

        generation_functions = {
            "global": lambda: self.global_level_camera_generation(w, l, h, scene_center, pc_for_generation),
            "wide": lambda: self.wide_angle_level_camera_generation(w, l, h, scene_center, pc_for_generation),
            "corner": lambda: self.corner_angle_level_camera_generation(w, l, h, scene_center, pc_for_generation, self.corner_lift_cam),
            "bird": lambda: self.bird_eye_level_camera_generation(w, l, h, scene_center, pc_for_generation, bird_height),
        }

        for mode_key in mode:
            if mode_key not in generation_functions:
                print(f"[Warning] Unsupported render mode '{mode_key}', ignored.")
                continue
            extrinsic, intrinsic = generation_functions[mode_key]()
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

            np.save(intrinsic_path, intrinsic)
            np.save(pose_path, extrinsic)
            
            self.render_pcd(
                extrinsic,
                intrinsic[:3, :3],
                self.image_width,
                self.image_height,
                scan_pc.to(self.device),
                color_path,
                depth_path
            )
        print("\n-> Rendering complete.")
        return extrinsic, intrinsic

    # ===================================================================
    # 相机视角生成函数（保留原始global，新增wide/corner/bird）
    # ===================================================================
    
    def global_level_camera_generation(self, w, l, h, scene_center, scan_pc):
        extrinsic_list = []
        intrinsic_list = []
        camera_locations = self.generate_camera_locations(scene_center, w, l, h, 5) # 数字 5 控制生成图像的数量
        for camera_location in camera_locations:
            camera_location[-1] += self.lift_cam
            pose_matrix = self.lookat(camera_location, scene_center, np.array([0, 0, -1]))
            pose_to_RBT = np.transpose(np.linalg.inv(np.transpose(pose_matrix)))
            intrinsic_calibrated = self.intrinsic_calibration(
                scan_pc, pose_to_RBT, self.image_width, self.image_height
            )
            extrinsic_list.append(pose_to_RBT)
            intrinsic_list.append(intrinsic_calibrated)
        return extrinsic_list, intrinsic_list

    def wide_angle_level_camera_generation(self, w, l, h, scene_center, scan_pc):
        third_width, third_length, half_height = w / 3, l / 3, h / 2
        half_width, half_length = w / 2, l / 2
        top_z = scene_center[2] + half_height
        bottom_z = scene_center[2] - half_height

        camera_positions = [
            np.array([scene_center[0] - third_width / 2, scene_center[1] - third_length / 2, top_z]),
            np.array([scene_center[0] + third_width / 2, scene_center[1] - third_length / 2, top_z]),
            np.array([scene_center[0] - third_width / 2, scene_center[1] + third_length / 2, top_z]),
            np.array([scene_center[0] + third_width / 2, scene_center[1] + third_length / 2, top_z]),
        ]

        target_positions = [
            np.array([scene_center[0] + half_width, scene_center[1] + half_length, bottom_z]),
            np.array([scene_center[0] - half_width, scene_center[1] + half_length, bottom_z]),
            np.array([scene_center[0] + half_width, scene_center[1] - half_length, bottom_z]),
            np.array([scene_center[0] - half_width, scene_center[1] - half_length, bottom_z]),
        ]

        scan_pc_np = np.asarray(scan_pc)
        scene_masks = [
            (scan_pc_np[:, 0] > camera_positions[0][0]) & (scan_pc_np[:, 1] > camera_positions[0][1]),
            (scan_pc_np[:, 0] < camera_positions[1][0]) & (scan_pc_np[:, 1] > camera_positions[1][1]),
            (scan_pc_np[:, 0] > camera_positions[2][0]) & (scan_pc_np[:, 1] < camera_positions[2][1]),
            (scan_pc_np[:, 0] < camera_positions[3][0]) & (scan_pc_np[:, 1] < camera_positions[3][1]),
        ]

        extrinsic_list, intrinsic_list = [], []

        for i, base_location in enumerate(camera_positions):
            mask_points = scan_pc_np[scene_masks[i]]
            if mask_points.shape[0] == 0:
                print("\n*** WARNING ***\nWide view skipped: no points visible from this quadrant.\n")
                continue

            camera_location = base_location.copy()
            camera_location[-1] += self.lift_cam
            pose_matrix = self.lookat(camera_location, target_positions[i], np.array([0, 0, -1]))
            pose_to_RBT = np.transpose(np.linalg.inv(np.transpose(pose_matrix)))
            intrinsic_calibrated = self.intrinsic_calibration(
                mask_points, pose_to_RBT, self.image_width, self.image_height
            )
            extrinsic_list.append(pose_to_RBT)
            intrinsic_list.append(intrinsic_calibrated)

        return extrinsic_list, intrinsic_list

    def corner_angle_level_camera_generation(self, w, l, h, scene_center, scan_pc, corner_lift_cam=None):
        half_width, half_length, half_height = w / 2, l / 2, h / 2
        top_z = scene_center[2] + half_height
        bottom_z = scene_center[2] - half_height

        camera_positions = [scene_center.copy() for _ in range(4)]

        target_positions = [
            np.array([scene_center[0] + half_width, scene_center[1] + half_length, bottom_z]),
            np.array([scene_center[0] - half_width, scene_center[1] + half_length, bottom_z]),
            np.array([scene_center[0] + half_width, scene_center[1] - half_length, bottom_z]),
            np.array([scene_center[0] - half_width, scene_center[1] - half_length, bottom_z]),
        ]

        scan_pc_np = np.asarray(scan_pc)
        scene_masks = [
            (scan_pc_np[:, 0] > camera_positions[0][0]) & (scan_pc_np[:, 1] > camera_positions[0][1]),
            (scan_pc_np[:, 0] < camera_positions[1][0]) & (scan_pc_np[:, 1] > camera_positions[1][1]),
            (scan_pc_np[:, 0] > camera_positions[2][0]) & (scan_pc_np[:, 1] < camera_positions[2][1]),
            (scan_pc_np[:, 0] < camera_positions[3][0]) & (scan_pc_np[:, 1] < camera_positions[3][1]),
        ]

        extrinsic_list, intrinsic_list = [], []
        corner_lift = corner_lift_cam if corner_lift_cam is not None else getattr(self, "corner_lift_cam", self.lift_cam)

        for i, base_location in enumerate(camera_positions):
            mask_points = scan_pc_np[scene_masks[i]]
            if mask_points.shape[0] == 0:
                print("\n*** WARNING ***\nCorner view skipped: no points visible from this quadrant.\n")
                continue

            camera_location = base_location.copy()
            camera_location[-1] = top_z + corner_lift
            pose_matrix = self.lookat(camera_location, target_positions[i], np.array([0, 0, -1]))
            pose_to_RBT = np.transpose(np.linalg.inv(np.transpose(pose_matrix)))
            intrinsic_calibrated = self.intrinsic_calibration(
                mask_points, pose_to_RBT, self.image_width, self.image_height
            )
            extrinsic_list.append(pose_to_RBT)
            intrinsic_list.append(intrinsic_calibrated)

        return extrinsic_list, intrinsic_list

    def bird_eye_level_camera_generation(self, w, l, h, scene_center, scan_pc, bird_lift_cam):
        extrinsic_list, intrinsic_list = [], []

        camera_height = scene_center[2] + (h / 2) + bird_lift_cam
        camera_location = np.array([scene_center[0], scene_center[1], camera_height])
        target_position = scene_center.copy()

        pose_matrix = self.lookat(camera_location, target_position, np.array([0, 1, 0]))
        pose_to_RBT = np.transpose(np.linalg.inv(np.transpose(pose_matrix)))

        intrinsic_calibrated = self.intrinsic_calibration(
            scan_pc, pose_to_RBT, self.image_width, self.image_height
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

    def generate_camera_locations(self, center, width, length, height, num_split=5):
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
    """
    主执行函数，负责批量处理所有指定的NPY点云文件。
    """
    INPUT_ROOT = "/home/Data/data2/wcl/DataSet/STPLS3D/Synthetic_v3_Instance_nosample_ply_processed_all/validation"
    # OUTPUT_ROOT = "/home/Data/data2/wcl/DataSet/STPLS3D_Open3DIS_testmapping"
    OUTPUT_ROOT = "/home/Data/data2/wcl/DataSet/STPLS3D_Open3DIS/2D"
    
    image_width = 2000
    image_height = 2000
    lift_cam = 20.0   # 当前设置下，相机指向场景中心，修改这个变化不明显
    zoomout = 1.0
    remove_lip = 0.0

    adjust_camera = [lift_cam, zoomout, remove_lip]
    image_size = [image_width, image_height]

    file_pattern = os.path.join(INPUT_ROOT, '*_points_GTv3_*.npy')
    npy_files = sorted(glob.glob(file_pattern))

    if not npy_files:
        print(f"错误: 未找到匹配 '{file_pattern}' 的文件。请检查INPUT_ROOT路径。")
        return

    print(f"找到 {len(npy_files)} 个场景进行处理。")

    for npy_path in tqdm(npy_files, desc="整体进度"):
        scene_name = os.path.basename(npy_path).replace('.npy', '')
        tqdm.write(f"\n--- 开始处理场景: {scene_name} ---")

        scene_output_dir = os.path.join(OUTPUT_ROOT, scene_name)
        
        snap_module = Snap(image_size, adjust_camera, scene_output_dir)

        pcd_rgb = snap_module.read_npy_point_cloud(npy_path)
        if pcd_rgb is None:
            tqdm.write(f"!! 跳过场景 {scene_name} 因为点云加载失败。")
            continue
        
        snap_module.scene_image_rendering(
            pcd_rgb[:, :6],
            scene_name,
            mode=["global"],
            bird_lift_cam=10.0,
        )  # 可选模式 "global", "wide", "corner", "bird"

    tqdm.write("\n\n所有场景处理完毕！")

if __name__ == "__main__":
    main()
