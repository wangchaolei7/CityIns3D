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
                radius = 0.01, # 0.007
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

        np.save(depth_name, zbuf)
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
        
        for folder in [color_folder, depth_folder, intrinsic_folder, pose_folder]:
            os.makedirs(folder, exist_ok=True)

        w_raw, l_raw, h_raw, scene_center = self.get3d_box_from_pcs(scan_pc_raw)
        scale_factor = self.zoomout + 1
        w, l, h = w_raw * scale_factor, l_raw * scale_factor, h_raw * scale_factor

        extrinsic_list = []
        intrinsic_list = []
        generation_functions = {"global": self.global_level_camera_generation}
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
    # 以下函数保持您提供的原始版本，未做任何修改
    # ===================================================================
    
    def global_level_camera_generation(self, w, l, h, scene_center, scan_pc):
        extrinsic_list = []
        intrinsic_list = []
        camera_locations = self.generate_camera_locations(scene_center, w, l, h, 3)
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
    INPUT_ROOT = "/data2/wcl/DataSet/Zhangzhou/D01/split_75"
    OUTPUT_ROOT = "/data2/wcl/DataSet/Zhangzhou_Open3DIS"
    
    image_width = 2000
    image_height = 2000
    lift_cam = 1.0
    zoomout = 1.0
    remove_lip = 0.0

    adjust_camera = [lift_cam, zoomout, remove_lip]
    image_size = [image_width, image_height]

    file_pattern = os.path.join(INPUT_ROOT, 'D01_*.txt')
    npy_files = sorted(glob.glob(file_pattern))

    if not npy_files:
        print(f"错误: 未找到匹配 '{file_pattern}' 的文件。请检查INPUT_ROOT路径。")
        return

    print(f"找到 {len(npy_files)} 个场景进行处理。")

    for npy_path in tqdm(npy_files, desc="整体进度"):
        scene_name = os.path.basename(npy_path).replace('.npy', '').replace('.txt', '')
        tqdm.write(f"\n--- 开始处理场景: {scene_name} ---")

        scene_output_dir = os.path.join(OUTPUT_ROOT, scene_name)
        
        snap_module = Snap(image_size, adjust_camera, scene_output_dir)

        pcd_rgb = snap_module.read_txt_point_cloud(npy_path)
        if pcd_rgb is None:
            tqdm.write(f"!! 跳过场景 {scene_name} 因为点云加载失败。")
            continue
        
        snap_module.scene_image_rendering(pcd_rgb[:, :6], scene_name, mode=["global"])

    tqdm.write("\n\n所有场景处理完毕！")

if __name__ == "__main__":
    main()