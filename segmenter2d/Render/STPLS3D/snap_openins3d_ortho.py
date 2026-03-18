import torch
import numpy as np
import torch
from pytorch3d.io import IO
from pytorch3d.renderer import (
    PerspectiveCameras,
    # 新增：导入正交相机
    OrthographicCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    AmbientLights,
    HardPhongShader,
    BlendParams,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor)
from tqdm import tqdm
import pytorch3d
from PIL import Image
import numpy as np
import os
import random
from pytorch3d.structures import Pointclouds
from PIL import Image, ImageDraw, ImageFont
import plyfile
import numpy as np
import pandas as pd
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from utils import mask_lable_location, mask_rasterization # 注释掉本地依赖
import glob
import sys

class Snap:
    def __init__(self, image_size, adjust_camera, save_folder):
        self.image_width, self.image_height = image_size
        self.lift_cam, self.zoomout, self.remove_lip = adjust_camera
        self.save_folder = save_folder
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"

    # --- 文件读取部分 (无改动) ---
    def read_ply_point_cloud(self, filepath):
        with open(filepath, "rb") as f:
            plydata = plyfile.PlyData.read(f)
        if "vertex" not in plydata: return None
        vertex_element = plydata["vertex"]
        props = [p.name for p in vertex_element.properties]
        if not all(p in props for p in ['x', 'y', 'z', 'red', 'green', 'blue']):
            print("Error: PLY file must contain x, y, z, red, green, blue properties.")
            return None
        points = np.vstack([vertex_element['x'], vertex_element['y'], vertex_element['z']]).T
        colors = np.vstack([vertex_element['red'], vertex_element['green'], vertex_element['blue']]).T
        return np.hstack([points, colors])
    
    def read_npy_point_cloud(self, filepath):
        data = np.load(filepath)
        if data.shape[1] < 6:
            print("Error: npy文件的点云数据至少需要6列（xyzrgb）")
            return None
        return data[:, :6]
    
    # --- 几何处理部分 (无改动) ---
    def get3d_box_from_pcs(self, pc):
        w = pc[:, 0].max() - pc[:, 0].min()
        l = pc[:, 1].max() - pc[:, 1].min()
        h = pc[:, 2].max() - pc[:, 2].min()
        scene_center = np.array([
            pc[:, 0].max() - w / 2, pc[:, 1].max() - l / 2, pc[:, 2].max() - h / 2,
        ])
        return w, l, h, scene_center

    # ==================== 代码修改点 1: 改造渲染函数以支持两种投影 ====================
    def render_pcd(self, pose, intrinsics, image_width, image_height, pcd, name, projection_type):
        """
        统一处理透视和正交两种投影的渲染。
        'intrinsics' 现在是一个包含相机参数的元组，而不是矩阵。
        """
        device = self.device
        # 统一将颜色归一化到[0,1]范围
        pts = torch.tensor(pcd[:, :3], dtype=torch.float32, device=device)
        feats = torch.tensor(pcd[:, 3:6] / 255.0, dtype=torch.float32, device=device)
        point_cloud = Pointclouds(points=[pts], features=[feats])
        
        # 统一数据类型为 float32
        pose_tensor = torch.tensor(pose, dtype=torch.float32, device=device)
        
        # PyTorch3D相机需要世界到相机的变换矩阵 R 和 T
        # pose 是 cam_to_world, 所以 world_to_cam 是它的逆
        world_to_cam = torch.inverse(pose_tensor)
        R = world_to_cam[:3, :3].unsqueeze(0)
        T = world_to_cam[:3, 3].unsqueeze(0)

        znear = 0.01
        zfar = 1000.0  # 将远裁剪平面从默认的100米大幅增加到1000米
        
        if projection_type == 'perspective':
            fx, fy, cx, cy = intrinsics
            camera = PerspectiveCameras(
                focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
                principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
                R=R, T=T, image_size=torch.tensor([[image_height, image_width]]), in_ndc=False, device=device
            )
        elif projection_type == 'orthographic':
            scale_x, scale_y, cx, cy = intrinsics
            camera = OrthographicCameras(
                focal_length=torch.tensor([[scale_x, scale_y]], dtype=torch.float32, device=device),
                principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
                R=R, T=T, image_size=torch.tensor([[image_height, image_width]]), in_ndc=False, device=device
            )
        else:
            raise ValueError(f"未知的投影类型: {projection_type}")
            
        raster_settings = PointsRasterizationSettings(
            image_size=(image_height, image_width), 
            radius=0.007,
            points_per_pixel=10
        )
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
            compositor=AlphaCompositor(background_color=(1, 1, 1)) # 白色背景
        )
        
        rendered_image = renderer(point_cloud)
        rendered_image_np = rendered_image[0, ..., :3].cpu().numpy()
        color_image = Image.fromarray((rendered_image_np * 255).astype(np.uint8))
        color_image.save(name)

    # ==================== 代码修改点 2: 主流程函数增加投影类型选项 ====================
    def scene_image_rendering(self, scan_pc_raw, scene_name, mode=["global"], mask=None, projection_type='perspective'):
        self.scene_name = scene_name
        data_type = "pcd" # 简化流程，只处理点云
        
        # ... 此处省略原版中处理 mesh 和 mask 的代码 ...

        # 应用 remove_lip
        z_max = scan_pc_raw[:, 2].max()
        idx_remained = scan_pc_raw[:, 2] <= (z_max - self.remove_lip)
        scan_pc_np = scan_pc_raw[idx_remained, :]
        scan_pc = torch.tensor(scan_pc_np, dtype=torch.float32)

        # 根据投影类型创建文件夹
        folder_prefix = f"{self.save_folder}_{projection_type}"
        pose_folder = f"{folder_prefix}/{scene_name}/pose"
        image_folder = f"{folder_prefix}/{scene_name}/image"
        for folder in [pose_folder, image_folder]:
            os.makedirs(folder, exist_ok=True)

        w_raw, l_raw, h_raw, scene_center = self.get3d_box_from_pcs(scan_pc_raw)
        scale_factor = self.zoomout + 1
        w, l, h = w_raw * scale_factor, l_raw * scale_factor, h_raw * scale_factor

        extrinsic_list = []
        intrinsic_list = []

        generation_functions = {"global": self.global_level_camera_generation}
        for mode_key, func in generation_functions.items():
            if mode_key in mode:
                # 将 projection_type 传递下去
                extrinsic, intrinsic_params = func(w, l, h, scene_center, scan_pc_raw, projection_type)
                extrinsic_list.extend(extrinsic)
                intrinsic_list.extend(intrinsic_params)

        print(f"\n***************** 开始为 {scene_name} 渲染图像 ({projection_type} 投影) *****************")

        for extrinsic, intrinsic, i in zip(extrinsic_list, intrinsic_list, range(len(extrinsic_list))):
            sys.stdout.write(f"\rSnap module: 正在渲染第 {i+1}/{len(extrinsic_list)} 张图像")
            sys.stdout.flush()

            np.save(f"{pose_folder}/pose_matrix_calibrated_angle_{i}.npy", extrinsic)
            
            # 调用改造后的渲染函数
            self.render_pcd(
                extrinsic,
                intrinsic, # 现在是参数元组
                self.image_width,
                self.image_height,
                scan_pc.to(self.device),
                f"{image_folder}/image_rendered_angle_{i}.png",
                projection_type
            )
        
        print("\n渲染完成.")
        return extrinsic_list, intrinsic_list

    # ==================== 代码修改点 3: 相机生成函数支持两种模式 ====================
    def global_level_camera_generation(self, w, l, h, scene_center, scan_pc, projection_type):
        extrinsic_list = []
        intrinsic_list = []
        
        # 使用环绕相机位置
        camera_locations = self.generate_camera_locations(scene_center, w, l, h, num_split=3)
        
        for camera_location in camera_locations:
            camera_location[-1] += self.lift_cam
            
            # lookat 函数生成 camera-to-world 矩阵
            pose_matrix = self.lookat(camera_location, scene_center, np.array([0, 0, 1]))
            
            # 根据投影类型选择不同的内参校准函数
            if projection_type == 'perspective':
                intrinsic_calibrated = self.perspective_intrinsic_calibration(
                    scan_pc, pose_matrix, self.image_width, self.image_height
                )
            elif projection_type == 'orthographic':
                intrinsic_calibrated = self.orthographic_intrinsic_calibration(
                    scan_pc, pose_matrix, self.image_width, self.image_height
                )
            else:
                raise ValueError(f"未知的投影类型: {projection_type}")

            extrinsic_list.append(pose_matrix)
            intrinsic_list.append(intrinsic_calibrated)
        return extrinsic_list, intrinsic_list

    # ==================== 代码修改点 4: 重命名原内参函数并新增正交内参函数 ====================
    def perspective_intrinsic_calibration(self, point_cloud, pose, width, height):
        """原有的透视校准逻辑，返回 (fx, fy, cx, cy) 元组。"""
        # 初始内参，作为基准
        fx, fy, cx, cy = 577.6, 578.7, width / 2, height / 2

        points_h = np.hstack([point_cloud[:, :3], np.ones((point_cloud.shape[0], 1))])
        world_to_cam = np.linalg.inv(pose)
        points_cam = (world_to_cam @ points_h.T).T
        
        # 避免除以零或负的z值
        points_cam = points_cam[points_cam[:, 2] > 0]
        if len(points_cam) == 0: return (fx, fy, cx, cy)

        # 投影
        x_proj = points_cam[:, 0] * fx / points_cam[:, 2]
        y_proj = points_cam[:, 1] * fy / points_cam[:, 2]

        # 调整主点以将物体中心移到(0,0)附近
        cx_new = -x_proj.min()
        cy_new = -y_proj.min()

        # 用新主点重新计算投影范围
        x_proj_new = x_proj + cx_new
        y_proj_new = y_proj + cy_new
        
        # 计算缩放比例
        scale_x = width / x_proj_new.max() if x_proj_new.max() > 0 else 1
        scale_y = height / y_proj_new.max() if y_proj_new.max() > 0 else 1
        scale = min(scale_x, scale_y) * 0.95 # 乘以0.95增加一点边距

        # 更新最终参数
        fx_new, fy_new = fx * scale, fy * scale
        cx_new, cy_new = cx_new * scale, cy_new * scale

        return (fx_new, fy_new, cx_new, cy_new)

    def orthographic_intrinsic_calibration(self, point_cloud, pose, width, height):
        """
        正交相机的内参标定：
          1. 对点云投影范围做 margin（乘 0.95），
          2. 自动把 3D 包围盒中心对齐到图像中心。
        返回：(scale_x, scale_y, cx, cy)
        """
        # 1. 把点变换到相机坐标系
        points_h = np.hstack([point_cloud[:, :3], np.ones((point_cloud.shape[0], 1))])
        world_to_cam = np.linalg.inv(pose)
        pts_cam = (world_to_cam @ points_h.T).T  # (N,4)

        # 2. 只关注 XY 维
        x_cam = pts_cam[:, 0]
        y_cam = pts_cam[:, 1]

        # 3. 计算投影后范围（未缩放）
        x_min, x_max = x_cam.min(), x_cam.max()
        y_min, y_max = y_cam.min(), y_cam.max()
        box_w = x_max - x_min
        box_h = y_max - y_min
        if box_w == 0 or box_h == 0:
            # 防止除 0
            return (1.0, 1.0, width / 2.0, height / 2.0)

        # 4. 留点边距，再算 scale
        scale = min(width / box_w, height / box_h) * 0.95

        # 5. 计算包围盒在相机空间中的中心
        x_center = (x_max + x_min) / 2.0
        y_center = (y_max + y_min) / 2.0

        # 6. 把 3D 中心对齐到图像中心
        #    投影公式： x_screen = scale * x_cam + cx
        #    我们希望 x_cam = x_center 时，对应 x_screen = width/2
        cx = width / 2.0 - x_center * scale
        cy = height / 2.0 - y_center * scale

        return (scale, scale, cx, cy)


    def generate_camera_locations(self, center, width, length, height, num_split=5):
        """环绕式生成相机位置。"""
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
    
    # ==================== 代码修改点 5: 修正lookat函数以返回标准的camera-to-world矩阵 ====================
    def lookat(self, eye, at, up):
        """生成一个标准的 camera-to-world 变换矩阵。"""
        eye = np.array(eye, dtype=np.float64)
        at = np.array(at, dtype=np.float64)
        up = np.array(up, dtype=np.float64)

        # z-axis 指向相机后方
        z_axis = eye - at
        z_axis /= np.linalg.norm(z_axis)
        
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        
        # y-axis 重新计算以保证正交性
        y_axis = np.cross(z_axis, x_axis)

        # cam_to_world 矩阵的旋转部分和平移部分
        cam_to_world = np.identity(4)
        cam_to_world[:3, 0] = x_axis
        cam_to_world[:3, 1] = y_axis
        cam_to_world[:3, 2] = z_axis
        cam_to_world[:3, 3] = eye
        
        return cam_to_world

def main():
    """
    展示如何使用SNAP模块，并为室外大场景配置和渲染两种投影视图。
    """
    # 推荐的室外场景参数
    image_width = 2000
    image_height = 2000
    lift_cam = 5      # 较高的相机高度以减少透视畸变
    zoomout = 0.05     # 较小的边距
    remove_lip = 0

    adjust_camera = [lift_cam, zoomout, remove_lip]
    image_size = [image_width, image_height]

    # 请修改为您的.npy文件路径
    npy_pcd_path = '/data2/wcl/DataSet/STPLS3D/Synthetic_v3_Instance_nosample_ply_processed/validation/10_points_GTv3_0.npy'

    # --- 1. 使用正交投影进行渲染 (推荐方案) ---
    snap_module_ortho = Snap(image_size, adjust_camera, save_folder="render_output/ortho")
    pcd_rgb_ortho = snap_module_ortho.read_npy_point_cloud(npy_pcd_path)
    if pcd_rgb_ortho is not None:
        snap_module_ortho.scene_image_rendering(
            pcd_rgb_ortho[:, :6], 
            "outdoor_scene", 
            mode=["global"],
            projection_type='orthographic' # 指定投影类型
        )

    # # --- 2. 使用透视投影进行渲染 (对比方案) ---
    # snap_module_persp = Snap(image_size, adjust_camera, save_folder="render_output")
    # pcd_rgb_persp = snap_module_persp.read_npy_point_cloud(npy_pcd_path)
    # if pcd_rgb_persp is not None:
    #     snap_module_persp.scene_image_rendering(
    #         pcd_rgb_persp[:, :6], 
    #         "outdoor_scene", 
    #         mode=["global"],
    #         projection_type='perspective' # 指定投影类型
    #     )

if __name__ == "__main__":
    main()