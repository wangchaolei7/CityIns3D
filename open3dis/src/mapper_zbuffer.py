import os
import re

import numpy as np
import torch
import torch_scatter
# from pytorch3d.renderer import PerspectiveCameras

class PointCloudToImageMapper(object):
    def __init__(
        self,
        image_dim,
        visibility_threshold=1.0,
        cut_bound=0,
        device="cpu",
        use_torch=False,  # Scannet: 0.1, Scannetpp: 0.1, KITTI360: 1.5, stpls3d: 1.0
        visibility_mode="dynamic",  # "fixed" or "dynamic"
        dynamic_vis_scale=0.01,    # used in dynamic mode
        dynamic_vis_min=0.05       # used in dynamic mode
    ):
        self.image_dim = image_dim
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound
        self.device = device
        # visibility threshold mode
        self.visibility_mode = visibility_mode
        self.dynamic_vis_scale = dynamic_vis_scale
        self.dynamic_vis_min = dynamic_vis_min

    def compute_mapping_torch(self, camera_to_world, coords, image_dim, depth=None, intrinsic=None, vis_thresh=None, id_1="scene", id_2="frame", visibility_mode=None):

        """
        :param camera_to_world: 4x4, 可以是 numpy array 或 torch.Tensor
        :param coords: N x 3 format, torch.Tensor
        :param depth: H x W format, 可以是 numpy array 或 torch.Tensor
        :param intrinsic: 3x3 format, 可以是 numpy array 或 torch.Tensor
        :return: mapping, N x 3 format, (H,W,mask)
        """
        device = coords.device
        if vis_thresh is not None:
            self.vis_thres = vis_thresh
        # allow per-call override of visibility mode (keeps backward compatibility)
        current_visibility_mode = self.visibility_mode if visibility_mode is None else visibility_mode
        
        if intrinsic is not None:
            current_intrinsic = intrinsic
        if isinstance(current_intrinsic, np.ndarray):
            current_intrinsic = torch.from_numpy(current_intrinsic).to(device).float()

        if current_intrinsic.shape == (4, 4):
            current_intrinsic = current_intrinsic[:3, :3]
        elif current_intrinsic.shape != (3, 3):
            raise ValueError(
                f"Intrinsic matrix has unexpected shape: {current_intrinsic.shape}. "
                f"Expected (3,3) or (4,4)."
            )

        if isinstance(camera_to_world, np.ndarray):
            camera_to_world = torch.from_numpy(camera_to_world).to(device).float()

        mapping = torch.zeros((3, coords.shape[0]), dtype=torch.long, device=device)
        coords_new = torch.cat([coords, torch.ones([coords.shape[0], 1], dtype=torch.float, device=device)], dim=1).T

        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = torch.linalg.inv(camera_to_world)
        p = world_to_camera @ coords_new.float()
        
        # p_proj = current_intrinsic @ p[:3, :]
        # p_proj_norm = p_proj / (p_proj[2, :] + 1e-8)
        # pi = torch.round(p_proj_norm[:2, :]).long()
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = torch.round(p).long()  # simply round the projected coordinates

        if depth is not None:
            if isinstance(depth, np.ndarray):
                depth = torch.from_numpy(depth).to(device)
            else:
                depth = depth.to(device)
            
            # 使用 depth 张量自身的维度进行边界检查，而不是 image_dim
            h, w = depth.shape
            base_mask = (
                (pi[0] >= self.cut_bound)
                * (pi[1] >= self.cut_bound)
                * (pi[0] < w - self.cut_bound)  # 使用 depth 的宽度
                * (pi[1] < h - self.cut_bound)  # 使用 depth 的高度
            )

            # 提前过滤，确保只有在图像内的点才会被用来索引depth
            valid_indices_mask = base_mask

            # 创建一个默认为False的遮挡掩码
            occ_ok = torch.zeros_like(valid_indices_mask, dtype=torch.bool)

            # 只对在图像内的点计算遮挡关系
            if valid_indices_mask.any():
                depth_here = depth
                z_cam = p[2]

                depth_values = depth_here[pi[1][valid_indices_mask], pi[0][valid_indices_mask]]
                z_cam_vals = z_cam[valid_indices_mask]

                if current_visibility_mode == "dynamic":
                    # 动态逐点阈值: tol = clamp(scale * z_cam, min=min_tol)
                    tol_vals = torch.clamp(self.dynamic_vis_scale * z_cam_vals, min=self.dynamic_vis_min)
                    occ_ok[valid_indices_mask] = torch.abs(depth_values - z_cam_vals) <= tol_vals
                else:
                    # 固定阈值
                    occ_ok[valid_indices_mask] = torch.abs(depth_values - z_cam_vals) <= self.vis_thres

            # 最终掩码：边界内且遮挡关系满足
            inside_mask = base_mask & occ_ok
        else:
            # 当没有深度图时，边界检查仍然应该基于 image_dim
            inside_mask = (
                (pi[0] >= self.cut_bound)
                * (pi[1] >= self.cut_bound)
                * (pi[0] < image_dim[1] - self.cut_bound)
                * (pi[1] < image_dim[0] - self.cut_bound)
            )
            front_mask = p[2] > 0
            inside_mask = front_mask * inside_mask

        new_inside_mask = inside_mask

        # 恢复：不在此函数中保存过滤点
        # 在depth_occlusion_mask计算之后，保存过滤后的点云
        filtered_coords = coords[new_inside_mask].cpu().numpy()
        scene_frame_id = f"{id_1}_{id_2}"
        if filtered_coords.shape[0] > 0:
            frame_save_path = "/home/Data/data2/wcl/Open3DIS/exp_stpls3d/version_SAM/mapping_vispoints"  # 可修改保存路径
            frame_save_path_dir = os.path.join(frame_save_path, id_1, f"{scene_frame_id}.txt")
            os.makedirs(os.path.dirname(frame_save_path_dir), exist_ok=True)
            np.savetxt(frame_save_path_dir, 
                    filtered_coords,
                    fmt='%.6f',  # 保留6位小数
                    delimiter=',',
                    header='x,y,z') 
            # print(f"已保存{len(filtered_coords)}个过滤后的点到{frame_save_path_dir}")

        mapping[0][new_inside_mask] = pi[1][new_inside_mask]
        mapping[1][new_inside_mask] = pi[0][new_inside_mask]
        mapping[2][new_inside_mask] = 1
        # n = save_visible_points(mapping, coords, "visible_points_frame123.ply", fmt="ply")
        # print("saved visible points:", n)

        return mapping.T



# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from pytorch3d.structures import Pointclouds
# from pytorch3d.renderer import (
#     PerspectiveCameras,
#     PointsRasterizationSettings,
#     PointsRasterizer,
# )

# # [NEW and FINAL version]
# def generate_depth_from_z_buffer(
#     points,
#     pose,
#     intrinsic,
#     image_dim,
#     cfg,      # 用于获取保存路径配置
#     frame,    # 用于获取文件名和基础路径
#     device,
#     radius=0.05, # Scannet: 0.05, Scannetpp: 0.05, KITTI360: 0.03
# ):
#     """
#     使用 pytorch3d 的 Z-Buffer 从点云生成深度图，并将其保存到磁盘。
#     此版本严格遵循提供的模板来修复 "expected Float but found Half" 错误。

#     Args:
#         points (torch.Tensor): (N, 3+) 点云坐标.
#         pose (np.ndarray): (4, 4) 相机位姿 (camera-to-world).
#         intrinsic (np.ndarray): (3, 3) 相机内参.
#         image_dim (tuple): (height, width) 输出图像的尺寸.
#         cfg (object): 包含配置信息的对象.
#         frame (dict): 包含当前帧信息的字典.
#         device (str): 计算设备.
#         radius (float): 点云光栅化半径.

#     Returns:
#         torch.Tensor: (H, W) 生成的深度图张量.
#     """

#     # --- 1. 输入准备和类型转换 (来自模板) ---
#     points = points.to(device=device, dtype=torch.float32)
#     if isinstance(pose, np.ndarray):
#         pose = torch.from_numpy(pose)
#     pose = pose.to(device=device, dtype=torch.float32)
#     if isinstance(intrinsic, np.ndarray):
#         intrinsic = torch.from_numpy(intrinsic)
#     intrinsic = intrinsic.to(device=device, dtype=torch.float32)

#     # --- 2. Pytorch3D 相机和光栅化器设置 (来自模板) ---
#     world_to_camera = torch.linalg.inv(pose)
#     R = world_to_camera[:3, :3].T.unsqueeze(0)
#     T = world_to_camera[:3, 3].unsqueeze(0)

#     fx, fy = intrinsic[0, 0], intrinsic[1, 1]
#     cx, cy = intrinsic[0, 2], intrinsic[1, 2]

#     cameras = PerspectiveCameras(
#         focal_length=((-fx, -fy),),
#         principal_point=((cx, cy),),
#         R=R,
#         T=T,
#         image_size=(image_dim,),
#         in_ndc=False,
#         device=device,
#     )
#     pcd_for_render = Pointclouds(points=[points[:, :3]])
#     raster_settings = PointsRasterizationSettings(
#         image_size=image_dim,
#         radius=radius,
#         points_per_pixel=10,
#         max_points_per_bin=2000000,
#     )
#     rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)

#     # --- 3. [核心修复] 执行光栅化 (完全遵循您提供的模板) ---
#     try:
#         # 检查CUDA和AMP模块是否存在以避免错误
#         if torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast'):
#             with torch.cuda.amp.autocast(enabled=False):
#                 # 确保在无autocast环境下，输入也是float32
#                 fragments = rasterizer(pcd_for_render)
#         else:
#             fragments = rasterizer(pcd_for_render)
#     except Exception as e:
#         print(f"Error during PyTorch3D rasterization: {e}")
#         raise # 重新引发异常

#     # --- 4. 提取 Z-Buffer ---
#     # zbuf的形状是 (1, H, W, points_per_pixel)，我们取最近的点
#     z_buffer = fragments.zbuf[0, ..., 0]

#     # --- 5. [新功能] 文件保存逻辑 ---
#     color_dir = os.path.dirname(frame["image_path"])
#     base_dir = os.path.dirname(color_dir)  # color 的上一级目录
#     frame_name = os.path.splitext(os.path.basename(frame["image_path"]))[0]
    
#     raw_depth_dir = os.path.join(base_dir, cfg.data.depth_zbuffer_folder)
#     color_depth_dir = os.path.join(base_dir, cfg.data.depth_zbuffer_color_folder)
#     os.makedirs(raw_depth_dir, exist_ok=True)
#     os.makedirs(color_depth_dir, exist_ok=True)
    
#     raw_depth_path = os.path.join(raw_depth_dir, f"{frame_name}.png")
#     color_depth_path = os.path.join(color_depth_dir, f"{frame_name}.jpg")

#     depth_map_np = z_buffer.cpu().numpy()
#     # 创建一个掩码，只选择有效的深度值（大于0且小于无穷大代理值）
#     valid_mask = (depth_map_np > 0) & (depth_map_np < 1e4)
    
#     if valid_mask.any():
#         min_depth = depth_map_np[valid_mask].min()
#         max_depth = depth_map_np[valid_mask].max()
        
#         if max_depth > min_depth:
#             normalized_depth = (depth_map_np - min_depth) / (max_depth - min_depth)
#             normalized_depth[~valid_mask] = 0 # 将背景设为0
#         else: # 如果所有有效深度都相同
#             normalized_depth = np.zeros_like(depth_map_np, dtype=np.float32)
            
#         # 保存16位单通道深度图
#         depth_u16 = (normalized_depth * 65535).astype(np.uint16)
#         cv2.imwrite(raw_depth_path, depth_u16)
        
#         # 保存彩色可视化深度图
#         cmap = plt.get_cmap('Spectral')
#         colored_depth = cmap(normalized_depth)[:, :, :3]
#         colored_depth_u8 = (colored_depth * 255).astype(np.uint8)
#         # 将无效区域设为黑色
#         colored_depth_u8[~valid_mask] = [0, 0, 0]
#         colored_depth_bgr = cv2.cvtColor(colored_depth_u8, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(color_depth_path, colored_depth_bgr)
    
#     # --- 6. 准备并返回用于 mapping 的张量 (来自模板) ---
#     # 背景像素深度为-1，将其设置为一个非常大的值
#     z_buffer[z_buffer < 0] = 1e5
    
#     return z_buffer

































# def generate_depth_from_z_buffer(
#     points, pose, intrinsic, image_dim, device, radius=0.005
# ):
#     """
#     使用 pytorch3d 的 Z-Buffer 方法从点云动态生成深度图。

#     Args:
#         points (torch.Tensor): (N, 3+) 点云坐标 (至少需要XYZ).
#         pose (torch.Tensor): (4, 4) 相机位姿 (camera-to-world).
#         intrinsic (torch.Tensor): (3, 3) 相机内参.
#         image_dim (tuple): (height, width) 输出图像的尺寸.
#         device (str): 计算设备.
#         radius (float): 点云光栅化半径，是需要根据场景调整的重要超参数.

#     Returns:
#         torch.Tensor: (H, W) 生成的深度图.
#     """

#     # Ensure all inputs are on the correct device and dtype (float32)
#     points = points.to(device=device, dtype=torch.float32)
#     if isinstance(pose, np.ndarray):
#         pose = torch.from_numpy(pose)
#     pose = pose.to(device=device, dtype=torch.float32)
#     if isinstance(intrinsic, np.ndarray):
#         intrinsic = torch.from_numpy(intrinsic)
#     intrinsic = intrinsic.to(device=device, dtype=torch.float32)

#     # PyTorch3D needs world-to-camera transform
#     world_to_camera = torch.linalg.inv(pose)
#     # PyTorch3D's camera R is C_view -> C_world rotation, so transpose
#     R = world_to_camera[:3, :3].T.unsqueeze(0)
#     T = world_to_camera[:3, 3].unsqueeze(0)

#     fx, fy = intrinsic[0, 0], intrinsic[1, 1]
#     cx, cy = intrinsic[0, 2], intrinsic[1, 2]

#     cameras = PerspectiveCameras(
#         focal_length=((fx, fy),),
#         principal_point=((cx, cy),),
#         R=R,
#         T=T,
#         image_size=(image_dim,),
#         in_ndc=False,
#         device=device,
#     )

#     # Use points' XYZ coordinates. Colors are not needed for depth map generation.
#     pcd_for_render = Pointclouds(
#         points=[points[:, :3]], features=[torch.ones_like(points[:, :3])]
#     )

#     raster_settings = PointsRasterizationSettings(
#         image_size=image_dim,
#         radius=radius,
#         points_per_pixel=10,
#     )

#     rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)

#     # --- START OF CHANGE ---
#     # Wrap the rasterization call in a no-autocast context to force float32
#     # This is the most common fix for "expected Float but found Half" errors
#     # when explicit type casting doesn't seem to solve it.
#     try:
#         # Check if CUDA is available and AMP module exists to avoid errors on CPU or older PyTorch
#         if torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast'):
#             with torch.cuda.amp.autocast(enabled=False):
#                 fragments = rasterizer(pcd_for_render)
#         else:
#             fragments = rasterizer(pcd_for_render)
#     except Exception as e:
#         print(f"Error during PyTorch3D rasterization: {e}")
#         # Re-raise the exception to propagate the error up the call stack
#         raise
#     # --- END OF CHANGE ---

#     # zbuf has shape (1, H, W, points_per_pixel), we take the nearest point
#     z_buffer = fragments.zbuf[0, ..., 0]

#     # Background pixels have depth -1, set them to a very large value (infinity)
#     z_buffer[z_buffer < 0] = 1e5
    
#     return z_buffer
