import os
import re

import numpy as np
import torch
import torch_scatter
import matplotlib.pyplot as plt
import math
import cv2
from datetime import datetime
from scipy.interpolate import griddata

from PIL import Image
import time
import psutil

class MappingCalculator:
    def __init__(
        self,
        image_dim,
        visibility_threshold=0.1,
        cut_bound=0,
        intrinsics=None,
        device="cpu",
        use_torch=False,
        point_cloud_resolution=0.1, #scannet200-0.05 # 点云分辨率（米）
        max_distance=20,            #scannet200-5 # 最大可见距离（米）
        expansion_factor=0.2,         #expansion_factor-2# 扩展因子k
        image_shape=(376, 1408),    #scannet200-(968, 1296)  # 图像尺寸（H, W）
        r_max=25                    #scannet200-5.5 # 区域采样半径（米）
    ):
        self.image_dim = image_dim
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound
        self.intrinsics = intrinsics
        self.device = device
        if use_torch and self.intrinsics is not None:
            self.intrinsics = torch.from_numpy(self.intrinsics).to(device)

        self.c = point_cloud_resolution
        self.R = max_distance
        self.k = expansion_factor
        self.H, self.W = image_shape
        self.r_max = r_max  # 区域采样半径

        self.total_time = 0.0

    def generate_depth_map(self, z_buffer, image_dim, max_depth=5.0, 
                        output_path='/data2/wcl/Open3DIS/exp_kitti360/YoloW-SAM_ablation/vis_projectpoints/false_depth/0000_sync_0000000372_0000000610/0000_sync_0000000372_0000000610.png'):
        # output_path='/data2/wcl/Open3DIS/exp_scannet200/version_Grounded-SAM_depth/vis_projectpoints/false_depth/scene0011_00/scene0011_00.png'
        """
        使用Z-Buffer生成深度图并保存为PNG。

        :param z_buffer: Z-Buffer，形状为 [H, W]，记录每个像素的最小深度值
        :param image_dim: 图像分辨率，(width, height)
        :param max_depth: 最大深度值，用于归一化，默认为5.0米
        :param output_path: 保存深度图的路径
        :return: 生成的深度图 (numpy数组)
        """
        # 将Z-Buffer中的无穷大值替换为0
        depth_map = z_buffer.clone()
        depth_map[depth_map == float('inf')] = 0

        # 转换为numpy数组
        depth_map_np = depth_map.cpu().numpy()

        # 归一化到uint16范围 (0-65535)
        valid_depth = depth_map_np[depth_map_np > 0]
        if valid_depth.size > 0:
            actual_max_depth = min(valid_depth.max(), max_depth)
        else:
            actual_max_depth = max_depth

        depth_map_np = np.clip(depth_map_np, 0, actual_max_depth)
        depth_map_np = (depth_map_np / actual_max_depth * 65535).astype(np.uint16)

        # 生成时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name, file_name = os.path.split(output_path)
        base_name, ext = os.path.splitext(file_name)
        new_filename = f"{base_name}_{timestamp}{ext}"
        new_output_path = os.path.join(dir_name, new_filename)

        # 保存深度图
        os.makedirs(dir_name, exist_ok=True)
        cv2.imwrite(new_output_path, depth_map_np)
        print(f'深度图已保存至 {new_output_path}')

        return depth_map_np

    def compute_mapping_zbuff(
        self,
        camera_to_world: np.ndarray,
        coords: torch.Tensor,  # 假设输入coords已经是Tensor并移到device
        intrinsic: torch.Tensor  # 假设intrinsic已经是Tensor
    ) -> np.ndarray:
        """
        :param camera_to_world: 4x4 相机到世界坐标系的变换矩阵
        :param coords: N×3 点云坐标（世界坐标系，Tensor）
        :param intrinsic: 3x3 相机内参矩阵（Tensor）
        :return: N×3 张量，列分别为(y坐标, x坐标, 可见性掩码)
        """
        start_mem = psutil.Process().memory_info().rss
        start_time = time.time()

        coords = coords.float().to(self.device)
        camera_position = torch.from_numpy(camera_to_world[:3, 3]).float().to(self.device)

        # 区域采样（Sphere Sampling）
        dist = torch.norm(coords - camera_position, dim=1)
        sphere_mask = dist <= self.r_max  # 半径r_max内的点
        coords_sampled = coords[sphere_mask]
        valid_indices_sphere = torch.where(sphere_mask)[0]


        # 转换到相机坐标系
        coords_homo = torch.cat([
            coords_sampled,
            torch.ones(len(coords_sampled), 1, device=self.device)
        ], dim=1)
        world_to_camera = torch.inverse(torch.from_numpy(camera_to_world).float().to(self.device))
        camera_coords = (coords_homo @ world_to_camera.T)[:, :3]


        # 计算相机的水平/垂直视野角
        fx = intrinsic[0, 0].item()
        fy = intrinsic[1, 1].item()
        image_width = self.W
        image_height = self.H
        theta_x = 2 * math.atan(image_width / (2 * fx))  # 水平视野角（弧度）
        theta_y = 2 * math.atan(image_height / (2 * fy))  # 垂直视野角（弧度）

        # 过滤有效点（深度、距离、横向视野）
        valid_mask = (
            (camera_coords[:, 2] > 0) &  # 点在相机前方
            (camera_coords[:, 2] <= self.R) &  # 距离不超过max_distance
            (torch.abs(camera_coords[:, 0]) <= camera_coords[:, 2] * math.tan(theta_x / 2)) &  # 水平视野裁剪
            (torch.abs(camera_coords[:, 1]) <= camera_coords[:, 2] * math.tan(theta_y / 2))  # 垂直视野裁剪
        )
        valid_coords = camera_coords[valid_mask]
        valid_indices = valid_indices_sphere[valid_mask]
        after_filter = psutil.Process().memory_info().rss

        # 计算立方体尺寸（延缓衰减）
        d = valid_coords[:, 2].float()
        cube_sizes = self.c * (1 + self.k * torch.exp(-d / self.R)).float()  # 延长衰减距离
        # cube_sizes = self.c * (self.k * torch.exp(2 - (-d / self.R)))  # 延长衰减距离

        # 投影到图像平面
        fx_val = intrinsic[0, 0].item()
        fy_val = intrinsic[1, 1].item()
        cx_val = intrinsic[0, 2].item()
        cy_val = intrinsic[1, 2].item()
        xs = ((valid_coords[:, 0] / valid_coords[:, 2] * fx_val) + cx_val).float()
        ys = ((valid_coords[:, 1] / valid_coords[:, 2] * fy_val) + cy_val).float()
        centers = torch.stack([xs, ys], dim=1).round().long()

        #         # 初始化Z-Buffer
        # z_buffer = torch.full((self.H, self.W), float('inf'), dtype=torch.float32, device=self.device)
        # mapping = torch.zeros((len(coords), 3), dtype=torch.int32, device=self.device)

        # # 向量化核心计算（替换原循环）
        # with torch.no_grad():
        #     # 提取关键参数
        #     valid_coords = valid_coords.float()
        #     centers = centers.float()
        #     cube_sizes = cube_sizes.float()
        #     fx_val = intrinsic[0, 0].item()
        #     fy_val = intrinsic[1, 1].item()

        #     # 计算所有点的区域边界
        #     d = valid_coords[:, 2].clamp_min(0.1)  # 防止除零
        #     half_size_x = (cube_sizes * fx_val / (2 * d)).int()
        #     half_size_y = (cube_sizes * fy_val / (2 * d)).int()

        #     x_center = centers[:, 0]
        #     y_center = centers[:, 1]

        #     x_min = (x_center - half_size_x).clamp_min(0).int()
        #     x_max = (x_center + half_size_x).clamp_max(self.W).int()
        #     y_min = (y_center - half_size_y).clamp_min(0).int()
        #     y_max = (y_center + half_size_y).clamp_max(self.H).int()

        #     # 筛选有效区域
        #     valid_x = (x_min >= self.cut_bound) & (x_max < self.W - self.cut_bound)
        #     valid_y = (y_min >= self.cut_bound) & (y_max < self.H - self.cut_bound)
        #     valid_mask = valid_x & valid_y

        #     # 仅保留有效点
        #     valid_indices = torch.where(valid_mask)[0]
        #     if valid_indices.numel() == 0:
        #         return mapping.cpu().numpy()

        #     # 提取有效点的参数
        #     valid_coords = valid_coords[valid_indices]
        #     x_min = x_min[valid_indices]
        #     x_max = x_max[valid_indices]
        #     y_min = y_min[valid_indices]
        #     y_max = y_max[valid_indices]
        #     d = d[valid_indices]

        #     # 批量更新Z-Buffer（向量化核心）
        #     # 注意：这一步需要根据具体逻辑实现，示例使用循环但减少计算量
        #     for idx in range(len(valid_indices)):
        #         # 这里仍需循环，但有效点数量已大幅减少
        #         p = valid_coords[idx]
        #         xm, xM = x_min[idx], x_max[idx]
        #         ym, yM = y_min[idx], y_max[idx]

        #         # 更新Z-Buffer（向量化操作）
        #         region = z_buffer[ym:yM, xm:xM]
        #         mask = (p[2] < region)
        #         region[mask] = p[2]

        #         # 更新可见性
        #         if mask.any():
        #             original_idx = valid_indices[idx]
        #             mapping[original_idx, 0] = (ym + yM) // 2
        #             mapping[original_idx, 1] = (xm + xM) // 2
        #             mapping[original_idx, 2] = 1
        # 初始化Z-Buffer和映射结果
        z_buffer = torch.full((self.H, self.W), float('inf'), dtype=torch.float32, device=self.device)
        mapping = torch.zeros((len(coords), 3), dtype=torch.int32, device=self.device)

        # 遍历每个点并更新Z-Buffer（CPU/GPU加速优化）
        for idx in range(len(valid_coords)):
            p = valid_coords[idx].float()
            center = centers[idx].float()
            cube_size = cube_sizes[idx]
            d = p[2].item()  # 点到相机的距离（米）

            # 防止除零
            d = max(d, 0.1)  # 设置最小距离阈值

            # 分方向计算像素半宽高
            half_size_x = int( (cube_size * fx_val) / (2 * d) )
            half_size_y = int( (cube_size * fy_val) / (2 * d) )

            # 计算区域边界
            x_min = max(0, int(center[0].item() - half_size_x))
            x_max = min(self.W, int(center[0].item() + half_size_x))
            y_min = max(0, int(center[1].item() - half_size_y))
            y_max = min(self.H, int(center[1].item() + half_size_y))

            # 检查是否超出图像边界（考虑cut_bound）
            if x_min < self.cut_bound or x_max >= self.W - self.cut_bound:
                continue
            if y_min < self.cut_bound or y_max >= self.H - self.cut_bound:
                continue

            # 更新Z-Buffer（向量化操作）
            mask = (p[2] < z_buffer[y_min:y_max, x_min:x_max])
            z_buffer[y_min:y_max, x_min:x_max][mask] = p[2].float()

            # 记录可见性
            if mask.any():
                original_idx = valid_indices[idx].item()
                mapping[original_idx, 0] = (y_min + y_max) // 2  # 中心y坐标
                mapping[original_idx, 1] = (x_min + x_max) // 2  # 中心x坐标
                mapping[original_idx, 2] = 1  # 可见性掩码


        # 生成深度图
        # image_dim = (self.W, self.H)  # 使用与 compute_mapping_zbuff 一致的分辨率
        # depth_map = self.generate_depth_map(z_buffer, image_dim, max_depth=5.0)

        # 保存可见点（可选）
        visible_indices = torch.where(mapping[:,2] == 1)[0]
        visible_coords = coords[visible_indices].cpu().numpy()
        # save_path = "/data2/wcl/Open3DIS/exp_scannet200/version_Grounded-SAM_test/vis_projectpoints/visible_points"
        save_path = "/data2/wcl/Open3DIS/exp_kitti360/YoloW-SAM_ablation/vis_projectpoints/visible_points"
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, "visible_points.txt")
        # np.savetxt(
        #     save_file,
        #     visible_coords,
        #     fmt='%.6f',
        #     delimiter=',',
        #     header='x,y,z'
        # )
        # print(f"已保存{len(visible_coords)}个可见点到{save_file}")

        end_mem = psutil.Process().memory_info().rss
        end_time = time.time()
        memory_used = (end_mem - start_mem) / (1024 ** 2)
        time_used = end_time - start_time
        print(f"Z-buffer 方法 - 内存使用: {memory_used:.2f} MB, 时间: {time_used:.4f} 秒")
        return mapping


class PointCloudToImageMapper(object):
    def __init__(
        self, image_dim, visibility_threshold=0.1, cut_bound=0, intrinsics=None, device="cpu", use_torch=False
    ):

        self.image_dim = image_dim
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound
        self.intrinsics = intrinsics

        self.device = device
        if use_torch:
            self.intrinsics = torch.from_numpy(self.intrinsics).to(device)

    def compute_mapping_torch(self, camera_to_world, coords, depth=None, intrinsic=None, vis_thresh = None, id_1=None, id_2=None):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        device = coords.device
        # scene_save_path = "/data2/wcl/Open3DIS/exp_scannet200/version_Grounded-SAM/vis_projectpoints/true_depth"
        # scene_save_path_dir = os.path.join(scene_save_path, id_1, f"{id_1}.txt")
        # os.makedirs(os.path.dirname(scene_save_path_dir), exist_ok=True)
        # np.savetxt(scene_save_path_dir,
        #         coords.cpu().numpy(),
        #         fmt='%.6f',
        #         delimiter=',',
        #         header='x,y,z')
        
        if vis_thresh != None:
            self.vis_thres = vis_thresh
        if intrinsic is not None: # adjust intrinsic
            self.intrinsics = intrinsic
        else:
            intrinsic = self.intrinsics
        camera_to_world = torch.from_numpy(camera_to_world).to(device).float()

        mapping = torch.zeros((3, coords.shape[0]), dtype=torch.long, device=device)
        coords_new = torch.cat([coords, torch.ones([coords.shape[0], 1], dtype=torch.float, device=device)], dim=1).T

        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = torch.linalg.inv(camera_to_world)
        p = world_to_camera.float() @ coords_new.float()
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = torch.round(p).long()  # simply round the projected coordinates
        inside_mask = (
            (pi[0] >= self.cut_bound)
            * (pi[1] >= self.cut_bound)
            * (pi[0] < self.image_dim[0] - self.cut_bound)
            * (pi[1] < self.image_dim[1] - self.cut_bound)
        )
        if depth is not None:
            depth = torch.from_numpy(depth).to(device)
            occlusion_mask = torch.abs(depth[pi[1][inside_mask], pi[0][inside_mask]] - p[2][inside_mask]) <= self.vis_thres
            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2] > 0  # make sure the depth is in front
            inside_mask = front_mask * inside_mask

        new_inside_mask = inside_mask
                # 在depth_occlusion_mask计算之后，保存过滤后的点云
        filtered_coords = coords[new_inside_mask].cpu().numpy()
        scene_frame_id = f"{id_1}_{id_2}"
        if filtered_coords.shape[0] > 0:
            frame_save_path = "/data2/wcl/Open3DIS/exp_scannet200/version_Grounded-SAM_depth/vis_projectpoints/vis_backgorund"  # 可修改保存路径
            # frame_save_path_dir = os.path.join(frame_save_path, id_1, f"{scene_frame_id}.txt")
            # os.makedirs(os.path.dirname(frame_save_path_dir), exist_ok=True)
            # np.savetxt(frame_save_path_dir, 
            #         filtered_coords,
            #         fmt='%.6f',  # 保留6位小数
            #         delimiter=',',
            #         header='x,y,z') 
            # print(f"已保存{len(filtered_coords)}个过滤后的点到{frame_save_path_dir}")

        mapping[0][new_inside_mask] = pi[1][new_inside_mask]
        mapping[1][new_inside_mask] = pi[0][new_inside_mask]
        mapping[2][new_inside_mask] = 1

        return mapping.T

    def compute_mapping_torch_scannetpp(self, camera_to_world, coords, depth=None, intrinsic=None, vis_thresh = None, id_1=None, id_2=None):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        device = coords.device
        # scene_save_path = "/data2/wcl/Open3DIS/exp_scannet200/version_Grounded-SAM/vis_projectpoints/true_depth"
        # scene_save_path_dir = os.path.join(scene_save_path, id_1, f"{id_1}.txt")
        # os.makedirs(os.path.dirname(scene_save_path_dir), exist_ok=True)
        # np.savetxt(scene_save_path_dir,
        #         coords.cpu().numpy(),
        #         fmt='%.6f',
        #         delimiter=',',
        #         header='x,y,z')
        
        if vis_thresh != None:
            self.vis_thres = vis_thresh
        if intrinsic is not None: # adjust intrinsic
            self.intrinsics = intrinsic
        else:
            intrinsic = self.intrinsics
        camera_to_world = torch.from_numpy(camera_to_world).to(device).float()

        mapping = torch.zeros((3, coords.shape[0]), dtype=torch.long, device=device)
        coords_new = torch.cat([coords, torch.ones([coords.shape[0], 1], dtype=torch.float, device=device)], dim=1).T

        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = torch.linalg.inv(camera_to_world)
        p = world_to_camera.float() @ coords_new.float()
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = torch.round(p).long()  # simply round the projected coordinates
        inside_mask = (
            (pi[0] >= self.cut_bound)
            * (pi[1] >= self.cut_bound)
            * (pi[0] < self.image_dim[0] - self.cut_bound)
            * (pi[1] < self.image_dim[1] - self.cut_bound)
        )
        if depth is not None:
            depth = torch.from_numpy(depth).to(device)
            occlusion_mask = torch.abs(depth[pi[1][inside_mask], pi[0][inside_mask]] - p[2][inside_mask]) <= self.vis_thres
            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2] > 0  # make sure the depth is in front
            inside_mask = front_mask * inside_mask

        new_inside_mask = inside_mask
                # 在depth_occlusion_mask计算之后，保存过滤后的点云
        filtered_coords = coords[new_inside_mask].cpu().numpy()
        scene_frame_id = f"{id_1}_{id_2}"
        if filtered_coords.shape[0] > 0:
            frame_save_path = "/data2/wcl/Open3DIS/exp_scannetpp/open3dis_0.45/version_Grounded-SAM/vis_projectpoints/vis_visiblpoints_fig1"  # 可修改保存路径
            frame_save_path_dir = os.path.join(frame_save_path, id_1, f"{scene_frame_id}.txt")
            # os.makedirs(os.path.dirname(frame_save_path_dir), exist_ok=True)
            # np.savetxt(frame_save_path_dir, 
            #         filtered_coords,
            #         fmt='%.6f',  # 保留6位小数
            #         delimiter=',',
            #         header='x,y,z') 
            # print(f"已保存{len(filtered_coords)}个过滤后的点到{frame_save_path_dir}")

        mapping[0][new_inside_mask] = pi[1][new_inside_mask]
        mapping[1][new_inside_mask] = pi[0][new_inside_mask]
        mapping[2][new_inside_mask] = 1

        return mapping.T




    # def generate_depth_map(self, filtered_coords, camera_to_world, intrinsic, image_dim, max_depth=5.0, 
    #                     output_path='/data2/wcl/Open3DIS/exp_scannet200/version_Grounded-SAM_depth/vis_projectpoints/false_depth/fig3/scene0011_00/scene0011_00.png', 
    #                     cube_size=0.05):
    #     """
    #     使用过滤后的可见点以立方体投影的方式生成深度图，并保存为PNG。
        
    #     :param filtered_coords: 过滤后的可见点，形状为 [N, 3]，世界坐标系下的点
    #     :param camera_to_world: 相机到世界的变换矩阵，4x4
    #     :param intrinsic: 相机内参矩阵，3x3
    #     :param image_dim: 图像分辨率，(width, height)
    #     :param max_depth: 最大深度值，用于归一化
    #     :param output_path: 保存深度图的路径
    #     :param cube_size: 立方体投影的尺寸（单位：米），控制投影区域大小
    #     :return: 生成的深度图 (numpy数组)
    #     """
    #     # 设置设备为 CUDA
    #     device = 'cuda'
        
    #     # 将输入转换为张量并移到 GPU 上
    #     camera_to_world = camera_to_world.to(device).double()
    #     world_to_camera = torch.linalg.inv(camera_to_world)  # 计算世界到相机的变换矩阵
    #     filtered_coords = torch.from_numpy(filtered_coords).to(device).double()

    #     # 将世界坐标系下的点转换为相机坐标系
    #     coords_homo = torch.cat([filtered_coords, torch.ones(filtered_coords.shape[0], 1, device=device)], dim=1).T  # [4, N]
    #     p_camera = world_to_camera @ coords_homo  # [4, N]
    #     p_camera = p_camera[:3]  # [3, N]，取前三维 (x, y, z)

    #     # 投影到图像平面
    #     fx, fy = intrinsic[0, 0], intrinsic[1, 1]  # 焦距
    #     cx, cy = intrinsic[0, 2], intrinsic[1, 2]  # 主点
    #     u = (p_camera[0] / p_camera[2] * fx) + cx  # 横坐标
    #     v = (p_camera[1] / p_camera[2] * fy) + cy  # 纵坐标
    #     depth = p_camera[2]  # 深度值

    #     # 过滤有效像素（在图像范围内且深度大于0）
    #     mask = (u >= 0) & (u < image_dim[0]) & (v >= 0) & (v < image_dim[1]) & (depth > 0)
    #     u = u[mask]
    #     v = v[mask]
    #     depth = depth[mask]

    #     # 初始化 Z-Buffer，用无穷大填充
    #     z_buffer = torch.full((image_dim[1], image_dim[0]), float('inf'), device=device)

    #     # 遍历每个点，以立方体投影的方式更新 Z-Buffer
    #     for i in range(u.shape[0]):
    #         center_u = int(round(u[i].item()))  # 投影中心的横坐标
    #         center_v = int(round(v[i].item()))  # 投影中心的纵坐标
    #         d = depth[i].item()  # 当前点的深度

    #         # 防止除零，设置最小深度
    #         d = max(d, 0.1)

    #         # 根据深度和立方体尺寸计算投影区域的像素半宽高
    #         half_size = int((cube_size * fx) / (2 * d))  # 假设 fx ≈ fy

    #         # 计算投影区域的边界，确保不超出图像范围
    #         x_min = max(0, center_u - half_size)
    #         x_max = min(image_dim[0], center_u + half_size)
    #         y_min = max(0, center_v - half_size)
    #         y_max = min(image_dim[1], center_v + half_size)

    #         # 在投影区域内更新 Z-Buffer，保留最小深度值
    #         for y in range(y_min, y_max):
    #             for x in range(x_min, x_max):
    #                 if depth[i] < z_buffer[y, x]:
    #                     z_buffer[y, x] = depth[i]

    #     # 将 Z-Buffer 转换为深度图
    #     depth_map = z_buffer.clone()
    #     depth_map[depth_map == float('inf')] = 0  # 未填充区域设为0
    #     depth_map_np = depth_map.cpu().numpy()

    #     # 归一化到 uint16 范围 (0-65535)
    #     valid_depth = depth_map_np[depth_map_np > 0]
    #     if valid_depth.size > 0:
    #         actual_max_depth = min(valid_depth.max(), max_depth)
    #     else:
    #         actual_max_depth = max_depth

    #     depth_map_np = np.clip(depth_map_np, 0, actual_max_depth)
    #     depth_map_np = (depth_map_np / actual_max_depth * 65535).astype(np.uint16)

    #     # 生成带时间戳的文件名
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     dir_name, file_name = os.path.split(output_path)
    #     base_name, ext = os.path.splitext(file_name)
    #     new_filename = f"{base_name}_{timestamp}{ext}"
    #     new_output_path = os.path.join(dir_name, new_filename)

    #     # 保存深度图
    #     os.makedirs(dir_name, exist_ok=True)
    #     cv2.imwrite(new_output_path, depth_map_np)
    #     print(f'深度图已保存至 {new_output_path}')

    #     return depth_map_np


    def generate_depth_map(self,filtered_coords, camera_to_world, intrinsic, image_dim, max_depth=5.0, 
                           output_path='/data2/wcl/Open3DIS/exp_scannet200/version_Grounded-SAM_depth/vis_projectpoints/false_depth/fig3/scene0011_00/scene0011_00.png'):
        """
        使用稀疏点云生成深度图，并通过插值填充空白区域，生成平滑的深度图并保存为PNG。
        
        :param filtered_coords: 过滤后的点云，形状为 [N, 3]，世界坐标系
        :param camera_to_world: 相机到世界的变换矩阵，4x4
        :param intrinsic: 相机内参，3x3
        :param image_dim: 图像分辨率，(width, height)
        :param max_depth: 最大深度值，用于归一化
        :param output_path: 保存深度图的路径
        :return: 生成的深度图 (numpy数组)
        """
        device = 'cuda'
        camera_to_world = camera_to_world.double()
        world_to_camera = torch.linalg.inv(camera_to_world)

        filtered_coords = torch.from_numpy(filtered_coords).to(device)
        # 将点云转换到相机坐标系
        coords_new = torch.cat([filtered_coords.double(), 
                            torch.ones([filtered_coords.shape[0], 1], dtype=torch.double, device=device)], dim=1).T  # [4, N]
        p_camera = world_to_camera @ coords_new  # [4, N]

        # 投影到图像平面
        p_image = p_camera[:3] / p_camera[2:3]  # [3, N]
        u = p_image[0] * intrinsic[0, 0] + intrinsic[0, 2]
        v = p_image[1] * intrinsic[1, 1] + intrinsic[1, 2]
        depth = p_camera[2]

        # 过滤有效像素
        mask = (u >= 0) & (u < image_dim[0]) & (v >= 0) & (v < image_dim[1]) & (depth > 0)
        u = u[mask]
        v = v[mask]
        depth = depth[mask]

        u_int = torch.round(u).long()
        v_int = torch.round(v).long()

        # 初始化深度图，空白区域为0
        depth_map = np.zeros((image_dim[1], image_dim[0]), dtype=np.float32)

        # 填充深度值（取最小深度）
        for i in range(u_int.shape[0]):
            curr_depth = depth[i].item()
            if depth_map[v_int[i], u_int[i]] == 0 or curr_depth < depth_map[v_int[i], u_int[i]]:
                depth_map[v_int[i], u_int[i]] = curr_depth

        # 创建掩膜：255表示有值区域，0表示空白区域
        mask = (depth_map > 0).astype(np.uint8) * 255

        # 使用cv2.inpaint进行插值
        depth_map_filled = cv2.inpaint(depth_map, ~mask, 1, cv2.INPAINT_NS)  ### cv2.INPAINT_TELEA cv2.INPAINT_NS

        # 归一化到uint16范围 (0-65535)
        valid_depth = depth_map_filled[depth_map_filled > 0]
        if valid_depth.size > 0:
            actual_max_depth = min(valid_depth.max(), max_depth)
        else:
            actual_max_depth = max_depth

        depth_map_filled = np.clip(depth_map_filled, 0, actual_max_depth)
        depth_map_filled = (depth_map_filled / actual_max_depth * 65535).astype(np.uint16)

        # 生成时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
        dir_name, file_name = os.path.split(output_path)         # 拆分路径
        base_name, ext = os.path.splitext(file_name)             # 拆分文件名与扩展名
        new_filename = f"{base_name}_{timestamp}{ext}"           # 拼接新文件名
        new_output_path = os.path.join(dir_name, new_filename)   # 组合完整路径**

        # 保存深度图
        cv2.imwrite(new_output_path, depth_map_filled)
        print(f'深度图已保存至 {new_output_path}')

        return depth_map_filled
    
    def compute_mapping_torch_rect200(self, camera_to_world, coords, depth=None, intrinsic=None, vis_thresh=None, downscale_factor=64, id_1=None, id_2=None):
        """
        :param downscale_factor: 图像降采样倍数
        其他参数与原函数一致
        """
        device = coords.device
        scene_save_path = "/data2/wcl/Open3DIS/exp_scannet200/version_Grounded-SAM_depth/vis_projectpoints/false_depth"
        scene_save_path_dir = os.path.join(scene_save_path, id_1, f"{id_1}.txt")
        # os.makedirs(os.path.dirname(scene_save_path_dir), exist_ok=True)
        # np.savetxt(scene_save_path_dir,
        #         coords.cpu().numpy(),
        #         fmt='%.6f',
        #         delimiter=',',
        #         header='x,y,z')
        # print(f"已保存场景点云{coords.shape[0]}个点到{scene_save_path}")

        if vis_thresh is not None:
            self.vis_thres = vis_thresh
        if intrinsic is not None:
            self.intrinsics = intrinsic
        else:
            intrinsic = self.intrinsics

        original_image_dim = self.image_dim
        current_image_dim = (original_image_dim[0] // downscale_factor, 
                        original_image_dim[1] // downscale_factor)
        intrinsic_low = intrinsic.clone() if torch.is_tensor(intrinsic) else intrinsic.copy()
        intrinsic_low[0, 0] /= downscale_factor  # fx
        intrinsic_low[1, 1] /= downscale_factor  # fy
        intrinsic_low[0, 2] /= downscale_factor  # cx
        intrinsic_low[1, 2] /= downscale_factor  # cy
        camera_to_world = torch.from_numpy(camera_to_world).to(device).double()

        mapping = torch.zeros((3, coords.shape[0]), dtype=torch.long, device=device)
        coords_new = torch.cat([coords.double(), torch.ones([coords.shape[0], 1], dtype=torch.double, device=device)], dim=1).T

        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = torch.linalg.inv(camera_to_world)
        p = world_to_camera.double() @ coords_new.double()

        p[0] = (p[0] * intrinsic_low[0][0]) / p[2] + intrinsic_low[0][2]
        p[1] = (p[1] * intrinsic_low[1][1]) / p[2] + intrinsic_low[1][2]

        pi = torch.round(p).long()
        
        inside_mask = (
            (pi[0] >= self.cut_bound // downscale_factor)
            * (pi[1] >= self.cut_bound // downscale_factor)
            * (pi[0] < current_image_dim[0] - self.cut_bound // downscale_factor)
            * (pi[1] < current_image_dim[1] - self.cut_bound // downscale_factor)
        )

        front_mask = p[2] > 0
        inside_mask = front_mask * inside_mask

        pi_x_ = pi[1][inside_mask]
        pi_y_ = pi[0][inside_mask]
        pi_depth_ = p[2][inside_mask]
        
        inds = pi_x_ * self.image_dim[0] + pi_y_

        unique_inds, inverse = torch.unique(inds, return_inverse=True)

        depth_min, argmin = torch_scatter.scatter_min(
            pi_depth_.double(), 
            inverse, 
            dim=0
        )

        valid_min_mask = depth_min > 0
        valid_argmin = argmin[valid_min_mask]

        if valid_argmin.numel() > 0:
            local_indices = torch.where(inside_mask)[0][valid_argmin]
            min_depth_coords = coords[local_indices]
            scene_frame_id = f"{id_1}_{id_2}"
            save_min_depth_path = "/data2/wcl/Open3DIS/exp_scannet200/version_Grounded-SAM_depth/vis_projectpoints/false_depth/fig3"
            save_min_depth_path_dir = os.path.join(save_min_depth_path, id_1, f"{scene_frame_id}_min.txt")
            os.makedirs(os.path.dirname(save_min_depth_path_dir), exist_ok=True)
            # np.savetxt(save_min_depth_path_dir, 
            #         min_depth_coords.cpu().numpy(),
            #         fmt='%.6f',
            #         delimiter=',',
            #         header='x,y,z')
            # print(f"已保存{len(min_depth_coords)}个最小深度点到{save_min_depth_path_dir}")

        valid_min_mask = depth_min > 0
        depth_min = torch.where(valid_min_mask, depth_min, torch.zeros_like(depth_min))
        depth_min_broadcast = depth_min[inverse]
        depth_diff = (pi_depth_ - depth_min_broadcast)

        THRESHOLD = 0.12
        depth_occlusion_mask = (depth_diff <= THRESHOLD) & (depth_diff >= 0)

        new_inside_mask = inside_mask.clone()
        new_inside_mask[inside_mask] = depth_occlusion_mask
        # 在depth_occlusion_mask计算之后，保存过滤后的点云
        filtered_coords = coords[new_inside_mask].cpu().numpy()
        if filtered_coords.shape[0] > 0:
            save_min_filter_path = "/data2/wcl/Open3DIS/exp_scannet200/version_Grounded-SAM_depth/vis_projectpoints/false_depth/fig3"  # 可修改保存路径
            save_min_filter_path_dir = os.path.join(save_min_filter_path, id_1, f"{scene_frame_id}_filter.txt")
            os.makedirs(os.path.dirname(save_min_filter_path_dir), exist_ok=True)
            # np.savetxt(save_min_filter_path_dir, 
            #         filtered_coords,
            #         fmt='%.6f',  # 保留6位小数
            #         delimiter=',',
            #         header='x,y,z')
            # print(f"已保存{len(filtered_coords)}个过滤后的点到{save_min_filter_path_dir}")

        mapping[0][new_inside_mask] = pi[1][new_inside_mask] * downscale_factor
        mapping[1][new_inside_mask] = pi[0][new_inside_mask] * downscale_factor
        mapping[2][new_inside_mask] = 1

        # depth_map = self.generate_depth_map(filtered_coords, camera_to_world, intrinsic, original_image_dim, max_depth=5.0)

        return mapping.T
    

    def compute_mapping_for_multi(self, camera_to_world, coords, depth=None, intrinsic=None, vis_thresh=None, 
                                  downscale_factor=24, threshold=0.1, id_1=None, id_2=None):

        device = coords.device
        if vis_thresh is not None:
            self.vis_thres = vis_thresh
        if intrinsic is not None:
            self.intrinsics = intrinsic
        else:
            intrinsic = self.intrinsics

        original_image_dim = self.image_dim
        current_image_dim = (original_image_dim[0] // downscale_factor, 
                        original_image_dim[1] // downscale_factor)
        intrinsic_low = intrinsic.clone() if torch.is_tensor(intrinsic) else intrinsic.copy()
        intrinsic_low[0, 0] /= downscale_factor  # fx
        intrinsic_low[1, 1] /= downscale_factor  # fy
        intrinsic_low[0, 2] /= downscale_factor  # cx
        intrinsic_low[1, 2] /= downscale_factor  # cy
        camera_to_world = torch.from_numpy(camera_to_world).to(device).double()

        mapping = torch.zeros((3, coords.shape[0]), dtype=torch.long, device=device)
        coords_new = torch.cat([coords.double(), torch.ones([coords.shape[0], 1], dtype=torch.double, device=device)], dim=1).T

        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = torch.linalg.inv(camera_to_world)
        p = world_to_camera.double() @ coords_new.double()

        p[0] = (p[0] * intrinsic_low[0][0]) / p[2] + intrinsic_low[0][2]
        p[1] = (p[1] * intrinsic_low[1][1]) / p[2] + intrinsic_low[1][2]

        pi = torch.round(p).long()
        
        inside_mask = (
            (pi[0] >= self.cut_bound // downscale_factor)
            * (pi[1] >= self.cut_bound // downscale_factor)
            * (pi[0] < current_image_dim[0] - self.cut_bound // downscale_factor)
            * (pi[1] < current_image_dim[1] - self.cut_bound // downscale_factor)
        )
        if depth is not None:
            depth_low = torch.nn.functional.avg_pool2d(
                depth.unsqueeze(0).unsqueeze(0),
                kernel_size=downscale_factor,
                stride=downscale_factor
            ).squeeze()
            occlusion_mask = torch.abs(depth_low[pi[1][inside_mask], pi[0][inside_mask]] - p[2][inside_mask]) <= self.vis_thres
            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2] > 0
            inside_mask = front_mask * inside_mask

        pi_x_ = pi[1][inside_mask]
        pi_y_ = pi[0][inside_mask]
        pi_depth_ = p[2][inside_mask]
        
        inds = pi_x_ * self.image_dim[0] + pi_y_

        unique_inds, inverse = torch.unique(inds, return_inverse=True)

        depth_min, argmin = torch_scatter.scatter_min(
            pi_depth_.double(), 
            inverse, 
            dim=0
        )

        valid_min_mask = depth_min > 0
        valid_argmin = argmin[valid_min_mask]

        if valid_argmin.numel() > 0:
            local_indices = torch.where(inside_mask)[0][valid_argmin]
            min_depth_coords = coords[local_indices]
            scene_frame_id = f"{id_1}_{id_2}"
            save_min_depth_path = "/data2/wcl/Open3DIS/exp_scannet200/version_Grounded-SAM/vis_projectpoints/false_depth"
            save_min_depth_path_dir = os.path.join(save_min_depth_path, id_1, f"{scene_frame_id}_min.txt")
            os.makedirs(os.path.dirname(save_min_depth_path_dir), exist_ok=True)
            np.savetxt(save_min_depth_path_dir, 
                    min_depth_coords.cpu().numpy(),
                    fmt='%.6f',
                    delimiter=',',
                    header='x,y,z')
            # print(f"已保存{len(min_depth_coords)}个最小深度点到{save_path}")

        valid_min_mask = depth_min > 0
        depth_min = torch.where(valid_min_mask, depth_min, torch.zeros_like(depth_min))
        depth_min_broadcast = depth_min[inverse]
        depth_diff = (pi_depth_ - depth_min_broadcast)

        # THRESHOLD = 0.13
        depth_occlusion_mask = (depth_diff <= threshold) & (depth_diff >= 0)

        new_inside_mask = inside_mask.clone()
        new_inside_mask[inside_mask] = depth_occlusion_mask
        # 在depth_occlusion_mask计算之后，保存过滤后的点云
        filtered_coords = coords[new_inside_mask].cpu().numpy()
        if filtered_coords.shape[0] > 0:
            save_min_filter_path = "/data2/wcl/Open3DIS/exp_scannet200/version_Grounded-SAM/vis_projectpoints/false_depth"  # 可修改保存路径
            save_min_filter_path_dir = os.path.join(save_min_filter_path, id_1, f"{scene_frame_id}_filter.txt")
            os.makedirs(os.path.dirname(save_min_filter_path_dir), exist_ok=True)
            np.savetxt(save_min_filter_path_dir, 
                    filtered_coords,
                    fmt='%.6f',  # 保留6位小数
                    delimiter=',',
                    header='x,y,z')
            # print(f"已保存{len(filtered_coords)}个过滤后的点到{save_path}")

        mapping[0][new_inside_mask] = pi[1][new_inside_mask] * downscale_factor
        mapping[1][new_inside_mask] = pi[0][new_inside_mask] * downscale_factor
        mapping[2][new_inside_mask] = 1

        return mapping.T

    def compute_mapping_torch_rect200_multi(self, camera_to_world, coords, depth=None, intrinsic=None, vis_thresh=None, 
                                               downscale_factors=[66,62,58,54], id_1=None, id_2=None):
        """
        :param downscale_factor: 图像降采样倍数
        其他参数与原函数一致
        """
        device = coords.device
        mappings = []
        scene_save_path = "/data2/wcl/Open3DIS/exp_scannet200/version_Grounded-SAM/vis_projectpoints/false_depth"
        scene_save_path_dir = os.path.join(scene_save_path, id_1, f"{id_1}.txt")
        os.makedirs(os.path.dirname(scene_save_path_dir), exist_ok=True)
        np.savetxt(scene_save_path_dir,
                coords.cpu().numpy(),
                fmt='%.6f',
                delimiter=',',
                header='x,y,z')
        # print(f"已保存场景点云{coords.shape[0]}个点到{scene_save_path}")
        
        thresholds = [0.16 - i*0.02 for i in range(len(downscale_factors))]
        for downscale_factor, threshold in zip(downscale_factors, thresholds):
            mapping = self.compute_mapping_for_multi(camera_to_world, coords, depth, intrinsic, vis_thresh, 
                                                     downscale_factor, threshold, id_1, id_2)
            mappings.append(mapping)

        final_mapping = torch.zeros((coords.shape[0], 3), dtype=torch.long, device=device)
        for mapping in mappings:
            visible_mask = mapping[:, 2] == 1
            final_mapping[visible_mask] = mapping[visible_mask]

        return final_mapping



        # # 原始分辨率可见性掩码
        # inside_mask_original = (
        #     (pi_original[0] >= self.cut_bound) &
        #     (pi_original[1] >= self.cut_bound) &
        #     (pi_original[0] < original_image_dim[0] - self.cut_bound) &
        #     (pi_original[1] < original_image_dim[1] - self.cut_bound)
        # )

        # # 获取所有原始分辨率可见点
        # original_visible_indices = torch.where(inside_mask_original)[0]
        # all_depths = p_original[2][original_visible_indices]

        # # 获取多分辨率可见点深度
        # visible_depths = p_original[2][unique_visible_indices]

        # # 使用广播机制计算深度差异矩阵
        # depth_diff = torch.abs(
        #     all_depths.unsqueeze(1) -  # [N_original, 1]
        #     visible_depths.unsqueeze(0)  # [1, N_visible]
        # )  # -> [N_original, N_visible]

        # # 找到每个点的最小深度差异
        # min_diff = torch.min(depth_diff, dim=1)[0]

        # # 应用阈值过滤
        # valid_mask = (min_diff <= THRESHOLD) & (all_depths >= 0)
        # final_visible = original_visible_indices[valid_mask]

        # # 更新最终映射
        # final_mapping[final_visible, 0] = pi_original[1][final_visible]
        # final_mapping[final_visible, 1] = pi_original[0][final_visible]
        # final_mapping[final_visible, 2] = 1

        # # 保存过滤后的点云
        # filtered_coords = coords[final_visible].cpu().numpy()
        # if filtered_coords.shape[0] > 0:
        #     scene_frame_id = f"{id_1}_{id_2}"
        #     save_path = os.path.join(
        #         "/data2/wcl/Open3DIS/exp_scannet200/version_Grounded-SAM/vis_projectpoints/false_depth",
        #         id_1, 
        #         f"{scene_frame_id}_filter.txt"
        #     )
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #     np.savetxt(
        #         save_path,
        #         filtered_coords,
        #         fmt='%.6f',
        #         delimiter=',',
        #         header='x,y,z'
        #     )

        # return final_mapping

        
    def compute_mapping_torch_rect(self, camera_to_world, coords, depth=None, intrinsic=None, vis_thresh=None, downscale_factor=8):
        """
        :param downscale_factor: 图像降采样倍数
        其他参数与原函数一致
        """
        device = coords.device
        if vis_thresh is not None:
            self.vis_thres = vis_thresh
        if intrinsic is not None:
            self.intrinsics = intrinsic
        else:
            intrinsic = self.intrinsics
        
        start_mem = psutil.Process().memory_info().rss
        start_time = time.time()

        # ========== 新增：分辨率降采样处理 ==========
        # 保存原始图像尺寸
        original_image_dim = self.image_dim
        # 计算降采样后的尺寸
        current_image_dim = (original_image_dim[0] // downscale_factor, 
                        original_image_dim[1] // downscale_factor)
        # 调整内参矩阵（适应降采样）
        intrinsic_low = intrinsic.clone() if torch.is_tensor(intrinsic) else intrinsic.copy()
        intrinsic_low[0, 0] /= downscale_factor  # fx
        intrinsic_low[1, 1] /= downscale_factor  # fy
        intrinsic_low[0, 2] /= downscale_factor  # cx
        intrinsic_low[1, 2] /= downscale_factor  # cy
        # ========================================

        camera_to_world = torch.from_numpy(camera_to_world).to(device).double()

        mapping = torch.zeros((3, coords.shape[0]), dtype=torch.long, device=device)
        coords_new = torch.cat([coords.double(), torch.ones([coords.shape[0], 1], dtype=torch.double, device=device)], dim=1).T

        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = torch.linalg.inv(camera_to_world)
        p = world_to_camera.double() @ coords_new.double()

        # ========== 修改：使用降采样后的内参 ==========
        p[0] = (p[0] * intrinsic_low[0][0]) / p[2] + intrinsic_low[0][2]
        p[1] = (p[1] * intrinsic_low[1][1]) / p[2] + intrinsic_low[1][2]
        # ========================================

        pi = torch.round(p).long()
        
        # ========== 修改：使用降采样后的边界条件 ==========
        inside_mask = (
            (pi[0] >= self.cut_bound // downscale_factor)
            * (pi[1] >= self.cut_bound // downscale_factor)
            * (pi[0] < current_image_dim[0] - self.cut_bound // downscale_factor)
            * (pi[1] < current_image_dim[1] - self.cut_bound // downscale_factor)
        )
        # ========================================

        if depth is not None:
            # ========== 新增：深度图降采样 ==========
            depth_low = torch.nn.functional.avg_pool2d(
                depth.unsqueeze(0).unsqueeze(0),
                kernel_size=downscale_factor,
                stride=downscale_factor
            ).squeeze()
            # ====================================
            occlusion_mask = torch.abs(depth_low[pi[1][inside_mask], pi[0][inside_mask]] - p[2][inside_mask]) <= self.vis_thres
            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2] > 0
            inside_mask = front_mask * inside_mask

        # ========== 修改：使用降采样后的图像尺寸 ==========
        pi_x_ = pi[1][inside_mask]
        pi_y_ = pi[0][inside_mask]
        pi_depth_ = p[2][inside_mask]
        
        inds = pi_x_ * self.image_dim[0] + pi_y_
        # ========================================

        unique_inds, inverse = torch.unique(inds, return_inverse=True)

        depth_min, argmin = torch_scatter.scatter_min(
            pi_depth_.double(), 
            inverse, 
            dim=0
        )

        valid_min_mask = depth_min > 0
        valid_argmin = argmin[valid_min_mask]

        if valid_argmin.numel() > 0:
            local_indices = torch.where(inside_mask)[0][valid_argmin]
            min_depth_coords = coords[local_indices]
            
            # 保存路径保持原逻辑
            save_path = "/data3/wcl/Open3DIS/exp_kitti360/YoloW-SAM/vis_projectpoints/depthmin_8_0.6.txt"
            # np.savetxt(save_path, 
            #         min_depth_coords.cpu().numpy(),
            #         fmt='%.6f',
            #         delimiter=',',
            #         header='x,y,z')
            # print(f"已保存{len(min_depth_coords)}个最小深度点到{save_path}")

        valid_min_mask = depth_min > 0
        depth_min = torch.where(valid_min_mask, depth_min, torch.zeros_like(depth_min))
        depth_min_broadcast = depth_min[inverse]
        depth_diff = (pi_depth_ - depth_min_broadcast)

        THRESHOLD = 0.6
        depth_occlusion_mask = (depth_diff <= THRESHOLD) & (depth_diff >= 0)

        new_inside_mask = inside_mask.clone()
        new_inside_mask[inside_mask] = depth_occlusion_mask
        # 在depth_occlusion_mask计算之后，保存过滤后的点云
        filtered_coords = coords[new_inside_mask].cpu().numpy()
        if filtered_coords.shape[0] > 0:
            save_path = "/data3/wcl/Open3DIS/exp_kitti360/YoloW-SAM/vis_projectpoints/filtered_points_8_0.6.txt"  # 可修改保存路径
            # np.savetxt(save_path, 
            #         filtered_coords,
            #         fmt='%.6f',  # 保留6位小数
            #         delimiter=',',
            #         header='x,y,z')
            # print(f"已保存{len(filtered_coords)}个过滤后的点到{save_path}")

        mapping[0][new_inside_mask] = pi[1][new_inside_mask] * downscale_factor
        mapping[1][new_inside_mask] = pi[0][new_inside_mask] * downscale_factor
        mapping[2][new_inside_mask] = 1

        end_mem = psutil.Process().memory_info().rss
        end_time = time.time()
        memory_used = (end_mem - start_mem) / (1024 ** 2)
        time_used = end_time - start_time
        print(f"Z-buffer 方法 - 内存使用: {memory_used:.2f} MB, 时间: {time_used:.4f} 秒")

        return mapping.T



    def compute_mapping(self, camera_to_world, coords, depth=None, intrinsic=None):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        if intrinsic is not None: # adjust intrinsic
            self.intrinsics = intrinsic
        else:
            intrinsic = self.intrinsics
            
        mapping = np.zeros((3, coords.shape[0]), dtype=int)
        coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = np.round(p).astype(np.int32)  # simply round the projected coordinates
        inside_mask = (
            (pi[0] >= self.cut_bound)
            * (pi[1] >= self.cut_bound)
            * (pi[0] < self.image_dim[0] - self.cut_bound)
            * (pi[1] < self.image_dim[1] - self.cut_bound)
        )
        if depth is not None:
            depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
            occlusion_mask = (
                np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]] - p[2][inside_mask]) <= self.vis_thres * depth_cur
            )
            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2] > 0  # make sure the depth is in front
            inside_mask = front_mask * inside_mask

        # NOTE detect occlusion
        pi_x_ = pi[1][inside_mask]
        pi_y_ = pi[0][inside_mask]
        pi_depth_ = pi[2][inside_mask]

        inds = (pi_x_ * self.image_dim[0] + pi_y_).astype(np.int32)
        _, inds = np.unique(inds, return_inverse=True)

        depth_min = torch_scatter.scatter_min(
            torch.from_numpy(pi_depth_).float(), torch.from_numpy(inds).long(), dim=0
        )[0]
        depth_min = torch.where(depth_min < 0.0, 0.0, depth_min)
        depth_min = depth_min.numpy()
        depth_min_broadcast = depth_min[inds]

        THRESHOLD = 0.2  # (meter)
        depth_occlusion_mask = (pi_depth_ - depth_min_broadcast) <= THRESHOLD

        new_inside_mask = inside_mask.copy()
        new_inside_mask[inside_mask] = depth_occlusion_mask
        ############################

        mapping[0][new_inside_mask] = pi[1][new_inside_mask]
        mapping[1][new_inside_mask] = pi[0][new_inside_mask]
        mapping[2][new_inside_mask] = 1

        return mapping.T
