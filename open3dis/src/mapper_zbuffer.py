import os
import re

import numpy as np
import torch
# import torch_scatter
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
            frame_save_path = "/data1/wangcl/project/CityIns3D/stpls3d/version_SAM/mapping_vispoints"  # 可修改保存路径
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


