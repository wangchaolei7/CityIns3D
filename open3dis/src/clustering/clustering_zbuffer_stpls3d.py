import itertools
import math
import os
import random
import time

import cv2
import numpy as np
import open3d as o3d
import pycocotools
import pyviz3d.visualizer as viz
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from open3dis.dataset.scannet200 import INSTANCE_CAT_SCANNET_200
from open3dis.dataset.scannet_loader import ScanNetReader, scaling_mapping
from open3dis.dataset_outdoor import build_dataset
from open3dis.src.clustering.clustering_utils import (
    compute_projected_pts,
    compute_projected_pts_torch,
    compute_relation_matrix_self,
    compute_relation_matrix_self_mem,
    compute_visibility_mask,
    compute_visibility_mask_torch,
    compute_visible_masked_pts,
    compute_visible_masked_pts_torch,
    custom_scatter_mean,
    find_connected_components,
    read_detectron_instances,
    resolve_overlapping_masks,
)
from open3dis.src.fusion_util import NMS_cuda
# from open3dis.src.mapper_zbuffer import PointCloudToImageMapper, generate_depth_from_z_buffer # for kitti360
from open3dis.src.mapper_zbuffer import PointCloudToImageMapper
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import pickle

### ablation study 
from open3dis.src.clustering.geometry_grow_stpls3d import vccs_grow_spp, vccs_growspp_dbscan, vccs_growspp_hdbscan

# ---------------- IO helpers for exporting results ----------------
def _ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _save_lines_txt(path, lines):
    with open(path, 'w') as f:
        for line in lines:
            if isinstance(line, (list, tuple, np.ndarray)):
                f.write(' '.join(map(str, list(line))) + '\n')
            else:
                f.write(str(line) + '\n')


def _save_points_txt(points_t: torch.Tensor, path: str):
    pts = points_t.detach().cpu().numpy()
    # 每行一个点，空格分隔（支持 Nx3 或 NxD 任意维度）
    with open(path, 'w') as f:
        for row in pts:
            f.write(' '.join(map(str, row.tolist())) + '\n')


def _save_instances_multi_line(instances_bool: torch.Tensor, path: str):
    """将实例集合按行写入，每行一个实例的点索引（空格分隔）。"""
    inst = instances_bool.detach().cpu()
    lines = []
    for i in range(inst.shape[0]):
        idx = torch.nonzero(inst[i]).view(-1).tolist()
        lines.append(idx)
    _save_lines_txt(path, lines)


def _save_point_labels(points_t: torch.Tensor, instances_bool: torch.Tensor, dir_path: str, prefix: str = 'final'):
    """
    生成每点的实例ID（-1 表示未分配），并保存：
    - <prefix>_point_labels.txt: 每行一个label
    - <prefix>_points_with_labels.txt: 每行 x y z ... inst_id
    """
    pts = points_t.detach().cpu().numpy()
    inst = instances_bool.detach().cpu()
    n_points = pts.shape[0]
    point_labels = torch.full((n_points,), -1, dtype=torch.int64)
    # 顺序赋值：先出现的实例优先，占用未分配点
    for i in range(inst.shape[0]):
        mask = inst[i]
        unassigned = point_labels == -1
        take = torch.nonzero(mask & unassigned).view(-1)
        if take.numel() > 0:
            point_labels[take] = i

    # 保存标签
    _save_lines_txt(os.path.join(dir_path, f"{prefix}_point_labels.txt"), point_labels.tolist())

    # 保存点+标签（末列为实例ID）
    with open(os.path.join(dir_path, f"{prefix}_points_with_labels.txt"), 'w') as f:
        for j in range(n_points):
            row = list(map(str, pts[j].tolist())) + [str(int(point_labels[j].item()))]
            f.write(' '.join(row) + '\n')
    return point_labels

def hierarchical_agglomerative_clustering_growspp_stpls3d(
    pcd_list,
    left,
    right,
    n_points,
    ious_level,
    level,
    inter,
    point_acc,
    iterative=True,
    points=None,
    spp=None,
    n_spp=None,
    edge_index_spp=None,
    edge_features=None,
    node_features=None,
    centroid=None,
    rgb_mean=None,
    save_intermediate_dir=None,
    spp_total_counts_global=None,
    update_seed_stats=True,
):
    # global num_point, dc_feature_matrix, dc_feature_spp
    '''
    point accumulation:
    view accumulation:
    visibility = point/view
    '''
    if left > right:
        return []
    if left == right:
        device = 'cuda'
        # Graph initialization
        index = left

        if pcd_list[index]["masks"] is None:
            return []
        
        masks = pcd_list[index]["masks"].cuda()
        mapping = pcd_list[index]["mapping"].cuda()
        image_dim_hw = pcd_list[index]["image_dim"]
        frame_tag = pcd_list[index].get("frame_id", index)
        total_spp_points = torch_scatter.scatter((mapping[:, 3] == 1).float(), spp, dim=0, reduce="sum")
        # total_spp_points_visible = torch_scatter.scatter((mapping[:, 3] == 1).float(), spp, dim=0, reduce="sum")

        ### Per mask processing
        mask3d = []
        highlight_indices = set() ### line：1181

        for m, mask in enumerate(masks):
            idx = torch.nonzero(mapping[:, 3] == 1).view(-1)
            highlight_points = idx[
                mask[mapping[idx][:, [1, 2]][:, 0], mapping[idx][:, [1, 2]][:, 1]].nonzero(as_tuple=True)[0]
            ].long()
            highlight_indices.update(highlight_points.cpu().numpy().tolist())

            # 额外保存：图切之前，mask 投影命中的原始点云坐标（每个 mask 单独保存）
            # if save_intermediate_dir is not None:
            #     try:
            #         _ensure_dir(save_intermediate_dir)
            #         if highlight_points.numel() > 0 and points is not None:
            #             _save_points_txt(
            #                 points[highlight_points],
            #                 os.path.join(save_intermediate_dir, f"frame_{frame_tag}_mask_{m}_projected_points_precut.txt"),
            #             )

            #         mask_np = mask.cpu().numpy().astype(np.uint8)
            #         mask_np *= 255
            #         mask_image = Image.fromarray(mask_np, mode='L')
            #         image_save_path = os.path.join(save_intermediate_dir, f"frame_{frame_tag}_mask_{m}_image.png")
            #         mask_image.save(image_save_path)
            #     except Exception:
            #         pass

            sieve_mask = torch.zeros((n_points), device=device)
            sieve_mask[highlight_points] = 1

            num_related_points = torch_scatter.scatter(sieve_mask.float(), spp, dim=0, reduce="sum")
            ##### spp和mask的重叠度iou 原版
            spp_weights = torch.zeros((n_spp), dtype=torch.float32, device=device)
            spp_weights = torch.where(
                total_spp_points==0, 0, num_related_points / total_spp_points
            )
            target_spp = torch.nonzero(spp_weights >= 0.5).view(-1)

            # 新版：增加双约束，提升鲁棒性
            # spp_weights_visible = torch.zeros((n_spp), dtype=torch.float32, device=device)
            # spp_weights_visible = torch.where(
            #     total_spp_points_visible > 0, 
            #     num_related_points / total_spp_points_visible, 
            #     0
            # )
            # mask_constraint_A = (spp_weights_visible >= 0.1)
            
            # # --- 约束B: 全局重叠度或绝对数量 ---
            # # B1: 命中点数 / 超点总点数
            # spp_weights_global = torch.zeros((n_spp), dtype=torch.float32, device=device)
            # spp_weights_global = torch.where(
            #     spp_total_counts_global > 0,
            #     num_related_points / spp_total_counts_global,
            #     0
            # )
            # mask_constraint_B1 = (spp_weights_global >= 0.1)
            
            # # B2: 命中点数
            # mask_constraint_B2 = (num_related_points >= 30)
            
            # # 组合约束B: 满足 B1 或 B2 均可
            # mask_constraint_B = mask_constraint_B1 | mask_constraint_B2
            
            # # --- 最终裁决: 必须同时满足 约束A 和 约束B ---
            # final_mask = mask_constraint_A & mask_constraint_B
            # target_spp = torch.nonzero(final_mask).view(-1)



            ##### spp和mask的重叠度点数
            # target_spp = torch.nonzero(num_related_points > 0).view(-1)
            if len(highlight_points) <= 10:
                continue

            group_tmp = vccs_growspp_dbscan(
            # group_tmp = vccs_growspp_hdbscan(
                highlight_points,
                points,
                spp,
                target_spp,
                edge_index=edge_index_spp,
                node_features=node_features,
                rgb_mean=rgb_mean,
                spp_counts=spp_total_counts_global,
                update_seed_stats=update_seed_stats,
            )

            if isinstance(group_tmp, list):
                if len(group_tmp) == 0:
                    continue
                group_tmp = torch.stack(group_tmp, dim=0)
            elif isinstance(group_tmp, torch.Tensor):
                if group_tmp.dim() == 1:
                    group_tmp = group_tmp.unsqueeze(0)
            else:
                continue

            group_tmp = group_tmp.to(torch.int8)

            point_acc[highlight_points] += 1
            mask3d.append(group_tmp)

            # 保存每个mask在该帧的中间结果：属于该mask的点坐标（原始点云坐标）
            # if save_intermediate_dir is not None:
            #     try:
            #         _ensure_dir(save_intermediate_dir)
            #         idx_member = torch.nonzero(group_tmp.sum(dim=0)).view(-1)
            #         if idx_member.numel() > 0 and points is not None:
            #             pts_member = points[idx_member]
            #             _save_points_txt(
            #                 pts_member,
            #                 os.path.join(save_intermediate_dir, f"frame_{frame_tag}_mask_{m}_points.txt"),
            #             )
            #     except Exception:
            #         pass

        if len(mask3d) == 0:
            return []
        mask3d = torch.cat(mask3d, dim=0)

        # 保存该帧汇总的高亮（投影命中）点坐标（原始点云坐标）
        # if save_intermediate_dir is not None:
        #     try:
        #         _ensure_dir(save_intermediate_dir)
        #         if len(highlight_indices) > 0 and points is not None:
        #             idx_tensor = torch.tensor(sorted(list(highlight_indices)), dtype=torch.long, device=points.device)
        #             pts_highlight = points[idx_tensor]
        #             _save_points_txt(
        #                 pts_highlight,
        #                 os.path.join(save_intermediate_dir, f"frame_{frame_tag}_highlight_points.txt"),
        #             )
        #     except Exception:
        #         pass
        return mask3d

    mid = int((left + right) / 2)
    graph_1_onehot = hierarchical_agglomerative_clustering_growspp_stpls3d(
        pcd_list,
        left,
        mid,
        n_points,
        ious_level,
        level + 1,
        inter,
        point_acc,
        iterative=iterative,
        points=points,
        spp=spp,
        n_spp=n_spp,
        edge_index_spp=edge_index_spp,
        edge_features=edge_features,
        node_features=node_features,
        centroid=centroid,
        rgb_mean=rgb_mean,
        save_intermediate_dir=save_intermediate_dir,
        spp_total_counts_global=spp_total_counts_global,
        update_seed_stats=update_seed_stats,
    )
    graph_2_onehot = hierarchical_agglomerative_clustering_growspp_stpls3d(
        pcd_list,
        mid + 1,
        right,
        n_points,
        ious_level,
        level + 1,
        inter,
        point_acc,
        iterative=iterative,
        points=points,
        spp=spp,
        n_spp=n_spp,
        edge_index_spp=edge_index_spp,
        edge_features=edge_features,
        node_features=node_features,
        centroid=centroid,
        rgb_mean=rgb_mean,
        save_intermediate_dir=save_intermediate_dir,
        spp_total_counts_global=spp_total_counts_global,
        update_seed_stats=update_seed_stats,
    )

    if len(graph_1_onehot) == 0 and len(graph_2_onehot) == 0:
        return []

    if len(graph_1_onehot) == 0:
        return graph_2_onehot

    if len(graph_2_onehot) == 0:
        return graph_1_onehot

    if iterative:
        new_graph = torch.cat([graph_1_onehot, graph_2_onehot], dim=0)

        iou_matrix, _, recall_matrix = compute_relation_matrix_self_mem(new_graph)
        
        #####
        import math
        visi = ious_level[min(int(math.floor(level // inter)), len(ious_level) - 1)]
        adjacency_matrix = (iou_matrix >= visi)
        adjacency_matrix |= (recall_matrix >= 0.98)    

        adjacency_matrix = adjacency_matrix | adjacency_matrix.T
        #####

        # if adjacency_matrix
        if adjacency_matrix.sum() == new_graph.shape[0]:
            return new_graph

        # merge instances based on the adjacency matrix
        
        connected_components = find_connected_components(adjacency_matrix)
        M = len(connected_components)
        merged_instance = torch.zeros((M, graph_2_onehot.shape[1]), dtype=torch.int8, device=graph_2_onehot.device)
        for i, cluster in enumerate(connected_components):
            merged_instance[i] = new_graph[cluster].sum(0)

        new_graph = merged_instance

        print(f'-----正在进行第{left}--{right}帧图像合并-----')
        return new_graph

def hierarchical_agglomerative_clustering_growspp_stpls3d_nogrow(
    pcd_list,
    left,
    right,
    n_points,
    ious_level,
    level,
    inter,
    point_acc,
    iterative=True,
    points=None,
    spp=None,
    n_spp=None,
    edge_index_spp=None,
    edge_features=None,
    node_features=None,
    centroid=None,
    rgb_mean=None,
    save_intermediate_dir=None,
    spp_total_counts_global=None,
):
    '''
    No-grow variant: after 2D mask projection, skip SPP graph-cut growth.
    Directly uses projected hits (highlight_points) to build group_tmp and
    appends to mask3d. All other logic mirrors the original function.
    '''
    if left > right:
        return []
    if left == right:
        device = 'cuda'
        index = left

        if pcd_list[index]["masks"] is None:
            return []

        masks = pcd_list[index]["masks"].cuda()
        mapping = pcd_list[index]["mapping"].cuda()
        image_dim_hw = pcd_list[index]["image_dim"]
        frame_tag = pcd_list[index].get("frame_id", index)
        total_spp_points_visible = torch_scatter.scatter((mapping[:, 3] == 1).float(), spp, dim=0, reduce="sum")

        mask3d = []
        highlight_indices = set()

        for m, mask in enumerate(masks):
            idx = torch.nonzero(mapping[:, 3] == 1).view(-1)
            highlight_points = idx[
                mask[mapping[idx][:, [1, 2]][:, 0], mapping[idx][:, [1, 2]][:, 1]].nonzero(as_tuple=True)[0]
            ].long()
            highlight_indices.update(highlight_points.cpu().numpy().tolist())

            # if save_intermediate_dir is not None:
            #     try:
            #         _ensure_dir(save_intermediate_dir)
            #         if highlight_points.numel() > 0 and points is not None:
            #             _save_points_txt(
            #                 points[highlight_points],
            #                 os.path.join(save_intermediate_dir, f"frame_{frame_tag}_mask_{m}_projected_points_precut.txt"),
            #             )

            #         mask_np = mask.cpu().numpy().astype(np.uint8)
            #         mask_np *= 255
            #         mask_image = Image.fromarray(mask_np, mode='L')
            #         image_save_path = os.path.join(save_intermediate_dir, f"frame_{frame_tag}_mask_{m}_image.png")
            #         mask_image.save(image_save_path)
            #     except Exception:
            #         pass

            sieve_mask = torch.zeros((n_points), device=device)
            sieve_mask[highlight_points] = 1

            num_related_points = torch_scatter.scatter(sieve_mask.float(), spp, dim=0, reduce="sum")

            spp_weights_visible = torch.zeros((n_spp), dtype=torch.float32, device=device)
            spp_weights_visible = torch.where(
                total_spp_points_visible > 0,
                num_related_points / total_spp_points_visible,
                0
            )
            mask_constraint_A = (spp_weights_visible >= 0.5)

            spp_weights_global = torch.zeros((n_spp), dtype=torch.float32, device=device)
            spp_weights_global = torch.where(
                spp_total_counts_global > 0,
                num_related_points / spp_total_counts_global,
                0
            )
            mask_constraint_B1 = (spp_weights_global >= 0.5)
            mask_constraint_B2 = (num_related_points >= 30)
            mask_constraint_B = mask_constraint_B1 | mask_constraint_B2
            final_mask = mask_constraint_A & mask_constraint_B
            target_spp = torch.nonzero(final_mask).view(-1)

            if len(highlight_points) <= 10:
                continue

            # No-grow: directly build group mask from projected hits
            group_points_mask = torch.zeros((n_points,), dtype=torch.int8, device=device)
            if highlight_points.numel() > 0:
                group_points_mask[highlight_points] = 1
            group_tmp = group_points_mask.unsqueeze(0)

            group_tmp = vccs_grow_spp(highlight_points, points, spp, target_spp, dc_feature_spp, dc_feature)
            # prevent returning an empty list
            if isinstance(group_tmp, list):
                if len(group_tmp) > 0:
                    group_tmp = torch.stack(group_tmp, dim=0)
                else:
                    group_tmp = torch.zeros((1, n_points), dtype=torch.int8, device=device)

            point_acc[highlight_points] += 1
            mask3d.append(group_tmp)

            # if save_intermediate_dir is not None:
            #     try:
            #         _ensure_dir(save_intermediate_dir)
            #         idx_member = highlight_points
            #         if idx_member.numel() > 0 and points is not None:
            #             pts_member = points[idx_member]
            #             _save_points_txt(
            #                 pts_member,
            #                 os.path.join(save_intermediate_dir, f"frame_{frame_tag}_mask_{m}_points.txt"),
            #             )
            #     except Exception:
            #         pass

        if len(mask3d) == 0:
            return []
        mask3d = torch.cat(mask3d, dim=0)

        # if save_intermediate_dir is not None:
        #     try:
        #         _ensure_dir(save_intermediate_dir)
        #         if len(highlight_indices) > 0 and points is not None:
        #             idx_tensor = torch.tensor(sorted(list(highlight_indices)), dtype=torch.long, device=points.device)
        #             pts_highlight = points[idx_tensor]
        #             _save_points_txt(
        #                 pts_highlight,
        #                 os.path.join(save_intermediate_dir, f"frame_{frame_tag}_highlight_points.txt"),
        #             )
        #     except Exception:
        #         pass
        return mask3d

    mid = int((left + right) / 2)
    graph_1_onehot = hierarchical_agglomerative_clustering_growspp_stpls3d_nogrow(
        pcd_list,
        left,
        mid,
        n_points,
        ious_level,
        level + 1,
        inter,
        point_acc,
        iterative=iterative,
        points=points,
        spp=spp,
        n_spp=n_spp,
        edge_index_spp=edge_index_spp,
        edge_features=edge_features,
        node_features=node_features,
        centroid=centroid,
        rgb_mean=rgb_mean,
        save_intermediate_dir=save_intermediate_dir,
        spp_total_counts_global=spp_total_counts_global,
    )
    graph_2_onehot = hierarchical_agglomerative_clustering_growspp_stpls3d_nogrow(
        pcd_list,
        mid + 1,
        right,
        n_points,
        ious_level,
        level + 1,
        inter,
        point_acc,
        iterative=iterative,
        points=points,
        spp=spp,
        n_spp=n_spp,
        edge_index_spp=edge_index_spp,
        edge_features=edge_features,
        node_features=node_features,
        centroid=centroid,
        rgb_mean=rgb_mean,
        save_intermediate_dir=save_intermediate_dir,
        spp_total_counts_global=spp_total_counts_global,
    )

    if len(graph_1_onehot) == 0 and len(graph_2_onehot) == 0:
        return []

    if len(graph_1_onehot) == 0:
        return graph_2_onehot

    if len(graph_2_onehot) == 0:
        return graph_1_onehot

    if iterative:
        new_graph = torch.cat([graph_1_onehot, graph_2_onehot], dim=0)

        iou_matrix, _, recall_matrix = compute_relation_matrix_self_mem(new_graph)

        import math
        visi = ious_level[min(int(math.floor(level // inter)), len(ious_level) - 1)]
        adjacency_matrix = (iou_matrix >= visi)
        adjacency_matrix |= (recall_matrix >= 0.95)

        adjacency_matrix = adjacency_matrix | adjacency_matrix.T

        if adjacency_matrix.sum() == new_graph.shape[0]:
            return new_graph

        connected_components = find_connected_components(adjacency_matrix)
        M = len(connected_components)
        merged_instance = torch.zeros((M, graph_2_onehot.shape[1]), dtype=torch.int8, device=graph_2_onehot.device)
        for i, cluster in enumerate(connected_components):
            merged_instance[i] = new_graph[cluster].sum(0)

        new_graph = merged_instance

        # print(f'-----正在进行第{left}--{right}帧图像合并(无增长)-----')
        return new_graph

def process_hierarchical_agglomerative_growspp_stpls3d(scene_id, cfg):
    # global num_instance, num_point, dc_feature_matrix, dc_feature_spp, dc_feature

    visi = cfg.cluster.visi
    simi = cfg.cluster.simi
    reca = cfg.cluster.recall
    iterative = cfg.cluster.iterative if hasattr(cfg.cluster, 'iterative') else True
    update_seed_stats = True
    if hasattr(cfg.cluster, 'update_seed_stats'):
        try:
            update_seed_stats = bool(cfg.cluster.update_seed_stats)
        except Exception:
            update_seed_stats = True

    exp_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name)

    spp_path = os.path.join(cfg.data.spp_path, f"{scene_id}.pt")

    
    mask2d_path = os.path.join(exp_path, cfg.exp.mask2d_output, scene_id + ".pth")

    # dc_feature_path = os.path.join(cfg.data.dc_features_path, scene_id + ".pth")

    scene_dir = os.path.join(cfg.data.datapath, scene_id)
    loader = build_dataset(root_path=scene_dir, cfg=cfg)

    # 测试读取SPP（仅用于获取点数和超点数）
    # spp_1 = loader.read_spp(spp_path)
    # unique_spp, spp_1, num_point = torch.unique(spp_1, return_inverse=True, return_counts=True)
    # n_spp_1 = len(unique_spp)

    img_dim = cfg.data.rgb_img_dim
    pointcloud_mapper = PointCloudToImageMapper(
        image_dim=img_dim, cut_bound=cfg.data.cut_num_pixel_boundary, device='cuda'
    )

    points = loader.read_pointcloud()
    points = torch.from_numpy(points).cuda()

    # 导出目录（默认启用保存）：exp/<exp_name>/txt_exports/<scene_id>/
    exp_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name)
    export_root = os.path.join(exp_path, 'txt_exports', scene_id)
    export_intermediate = os.path.join(export_root, 'intermediate')
    _ensure_dir(export_root)
    # 保存原始点云（逐行空格分隔）
    try:
        _save_points_txt(points, os.path.join(export_root, 'points.txt'))
    except Exception:
        pass
    n_points = points.shape[0]

    # 加载SPP图结构与特征
    spp_feature_dir = getattr(cfg.data, "dc_features_path", None)
    spp_feature_level = 2
    if hasattr(cfg, "graphcut"):
        spp_feature_dir = getattr(cfg.graphcut, "spp_feature_dir", spp_feature_dir)
        spp_feature_level = getattr(cfg.graphcut, "spp_feature_level", spp_feature_level)
    if hasattr(cfg, "superpoint"):
        spp_feature_dir = getattr(cfg.superpoint, "feature_dir", spp_feature_dir)
        spp_feature_level = getattr(cfg.superpoint, "feature_level", spp_feature_level)
    if spp_feature_dir is None:
        raise ValueError("未找到SPP特征目录，请在配置中指定dc_features_path或superpoint.feature_dir")
    pack_path = os.path.join(spp_feature_dir, f"{scene_id}_L{spp_feature_level}.pt")
    pack = torch.load(pack_path, map_location="cpu")

    spp = pack["level0_to_levelL"].to("cuda")
    n_spp = int(pack["num_nodes"]) if "num_nodes" in pack else int(spp.max().item() + 1)

    def _to_cuda_tensor(data, dtype=None):
        if data is None:
            return None
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)
        if dtype is not None:
            data = data.to(dtype)
        return data.to("cuda")

    edge_index_spp = _to_cuda_tensor(pack["edge_index"], dtype=torch.long)  # [2, E]
    # edge features dict: ensure 1D
    edge_features = {}
    for k, v in pack["edge_features"].items():
        tensor_v = torch.as_tensor(v)
        if tensor_v.dim() > 1:
            tensor_v = tensor_v.squeeze(-1)
        edge_features[k] = _to_cuda_tensor(tensor_v, dtype=torch.float32)

    node_features = {k: _to_cuda_tensor(torch.as_tensor(v), dtype=torch.float32) for k, v in pack["node_features"].items()}
    centroid = _to_cuda_tensor(pack["centroid"], dtype=torch.float32)
    rgb_mean = _to_cuda_tensor(pack.get("rgb_mean"), dtype=torch.float32)

    # dc_feature = pack.get("point_features", None)
    # if dc_feature is not None:
    #     dc_feature = dc_feature.to("cuda")

    # dc_feature_spp = pack.get("dc_feature_spp", None)
    # if dc_feature_spp is not None:
    #     dc_feature_spp = dc_feature_spp.to("cuda")

    visibility = torch.zeros((n_points), dtype=torch.int, device='cuda')
    groundedsam_data_dict = torch.load(mask2d_path)

    pcd_list = []
    k_factor = 1
    for i in trange(0, len(loader), cfg.data.img_interval * k_factor):
        frame = loader[i]
        frame_id = frame["frame_id"]
        # FIXME
        masks = None
        if frame_id not in groundedsam_data_dict.keys():
            if 'masks' in groundedsam_data_dict.keys(): # detic scannet200
                index = groundedsam_data_dict['frame_id'].index(frame_id)
                frame_id = index

            else:
                print('skip: ', frame_id)
                continue
        encoded_masks = None
        if masks == None:
            try: # normal grounding2D
                groundedsam_data = groundedsam_data_dict[frame_id]
                encoded_masks = groundedsam_data["masks"]
            except: # old detic
                groundedsam_data = groundedsam_data_dict['masks']
                encoded_masks = groundedsam_data[frame_id]


        pose = loader.read_pose(frame["pose_path"])
        depth = loader.read_depth(frame["depth_path"])
        rgb_img = loader.read_image(frame["image_path"])
        instrinsic = loader.read_intrinsic(frame["intrinsic_path"])

        rgb_img_dim = rgb_img.shape[:2]
        

        if encoded_masks is not None:
            masks = []
            for mask in encoded_masks:
                masks.append(torch.tensor(pycocotools.mask.decode(mask)))
            masks = torch.stack(masks, dim=0).cpu() # cuda fast but OOM
        if masks != None and masks.shape[1:3] != rgb_img_dim: # interpolate to rgb_image
            masks = torch.nn.functional.interpolate(masks.unsqueeze(0).to(torch.float), rgb_img_dim).to(torch.uint8).squeeze(0)
        

        if "kitti360" in cfg.data.dataset_name:
            rgb_intrinsic = frame["global_intrinsic"] 
            rgb_dim_hw = (cfg.data.rgb_img_dim[1], cfg.data.rgb_img_dim[0])
            # if depth is None:
            generated_depth = generate_depth_from_z_buffer(
                points, pose, rgb_intrinsic, rgb_dim_hw, cfg, frame, device='cuda'
            )
            # else:
            #     generated_depth = depth
            mapping = torch.ones([n_points, 4], dtype=int, device=points.device)
            mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(
                pose, points, rgb_dim_hw, generated_depth, intrinsic=rgb_intrinsic
            )
        
        elif "stpls3d" in cfg.data.dataset_name:  # Map on image resolution in Scannetpp only
            rgb_dim_hw = cfg.data.img_dim # use actual image H,W for STPLS3D
            mapping = torch.ones([n_points, 4], dtype=int, device='cuda')
            mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, rgb_dim_hw, depth=depth, intrinsic=instrinsic, id_1=scene_id, id_2=frame_id)

        else:
            raise ValueError(f"Unknown dataset: {cfg.data.dataset_name}")
        

        visibility[mapping[:, 3] == 1] += 1

        # 在处理每一帧时调用
        # visualize_projection(points, mapping, rgb_img, frame["frame_id"])

        dic = {"mapping": mapping.cpu(), "masks": masks, "image_dim": rgb_dim_hw, "frame_id": frame_id}
        pcd_list.append(dic)
    
    proposals_pred = None
    confidence = None
   
    torch.cuda.empty_cache()
    num_instance = 0
    ###
    import math
    level = 0
    # 添加长度检查避免空列表
    if len(pcd_list) < 1:
        print(f"Warning: Empty pcd_list for scene {scene_id}")
        return None, None
        
    # 确保至少有2个元素才能进行分层聚类
    maxlevel = math.log2(max(len(pcd_list) - 1, 1))  # 保证参数>=1

##################⭐⭐⭐⭐⭐##################   
    # ious_level = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9] # 6 ious level
    # ious_level = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # ious_level = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # ious_level = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]  
    # ious_level = [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]
    # ious_level = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  
    # ious_level = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]  
    ious_level = [0.3, 0.4, 0.5]  
    # ious_level = [0.4, 0.5, 0.6]
    # ious_level = [0.2, 0.3, 0.4]

    # 动态调整阈值配置（添加保护条件）
    while len(ious_level) > 1 and (int(maxlevel // len(ious_level)) == 0):
        ious_level = ious_level[1:]  # 安全截断
    
    # 空列表保护
    if not ious_level:
        ious_level = [0.4]  # 设置最小默认阈值
    
    # 安全计算层级间隔（防止除零）
    valid_len = max(len(ious_level), 1)
    inter = int(maxlevel // valid_len)
    point_acc = torch.zeros((n_points), dtype=int)
    spp_total_counts_global = torch.bincount(spp, minlength=n_spp).float()
    if cfg.agglomerative.hierarchical:
        groups = hierarchical_agglomerative_clustering_growspp_stpls3d(
            pcd_list,
            0,
            len(pcd_list) - 1,
            n_points,
            ious_level,
            level,
            inter,
            point_acc,
            iterative=iterative,
            points=points,
            spp=spp,
            n_spp=n_spp,
            edge_index_spp=edge_index_spp,
            edge_features=edge_features,
            node_features=node_features,
            centroid=centroid,
            rgb_mean=rgb_mean,
            save_intermediate_dir=export_intermediate,
            spp_total_counts_global=spp_total_counts_global,
            update_seed_stats=update_seed_stats,
        )
    ###
    if len(groups) == 0:
        return None, None

    groups = groups.to(torch.int64).cpu()
    confidence = torch.ones(groups.shape[0], dtype=torch.float32, device=groups.device)

    proposals_pred = groups[:, :]  # .bool()
    proposals_pred_raw = proposals_pred.clone()

    # 保存分层聚合后的初始实例（可视为中间结果）：改为“每点的实例ID，以及点+标签”
    try:
        _ensure_dir(export_intermediate)
        _save_point_labels(points, (proposals_pred > 0), export_intermediate, prefix='groups_raw')
    except Exception:
        pass
    del groups
    torch.cuda.empty_cache()

# # # # # # # # ⭐ # # # # # # # # # 

    post_filter = 0.2
    # These lines take a lot of memory # achieveing in paper result-> unlock this
    if cfg.cluster.point_visi > 0:
        point_acc = point_acc.cuda()
        visibility -= point_acc
        start = 0
        end = proposals_pred.shape[0]
        inst_visibility = torch.zeros_like(proposals_pred, dtype=torch.float64).cpu()
        bs = 1000
        while(start<end):
            inst_visibility[start:start+bs] = (proposals_pred[start:start+bs] / visibility.clip(min=1e-6)[None, :].cpu().to(torch.float64))
            start += bs
        torch.cuda.empty_cache()    
        proposals_pred[inst_visibility < post_filter] = 0
    else:
        pass    
    proposals_pred = proposals_pred.bool()

    if cfg.cluster.point_visi > 0:
        proposals_pred_final = custom_scatter_mean(
            proposals_pred,
            spp[None, :].expand(len(proposals_pred), -1),
            dim=-1,
            pool=True,
            output_type=torch.float64,
        )
        proposals_pred = (proposals_pred_final >= 0.5)[:, spp]

    ## Valid points
    mask_valid = proposals_pred.sum(1) > cfg.cluster.valid_points
    proposals_pred = proposals_pred[mask_valid].cpu()


    # 最终结果保存：
    # try:
    #     # 1) 每行实例的点索引
    #     _save_instances_multi_line(proposals_pred, os.path.join(export_root, 'final_instances.txt'))
    #     # 2) 每点的实例ID，以及点+标签
    #     _save_point_labels(points, proposals_pred, export_root, prefix='final')
    #     # 3) 置信度（若需要）
    #     if confidence is not None:
    #         _save_lines_txt(os.path.join(export_root, 'final_confidence.txt'), confidence.detach().cpu().tolist())
    # except Exception:
    #     pass
    # 最终结果保存：
    try:
        # 1) 每行实例的点索引
        # _save_instances_multi_line(proposals_pred, os.path.join(export_root, 'final_instances.txt'))
        
        # 2) 每点的实例ID，以及点+标签 —— 捕获返回的标签用于统计
        point_labels_final = _save_point_labels(points, proposals_pred, export_root, prefix='final')
        
        # 3) 统计 -1 标签的点数和比例
        n_points_total = point_labels_final.numel()
        n_unassigned = (point_labels_final == -1).sum().item()
        ratio_unassigned = n_unassigned / n_points_total if n_points_total > 0 else 0.0
        
        print(f"📊 最终结果中未分配点（标签=-1）统计：")
        print(f"   总点数: {n_points_total}")
        print(f"   未分配点数: {n_unassigned}")
        print(f"   占比: {ratio_unassigned:.2%}")
        
        # 4) 置信度（若需要）
        # if confidence is not None:
        #     _save_lines_txt(os.path.join(export_root, 'final_confidence.txt'), confidence.detach().cpu().tolist())
            
    except Exception as e:
        print(f"❌ 保存最终结果时出错: {e}")

    return proposals_pred, confidence
