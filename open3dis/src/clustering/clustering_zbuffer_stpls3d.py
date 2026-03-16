import itertools
import math
import os
import random
import time

import cv2
import numpy as np
import open3d as o3d
import pycocotools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
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
from open3dis.src.clustering.geometry_grow_stpls3d import (
    aggregate_spp_features,
    build_spp_adjacency_point_knn,
    build_spp_members,
    compute_proposal_features_from_spp,
    grow_split_clusters_with_utonia,
    split_projected_clusters,
    vccs_grow_spp,
    vccs_growspp_dbscan,
    vccs_growspp_hdbscan,
)

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
    spp_neighbors=None,
    spp_members=None,
    spp_features=None,
    spp_centroids=None,
    save_intermediate_dir=None,
    spp_total_counts_global=None,
    cluster_cfg=None,
    update_seed_stats=True,
):
    '''
    point accumulation:
    view accumulation:
    visibility = point/view
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

        mask3d = []
        enable_splitting = bool(
            getattr(cluster_cfg, "enable_splitting", True) if cluster_cfg is not None else True
        )
        split_method = getattr(cluster_cfg, "split_method", "dbscan") if cluster_cfg is not None else "dbscan"
        split_dbscan_eps = float(
            getattr(cluster_cfg, "split_dbscan_eps", 0.5) if cluster_cfg is not None else 0.5
        )
        split_dbscan_min_samples = int(
            getattr(cluster_cfg, "split_dbscan_min_samples", 50) if cluster_cfg is not None else 50
        )
        split_hdbscan_min_cluster_size = int(
            getattr(cluster_cfg, "split_hdbscan_min_cluster_size", 50) if cluster_cfg is not None else 50
        )
        split_hdbscan_min_samples = int(
            getattr(cluster_cfg, "split_hdbscan_min_samples", 20) if cluster_cfg is not None else 20
        )
        split_cluster_min_points = int(
            getattr(cluster_cfg, "split_cluster_min_points", 50) if cluster_cfg is not None else 50
        )
        grow_feature_threshold = float(
            getattr(cluster_cfg, "grow_feature_threshold", getattr(cluster_cfg, "simi", 0.6))
            if cluster_cfg is not None
            else 0.6
        )
        grow_min_seed_overlap = float(
            getattr(cluster_cfg, "grow_min_seed_overlap", 0.5) if cluster_cfg is not None else 0.5
        )
        grow_use_geometry = bool(
            getattr(cluster_cfg, "grow_use_geometry", False) if cluster_cfg is not None else False
        )
        grow_centroid_dist_threshold = float(
            getattr(cluster_cfg, "grow_centroid_dist_threshold", 1.5)
            if cluster_cfg is not None
            else 1.5
        )

        for m, mask in enumerate(masks):
            idx = torch.nonzero(mapping[:, 3] == 1).view(-1)
            highlight_points = idx[
                mask[mapping[idx][:, [1, 2]][:, 0], mapping[idx][:, [1, 2]][:, 1]].nonzero(as_tuple=True)[0]
            ].long()
            if highlight_points.numel() <= 10:
                continue

            if enable_splitting:
                split_clusters = split_projected_clusters(
                    highlight_points,
                    points,
                    method=split_method,
                    dbscan_eps=split_dbscan_eps,
                    dbscan_min_samples=split_dbscan_min_samples,
                    hdbscan_min_cluster_size=split_hdbscan_min_cluster_size,
                    hdbscan_min_samples=split_hdbscan_min_samples,
                    cluster_min_points=split_cluster_min_points,
                )
            else:
                split_clusters = [highlight_points]
            group_tmp = grow_split_clusters_with_utonia(
                split_clusters,
                n_points=n_points,
                spp=spp,
                spp_members=spp_members,
                spp_neighbors=spp_neighbors,
                spp_features=spp_features,
                spp_centroids=spp_centroids,
                spp_counts=spp_total_counts_global,
                grow_feature_threshold=grow_feature_threshold,
                grow_min_seed_overlap=grow_min_seed_overlap,
                grow_use_geometry=grow_use_geometry,
                grow_centroid_dist_threshold=grow_centroid_dist_threshold,
                grow_update_region_stats=update_seed_stats,
            )

            if not group_tmp:
                continue

            group_tmp = torch.stack(group_tmp, dim=0).to(torch.int8)
            point_acc[highlight_points] += 1
            mask3d.append(group_tmp)

        if len(mask3d) == 0:
            return []
        mask3d = torch.cat(mask3d, dim=0)
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
        spp_neighbors=spp_neighbors,
        spp_members=spp_members,
        spp_features=spp_features,
        spp_centroids=spp_centroids,
        save_intermediate_dir=save_intermediate_dir,
        spp_total_counts_global=spp_total_counts_global,
        cluster_cfg=cluster_cfg,
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
        spp_neighbors=spp_neighbors,
        spp_members=spp_members,
        spp_features=spp_features,
        spp_centroids=spp_centroids,
        save_intermediate_dir=save_intermediate_dir,
        spp_total_counts_global=spp_total_counts_global,
        cluster_cfg=cluster_cfg,
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
        visi = ious_level[min(int(math.floor(level // inter)), len(ious_level) - 1)]
        adjacency_matrix = (iou_matrix >= visi)
        merge_recall = float(
            getattr(cluster_cfg, "recall", 0.98) if cluster_cfg is not None else 0.98
        )
        adjacency_matrix |= (recall_matrix >= merge_recall)
        adjacency_matrix = adjacency_matrix | adjacency_matrix.T

        if bool(getattr(cluster_cfg, "merge_feature_veto", True) if cluster_cfg is not None else True):
            veto_threshold = float(
                getattr(
                    cluster_cfg,
                    "merge_feature_veto_threshold",
                    getattr(cluster_cfg, "simi", 0.3) if cluster_cfg is not None else 0.3,
                )
            )
            proposal_feat = compute_proposal_features_from_spp(
                new_graph.bool(),
                spp,
                n_spp,
                spp_features,
                chunk_size=int(
                    getattr(cluster_cfg, "feature_pool_chunk_size", 64)
                    if cluster_cfg is not None
                    else 64
                ),
            )
            feature_sim = proposal_feat @ proposal_feat.T
            feature_ok = feature_sim >= veto_threshold
            eye = torch.eye(feature_ok.shape[0], dtype=torch.bool, device=feature_ok.device)
            adjacency_matrix &= (feature_ok | eye)

        if adjacency_matrix.sum() == new_graph.shape[0]:
            return new_graph

        connected_components = find_connected_components(adjacency_matrix)
        M = len(connected_components)
        merged_instance = torch.zeros((M, graph_2_onehot.shape[1]), dtype=torch.int8, device=graph_2_onehot.device)
        for i, cluster in enumerate(connected_components):
            merged_instance[i] = new_graph[cluster].sum(0)

        new_graph = merged_instance

        print(f'-----正在进行第{left}--{right}帧图像合并-----')
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
    mask2d_path = os.path.join(exp_path, cfg.exp.mask2d_output, scene_id + ".pth")

    scene_dir = os.path.join(cfg.data.datapath, scene_id)
    loader = build_dataset(root_path=scene_dir, cfg=cfg)

    img_dim = cfg.data.rgb_img_dim
    pointcloud_mapper = PointCloudToImageMapper(
        image_dim=img_dim, cut_bound=cfg.data.cut_num_pixel_boundary, device='cuda'
    )

    points = loader.read_pointcloud()
    points = torch.from_numpy(points).cuda()
    device = points.device

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

    def _resolve_scene_file(base_dirs, scene_id, exts):
        for base_dir in base_dirs:
            if base_dir is None:
                continue
            for ext in exts:
                candidate = os.path.join(base_dir, f"{scene_id}{ext}")
                if os.path.exists(candidate):
                    return candidate
        return None

    spp_dirs = []
    # if hasattr(cfg, "superpoint"):
    #     spp_dirs.append(getattr(cfg.superpoint, "save_dir", None))
    spp_dirs.append(getattr(cfg.data, "spp_path", None))
    spp_file = _resolve_scene_file(spp_dirs, scene_id, [".pth", ".pt"])
    if spp_file is None:
        raise FileNotFoundError(
            f"未找到场景 {scene_id} 的SPP标签，请检查 superpoint.save_dir 或 data.spp_path。"
        )
    spp = loader.read_spp(spp_file, device=str(device)).long()
    unique_spp, spp, _ = torch.unique(spp, return_inverse=True, return_counts=True)
    n_spp = len(unique_spp)

    point_feature_path = os.path.join(cfg.data.point_features_path, f"{scene_id}.pth")
    if not os.path.exists(point_feature_path):
        raise FileNotFoundError(
            f"未找到场景 {scene_id} 的Utonia点特征: {point_feature_path}"
        )
    point_features = loader.read_feature(point_feature_path, device=str(device))
    point_features = F.normalize(point_features.to(torch.float32), dim=1, p=2)
    if point_features.shape[0] != n_points:
        raise ValueError(
            f"点特征数量与点云数量不一致: points={n_points}, features={point_features.shape[0]}"
        )

    spp_features, spp_centroids, spp_total_counts_global = aggregate_spp_features(
        point_features,
        spp,
        points,
        n_spp,
    )
    del point_features
    spp_members = build_spp_members(spp, n_spp)
    spp_neighbors = build_spp_adjacency_point_knn(
        points,
        spp,
        n_spp,
        k=int(getattr(cfg.cluster, "spp_graph_k", 8)),
        max_neighbor_dist=getattr(cfg.cluster, "spp_graph_max_neighbor_dist", None),
    )
    total_edges = sum(len(neighbors) for neighbors in spp_neighbors) // 2
    print(
        f"[Utonia-SPP] scene={scene_id} n_spp={n_spp} "
        f"graph_mode=point_knn edges={total_edges}"
    )

    visibility = torch.zeros((n_points,), dtype=torch.int32, device=device)
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
            mapping = torch.ones([n_points, 4], dtype=int, device=device)
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
    point_acc = torch.zeros((n_points,), dtype=torch.int32, device=device)
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
            spp_neighbors=spp_neighbors,
            spp_members=spp_members,
            spp_features=spp_features,
            spp_centroids=spp_centroids,
            save_intermediate_dir=export_intermediate,
            spp_total_counts_global=spp_total_counts_global,
            cluster_cfg=cfg.cluster,
            update_seed_stats=update_seed_stats,
        )
    ###
    if len(groups) == 0:
        return None, None

    groups = groups.to(torch.int64)
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
        visibility = visibility - point_acc
        visibility_safe = visibility.clamp(min=1).to(torch.float32)
        bs = 512
        start = 0
        end = proposals_pred.shape[0]
        while start < end:
            batch_end = min(start + bs, end)
            batch_visibility = proposals_pred[start:batch_end].to(torch.float32) / visibility_safe.unsqueeze(0)
            batch_keep = batch_visibility >= post_filter
            proposals_pred[start:batch_end] = proposals_pred[start:batch_end] * batch_keep.to(proposals_pred.dtype)
            start = batch_end
        torch.cuda.empty_cache()
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
        # custom_scatter_mean() 当前按 batch 在 GPU 上聚合、再回收到 CPU 以避免 OOM，
        # 所以后续这里也保持在 CPU 上完成索引，避免 CPU/GPU 混用报错。
        proposals_pred = (proposals_pred_final >= 0.5)[:, spp.cpu()]

    ## Valid points
    mask_valid = proposals_pred.sum(1) > cfg.cluster.valid_points
    proposals_pred = proposals_pred[mask_valid].cpu()
    confidence = confidence.cpu()[mask_valid.cpu()]


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
