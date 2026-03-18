import os
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from detectron2.structures import BitMasks
from tqdm import trange

from open3dis.dataset_outdoor import build_dataset
from open3dis.src.mapper_zbuffer import PointCloudToImageMapper
from util2d.openai_clip import CLIP_OpenAI
from util2d.util import masks_to_rle


class BaseMaskStpls3d:
    def __init__(self, cfg):
        clip2d = CLIP_OpenAI(cfg)
        self.clip_adapter = clip2d.clip_adapter
        self.clip_preprocess = clip2d.clip_preprocess
        self.score_name = "score"
        requested_device = getattr(cfg.foundation_model, "device", "cuda")
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
        self.device = device

    def gen_grounded_mask_and_feat(self, scene_id, class_names, cfg, gen_feat=False):
        """
        Generate 2D masks with an optional second SAM3 point-prompt completion stage.
        Returns stage outputs as:
            {
              "stage1": {frame_id: {...}},
              "stage2": {frame_id: {...}},
              "final":  {frame_id: {...}},
            }
        """
        scene_dir = os.path.join(cfg.data.datapath, scene_id)
        loader = build_dataset(root_path=scene_dir, cfg=cfg)
        pointcloud_mapper = PointCloudToImageMapper(
            image_dim=cfg.data.img_dim,
            cut_bound=cfg.data.cut_num_pixel_boundary,
        )

        points = torch.from_numpy(loader.read_pointcloud()).to(self.device)
        n_points = points.shape[0]
        grounded_features = (
            torch.zeros((n_points, cfg.foundation_model.clip_dim), device=self.device)
            if gen_feat
            else None
        )

        stage_outputs = {
            "stage1": {},
            "stage2": {},
            "final": {},
        }
        stage2_scene = self._load_stage2_scene_context(scene_id, loader, cfg) if self._stage2_enabled(cfg) else None

        for i in trange(0, len(loader), cfg.data.img_interval):
            frame = loader[i]
            frame_id = frame["frame_id"]
            image_path = frame["image_path"]

            image_pil = Image.open(image_path).convert("RGB")
            image_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            valid_mask = loader.read_valid_mask(frame.get("valid_mask_path"))
            valid_meta = loader.read_frame_meta(frame.get("meta_path"))
            valid_region = self._prepare_valid_region(
                valid_mask=valid_mask,
                image_shape=image_rgb.shape[:2],
                cfg=cfg,
            )

            crop_box = self._compute_valid_crop(
                valid_region=valid_region,
                cfg=cfg,
                valid_meta=valid_meta,
            )
            x0, y0, x1, y1 = crop_box
            cropped_rgb = image_rgb[y0:y1, x0:x1, :]
            cropped_pil = Image.fromarray(cropped_rgb)

            stage1_crop_mask_dicts = self.generate_crop_masks(
                image_pil=cropped_pil,
                image_rgb=cropped_rgb,
                class_names=class_names,
                cfg=cfg,
            )
            stage1_mask_dicts = self._restore_masks_to_full_image(
                mask_dicts=stage1_crop_mask_dicts,
                crop_box=crop_box,
                full_shape=image_rgb.shape[:2],
            )
            stage1_mask_dicts = self._postprocess_mask_dicts(
                mask_dicts=stage1_mask_dicts,
                image_rgb=image_rgb,
                valid_mask=valid_region,
                cfg=cfg,
            )

            stage2_mask_dicts = []
            mapping = None
            if stage2_scene is not None:
                pose = loader.read_pose(frame["pose_path"])
                depth = loader.read_depth(frame["depth_path"])
                intrinsic = loader.read_intrinsic(frame["intrinsic_path"])
                mapping = self._compute_frame_mapping(
                    pointcloud_mapper=pointcloud_mapper,
                    points=points,
                    pose=pose,
                    depth=depth,
                    intrinsic=intrinsic,
                    image_dim=image_rgb.shape[:2],
                )
                stage2_candidates = self._build_stage2_prompt_candidates(
                    mapping=mapping,
                    coarse_spp=stage2_scene["spp"],
                    coarse_spp_counts=stage2_scene["counts"],
                    image_shape=image_rgb.shape[:2],
                    crop_box=crop_box,
                    valid_region=valid_region,
                    stage1_mask_dicts=stage1_mask_dicts,
                    cfg=cfg,
                )
                if stage2_candidates:
                    prompt_points_crop = [candidate["point_crop"] for candidate in stage2_candidates]
                    stage2_crop_mask_dicts = self.generate_point_prompt_masks(
                        image_pil=cropped_pil,
                        point_prompts=prompt_points_crop,
                        cfg=cfg,
                    )
                    stage2_mask_dicts = self._restore_masks_to_full_image(
                        mask_dicts=stage2_crop_mask_dicts,
                        crop_box=crop_box,
                        full_shape=image_rgb.shape[:2],
                    )
                    stage2_mask_dicts = self._postprocess_mask_dicts(
                        mask_dicts=stage2_mask_dicts,
                        image_rgb=image_rgb,
                        valid_mask=valid_region,
                        resolve_overlaps=False,
                        cfg=cfg,
                    )
                    stage2_mask_dicts = self._filter_stage2_mask_dicts(
                        stage2_mask_dicts=stage2_mask_dicts,
                        stage1_mask_dicts=stage1_mask_dicts,
                        stage2_candidates=stage2_candidates,
                        cfg=cfg,
                    )

            final_mask_dicts = self._merge_stage_mask_dicts(
                stage1_mask_dicts=stage1_mask_dicts,
                stage2_mask_dicts=stage2_mask_dicts,
            )

            stage1_record, stage1_pack = self._build_frame_output(
                mask_dicts=stage1_mask_dicts,
                image_rgb=image_rgb,
                include_img_feat=False,
            )
            stage2_record, stage2_pack = self._build_frame_output(
                mask_dicts=stage2_mask_dicts,
                image_rgb=image_rgb,
                include_img_feat=False,
            )
            final_record, final_pack = self._build_frame_output(
                mask_dicts=final_mask_dicts,
                image_rgb=image_rgb,
                include_img_feat=True,
            )

            stage_outputs["stage1"][frame_id] = stage1_record
            stage_outputs["stage2"][frame_id] = stage2_record
            stage_outputs["final"][frame_id] = final_record

            if stage2_scene is None:
                self._save_visualizations(
                    scene_id=scene_id,
                    frame_id=frame_id,
                    image_pil=image_pil,
                    image_rgb=image_rgb,
                    masks=final_pack["masks"],
                    boxes_filt=final_pack["boxes"],
                    confs_filt=final_pack["conf"],
                    cfg=cfg,
                    vis_name=None,
                )
            else:
                self._save_visualizations(
                    scene_id=scene_id,
                    frame_id=frame_id,
                    image_pil=image_pil,
                    image_rgb=image_rgb,
                    masks=stage1_pack["masks"],
                    boxes_filt=stage1_pack["boxes"],
                    confs_filt=stage1_pack["conf"],
                    cfg=cfg,
                    vis_name="stage1",
                )
                self._save_visualizations(
                    scene_id=scene_id,
                    frame_id=frame_id,
                    image_pil=image_pil,
                    image_rgb=image_rgb,
                    masks=stage2_pack["masks"],
                    boxes_filt=stage2_pack["boxes"],
                    confs_filt=stage2_pack["conf"],
                    cfg=cfg,
                    vis_name="stage2",
                )
                self._save_visualizations(
                    scene_id=scene_id,
                    frame_id=frame_id,
                    image_pil=image_pil,
                    image_rgb=image_rgb,
                    masks=final_pack["masks"],
                    boxes_filt=final_pack["boxes"],
                    confs_filt=final_pack["conf"],
                    cfg=cfg,
                    vis_name="final",
                )

            if gen_feat and final_pack["masks"] is not None and final_pack["img_feat"] is not None:
                if mapping is None:
                    pose = loader.read_pose(frame["pose_path"])
                    depth = loader.read_depth(frame["depth_path"])
                    intrinsic = loader.read_intrinsic(frame["intrinsic_path"])
                    mapping = self._compute_frame_mapping(
                        pointcloud_mapper=pointcloud_mapper,
                        points=points,
                        pose=pose,
                        depth=depth,
                        intrinsic=intrinsic,
                        image_dim=image_rgb.shape[:2],
                    )

                idx = torch.where(mapping[:, 3] == 1)[0]
                if len(idx) >= 100:
                    pred_masks = BitMasks(final_pack["masks"].squeeze(1))
                    final_feat = torch.einsum(
                        "qc,qhw->chw",
                        final_pack["img_feat"].float(),
                        pred_masks.tensor.float(),
                    )
                    grounded_features[idx] += final_feat[
                        :, mapping[idx, 1], mapping[idx, 2]
                    ].permute(1, 0)

        return stage_outputs, grounded_features

    def generate_crop_masks(self, image_pil, image_rgb, class_names, cfg):
        raise NotImplementedError

    def generate_point_prompt_masks(self, image_pil, point_prompts, cfg):
        raise NotImplementedError("This backend does not implement point-prompt completion.")

    def _stage2_enabled(self, cfg):
        return bool(getattr(cfg.foundation_model, "sam3_stage2_enable", False))

    def _build_frame_output(self, mask_dicts, image_rgb, include_img_feat):
        masks, confs_filt, boxes_filt = self._mask_dicts_to_tensors(mask_dicts)
        if masks is None:
            return {
                "masks": [],
                "conf": torch.zeros((0,), dtype=torch.float32),
            }, {
                "masks": None,
                "conf": None,
                "boxes": None,
                "img_feat": None,
            }

        record = {
            "masks": masks_to_rle(masks),
            "conf": confs_filt.detach().cpu(),
        }
        image_features = None
        if include_img_feat:
            image_features = self._extract_clip_features(
                image_rgb=image_rgb,
                masks=masks,
                boxes_filt=boxes_filt,
            )
            record["img_feat"] = image_features.detach().cpu()

        return record, {
            "masks": masks,
            "conf": confs_filt,
            "boxes": boxes_filt,
            "img_feat": image_features,
        }

    def _load_stage2_scene_context(self, scene_id, loader, cfg):
        spp_root = getattr(cfg.foundation_model, "sam3_stage2_spp_path", None)
        if not spp_root:
            return None

        spp_path = os.path.join(spp_root, f"{scene_id}.pth")
        if not os.path.exists(spp_path):
            print(f"[SAM3][stage2] Missing coarse spp labels for {scene_id}: {spp_path}")
            return None

        coarse_spp = loader.read_spp(spp_path, device=str(self.device))
        if coarse_spp.numel() == 0:
            return None

        _, coarse_spp = torch.unique(coarse_spp.to(torch.long), sorted=True, return_inverse=True)
        coarse_spp_counts = torch.bincount(coarse_spp, minlength=int(coarse_spp.max().item()) + 1).to(torch.float32)
        return {
            "spp": coarse_spp,
            "counts": coarse_spp_counts.to(self.device),
        }

    def _compute_frame_mapping(
        self,
        *,
        pointcloud_mapper,
        points,
        pose,
        depth,
        intrinsic,
        image_dim,
    ):
        device = points.device
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose).to(device).float()
        else:
            pose = pose.to(device).float()

        if isinstance(intrinsic, np.ndarray):
            intrinsic = torch.from_numpy(intrinsic).to(device).float()
        else:
            intrinsic = intrinsic.to(device).float()
        if intrinsic.shape == (4, 4):
            intrinsic = intrinsic[:3, :3]

        depth = torch.as_tensor(depth, device=device)
        h, w = depth.shape[-2:]

        mapping = torch.ones((points.shape[0], 4), dtype=torch.long, device=device)
        coords_new = torch.cat(
            [points, torch.ones((points.shape[0], 1), dtype=torch.float32, device=device)],
            dim=1,
        ).T
        world_to_camera = torch.linalg.inv(pose)
        proj = world_to_camera @ coords_new.float()
        proj_x = (proj[0] * intrinsic[0, 0]) / proj[2].clamp_min(1e-8) + intrinsic[0, 2]
        proj_y = (proj[1] * intrinsic[1, 1]) / proj[2].clamp_min(1e-8) + intrinsic[1, 2]
        pi_x = torch.round(proj_x).long()
        pi_y = torch.round(proj_y).long()

        base_mask = (
            (pi_x >= pointcloud_mapper.cut_bound)
            & (pi_y >= pointcloud_mapper.cut_bound)
            & (pi_x < w - pointcloud_mapper.cut_bound)
            & (pi_y < h - pointcloud_mapper.cut_bound)
        )
        front_mask = proj[2] > 0
        valid_mask = base_mask & front_mask

        occ_ok = torch.zeros_like(valid_mask, dtype=torch.bool)
        if valid_mask.any():
            depth_values = depth[pi_y[valid_mask], pi_x[valid_mask]]
            z_cam_vals = proj[2][valid_mask]
            if getattr(pointcloud_mapper, "visibility_mode", "dynamic") == "dynamic":
                tol_vals = torch.clamp(
                    float(getattr(pointcloud_mapper, "dynamic_vis_scale", 0.01)) * z_cam_vals,
                    min=float(getattr(pointcloud_mapper, "dynamic_vis_min", 0.05)),
                )
                occ_ok[valid_mask] = torch.abs(depth_values - z_cam_vals) <= tol_vals
            else:
                occ_ok[valid_mask] = torch.abs(depth_values - z_cam_vals) <= float(pointcloud_mapper.vis_thres)

        inside_mask = valid_mask & occ_ok
        mapping[:, 1] = pi_y
        mapping[:, 2] = pi_x
        mapping[:, 3] = inside_mask.long()
        return mapping

    def _build_stage2_prompt_candidates(
        self,
        *,
        mapping,
        coarse_spp,
        coarse_spp_counts,
        image_shape,
        crop_box,
        valid_region,
        stage1_mask_dicts,
        cfg,
    ):
        visible_ratio_thresh = float(
            getattr(cfg.foundation_model, "sam3_stage2_visible_ratio_thresh", 0.3)
        )
        min_uncovered_pixels = int(
            getattr(cfg.foundation_model, "sam3_stage2_min_uncovered_pixels", 200)
        )
        max_prompts = int(
            getattr(cfg.foundation_model, "sam3_stage2_max_prompts_per_frame", 64)
        )
        dilate_kernel = int(
            getattr(cfg.foundation_model, "sam3_stage2_region_dilate", 5)
        )
        close_kernel = int(
            getattr(cfg.foundation_model, "sam3_stage2_region_close", dilate_kernel)
        )

        x0, y0, x1, y1 = crop_box
        stage1_union = self._mask_union(stage1_mask_dicts, image_shape)
        visible_idx = torch.where(mapping[:, 3] == 1)[0]
        if visible_idx.numel() == 0:
            return []

        visible_spp = coarse_spp[visible_idx]
        unique_spp, visible_counts = torch.unique(visible_spp, return_counts=True)
        visible_ratio = visible_counts.to(torch.float32) / coarse_spp_counts[unique_spp].clamp_min(1.0)
        keep_unique = unique_spp[visible_ratio >= visible_ratio_thresh]
        keep_ratio = visible_ratio[visible_ratio >= visible_ratio_thresh]
        if keep_unique.numel() == 0:
            return []

        visible_rows = mapping[visible_idx, 1].detach().cpu().numpy()
        visible_cols = mapping[visible_idx, 2].detach().cpu().numpy()
        visible_spp_np = visible_spp.detach().cpu().numpy()
        keep_unique_np = keep_unique.detach().cpu().numpy()
        keep_ratio_np = keep_ratio.detach().cpu().numpy()

        candidates = []
        for spp_id, ratio in zip(keep_unique_np.tolist(), keep_ratio_np.tolist()):
            spp_mask = visible_spp_np == spp_id
            spp_rows = visible_rows[spp_mask]
            spp_cols = visible_cols[spp_mask]
            if spp_rows.size == 0:
                continue

            row_min = int(spp_rows.min())
            row_max = int(spp_rows.max())
            col_min = int(spp_cols.min())
            col_max = int(spp_cols.max())
            local_h = row_max - row_min + 1
            local_w = col_max - col_min + 1
            if local_h <= 0 or local_w <= 0:
                continue

            local_region = np.zeros((local_h, local_w), dtype=np.uint8)
            local_region[spp_rows - row_min, spp_cols - col_min] = 1

            if dilate_kernel > 1:
                kernel = np.ones((dilate_kernel, dilate_kernel), dtype=np.uint8)
                local_region = cv2.dilate(local_region, kernel, iterations=1)
            if close_kernel > 1:
                kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
                local_region = cv2.morphologyEx(local_region, cv2.MORPH_CLOSE, kernel)

            stage1_slice = stage1_union[row_min : row_max + 1, col_min : col_max + 1]
            valid_slice = valid_region[row_min : row_max + 1, col_min : col_max + 1]
            uncovered = local_region.astype(bool) & valid_slice & (~stage1_slice)
            if int(uncovered.sum()) < min_uncovered_pixels:
                continue

            num_components, labels, stats, _ = cv2.connectedComponentsWithStats(
                uncovered.astype(np.uint8),
                connectivity=8,
            )
            if num_components <= 1:
                continue

            largest_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            component = labels == largest_id
            uncovered_pixels = int(component.sum())
            if uncovered_pixels < min_uncovered_pixels:
                continue

            comp_rows, comp_cols = np.where(component)
            comp_row_min = int(comp_rows.min())
            comp_row_max = int(comp_rows.max()) + 1
            comp_col_min = int(comp_cols.min())
            comp_col_max = int(comp_cols.max()) + 1
            uncovered_mask_local = component[
                comp_row_min:comp_row_max,
                comp_col_min:comp_col_max,
            ].copy()
            uncovered_bbox_full = [
                int(col_min + comp_col_min),
                int(row_min + comp_row_min),
                int(col_min + comp_col_max),
                int(row_min + comp_row_max),
            ]

            distance_map = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
            local_y, local_x = np.unravel_index(np.argmax(distance_map), distance_map.shape)
            full_x = int(col_min + local_x)
            full_y = int(row_min + local_y)

            if full_x < x0 or full_x >= x1 or full_y < y0 or full_y >= y1:
                continue

            candidates.append(
                {
                    "spp_id": int(spp_id),
                    "visible_ratio": float(ratio),
                    "uncovered_pixels": uncovered_pixels,
                    "point_full": [full_x, full_y],
                    "point_crop": [full_x - x0, full_y - y0],
                    "uncovered_bbox_full": uncovered_bbox_full,
                    "uncovered_mask_local": uncovered_mask_local,
                }
            )

        candidates.sort(
            key=lambda item: (item["uncovered_pixels"], item["visible_ratio"]),
            reverse=True,
        )
        if max_prompts > 0:
            candidates = candidates[:max_prompts]
        return candidates

    def _filter_stage2_mask_dicts(self, *, stage2_mask_dicts, stage1_mask_dicts, stage2_candidates, cfg):
        if not stage2_mask_dicts:
            return []

        stage1_union = self._mask_union(
            stage1_mask_dicts,
            stage2_mask_dicts[0]["segmentation"].shape,
        )
        overlap_thresh = float(
            getattr(cfg.foundation_model, "sam3_stage2_overlap_with_stage1_thresh", 0.5)
        )
        nms_iou_thresh = float(
            getattr(cfg.foundation_model, "sam3_stage2_nms_iou_thresh", 0.5)
        )
        uncovered_purity_thresh = float(
            getattr(
                cfg.foundation_model,
                "sam3_stage2_uncovered_purity_thresh",
                getattr(cfg.foundation_model, "sam3_stage2_uncovered_coverage_thresh", 0.5),
            )
        )
        candidate_by_index = {idx: candidate for idx, candidate in enumerate(stage2_candidates)}

        filtered = []
        for mask_dict in stage2_mask_dicts:
            seg = self._to_bool_mask(mask_dict["segmentation"]).copy()
            original_area = int(seg.sum())
            if original_area <= 0:
                continue

            prompt_index = mask_dict.get("prompt_index")
            candidate = candidate_by_index.get(prompt_index)
            if candidate is not None:
                purity = self._compute_stage2_uncovered_purity(
                    segmentation=seg,
                    uncovered_bbox_full=candidate["uncovered_bbox_full"],
                    uncovered_mask_local=candidate["uncovered_mask_local"],
                )
                if purity < uncovered_purity_thresh:
                    continue

            overlap = np.logical_and(seg, stage1_union)
            overlap_ratio = float(overlap.sum()) / float(original_area)
            if overlap_ratio >= overlap_thresh:
                continue

            if np.any(overlap):
                seg = np.logical_and(seg, np.logical_not(stage1_union))
                if not np.any(seg):
                    continue

            point_coords = mask_dict.get("point_coords", []) or []
            if point_coords:
                seg = self._keep_prompt_component(seg, point_coords[0])
                if not np.any(seg):
                    continue

            bbox = self._bbox_from_mask(seg)
            if bbox is None:
                continue

            updated = dict(mask_dict)
            updated["segmentation"] = seg
            updated["bbox"] = bbox
            updated["area"] = int(seg.sum())
            filtered.append(updated)

        if not filtered:
            return []

        filtered.sort(key=lambda item: float(item.get("predicted_iou", 0.0)), reverse=True)
        kept = []
        for mask_dict in filtered:
            keep = True
            for kept_dict in kept:
                if self._mask_iou(mask_dict["segmentation"], kept_dict["segmentation"]) > nms_iou_thresh:
                    keep = False
                    break
            if keep:
                kept.append(mask_dict)
        return kept

    def _merge_stage_mask_dicts(self, *, stage1_mask_dicts, stage2_mask_dicts):
        if not stage2_mask_dicts:
            return list(stage1_mask_dicts)
        return list(stage1_mask_dicts) + list(stage2_mask_dicts)

    def _mask_union(self, mask_dicts, image_shape):
        union = np.zeros(image_shape, dtype=np.bool_)
        for mask_dict in mask_dicts:
            union |= self._to_bool_mask(mask_dict["segmentation"])
        return union

    def _mask_iou(self, mask_a, mask_b):
        inter = np.logical_and(mask_a, mask_b).sum()
        if inter <= 0:
            return 0.0
        union = np.logical_or(mask_a, mask_b).sum()
        if union <= 0:
            return 0.0
        return float(inter) / float(union)

    def _compute_stage2_uncovered_purity(self, *, segmentation, uncovered_bbox_full, uncovered_mask_local):
        x0, y0, x1, y1 = [int(v) for v in uncovered_bbox_full]
        if x1 <= x0 or y1 <= y0:
            return 0.0
        seg_slice = segmentation[y0:y1, x0:x1]
        if seg_slice.shape != uncovered_mask_local.shape:
            return 0.0
        mask_area = int(self._to_bool_mask(segmentation).sum())
        if mask_area <= 0:
            return 0.0
        inter = np.logical_and(seg_slice, uncovered_mask_local).sum()
        return float(inter) / float(mask_area)

    def _keep_prompt_component(self, segmentation, point_xy):
        point_x, point_y = int(point_xy[0]), int(point_xy[1])
        if (
            point_x < 0
            or point_y < 0
            or point_y >= segmentation.shape[0]
            or point_x >= segmentation.shape[1]
            or not segmentation[point_y, point_x]
        ):
            return np.zeros_like(segmentation, dtype=np.bool_)

        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(
            segmentation.astype(np.uint8),
            connectivity=8,
        )
        if num_components <= 1:
            return segmentation.astype(np.bool_)

        component_id = int(labels[point_y, point_x])
        if component_id <= 0:
            return np.zeros_like(segmentation, dtype=np.bool_)

        component = labels == component_id
        if stats[component_id, cv2.CC_STAT_AREA] <= 0:
            return np.zeros_like(segmentation, dtype=np.bool_)
        return component.astype(np.bool_)

    def _prepare_valid_region(self, *, valid_mask, image_shape, cfg):
        height, width = image_shape
        if valid_mask is None:
            return np.ones((height, width), dtype=np.bool_)

        valid_region = np.asarray(valid_mask, dtype=np.bool_)
        if valid_region.shape != (height, width):
            resized = cv2.resize(
                valid_region.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            valid_region = resized.astype(np.bool_)

        close_kernel = int(getattr(cfg.foundation_model, "mask_valid_close", 11))
        dilate_kernel = int(getattr(cfg.foundation_model, "mask_valid_dilate", 5))

        if close_kernel > 1:
            kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
            valid_region = cv2.morphologyEx(valid_region.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(np.bool_)
        if dilate_kernel > 1:
            kernel = np.ones((dilate_kernel, dilate_kernel), dtype=np.uint8)
            valid_region = cv2.dilate(valid_region.astype(np.uint8), kernel, iterations=1).astype(np.bool_)

        return valid_region

    def _compute_valid_crop(self, *, valid_region, cfg, valid_meta=None):
        height, width = valid_region.shape
        crop_pad = int(
            getattr(
                cfg.foundation_model,
                "mask_valid_crop_pad",
                getattr(cfg.foundation_model, "mask_crop_pad", 10),
            )
        )

        bbox = None
        if valid_meta:
            bbox_arr = valid_meta.get("valid_bbox")
            if bbox_arr is not None:
                bbox_arr = np.asarray(bbox_arr).astype(np.int64).reshape(-1)
                if bbox_arr.size >= 4:
                    x0, y0, x1, y1 = [int(v) for v in bbox_arr[:4]]
                    if x1 > x0 and y1 > y0:
                        bbox = (x0, y0, x1, y1)

        if bbox is None:
            if not np.any(valid_region):
                return (0, 0, width, height)
            ys, xs = np.where(valid_region)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

        x0, y0, x1, y1 = bbox
        x0 = max(0, x0 - crop_pad)
        y0 = max(0, y0 - crop_pad)
        x1 = min(width, x1 + crop_pad)
        y1 = min(height, y1 + crop_pad)
        if (x1 - x0) < 2 or (y1 - y0) < 2:
            return (0, 0, width, height)
        return (x0, y0, x1, y1)

    def _restore_masks_to_full_image(self, mask_dicts, crop_box, full_shape):
        if not mask_dicts:
            return []

        x0, y0, x1, y1 = crop_box
        full_height, full_width = full_shape
        restored = []
        for mask_dict in mask_dicts:
            seg_crop = self._to_bool_mask(mask_dict["segmentation"])
            seg_full = np.zeros((full_height, full_width), dtype=np.bool_)
            seg_full[y0:y1, x0:x1] = np.logical_or(
                seg_full[y0:y1, x0:x1],
                seg_crop,
            )

            bbox = mask_dict.get("bbox")
            if bbox is None:
                bbox_full = self._bbox_from_mask(seg_full)
            else:
                bbox_full = [
                    int(bbox[0]) + x0,
                    int(bbox[1]) + y0,
                    int(bbox[2]),
                    int(bbox[3]),
                ]

            point_coords = []
            for point in mask_dict.get("point_coords", []) or []:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    point_coords.append([point[0] + x0, point[1] + y0])

            restored_mask = dict(mask_dict)
            restored_mask["segmentation"] = seg_full
            restored_mask["bbox"] = bbox_full
            restored_mask["area"] = int(seg_full.sum())
            restored_mask["point_coords"] = point_coords
            restored.append(restored_mask)
        return restored

    def _postprocess_mask_dicts(self, mask_dicts, image_rgb, valid_mask, cfg, resolve_overlaps=True):
        if not mask_dicts:
            return []

        min_mask_box_area = int(
            getattr(cfg.foundation_model, "mask_min_box_area", 100)
        )
        area_ratio_max = float(
            getattr(cfg.foundation_model, "mask_area_ratio_max", 0.7)
        )
        valid_overlap_thresh = float(
            getattr(cfg.foundation_model, "mask_valid_overlap_thresh", 0.3)
        )
        clip_to_valid = bool(
            getattr(cfg.foundation_model, "mask_clip_to_valid", True)
        )
        valid_region = np.asarray(valid_mask, dtype=np.bool_)
        if valid_region.shape != image_rgb.shape[:2]:
            valid_region = np.ones(image_rgb.shape[:2], dtype=np.bool_)
        valid_area = int(valid_region.sum()) if np.any(valid_region) else image_rgb.shape[0] * image_rgb.shape[1]

        filtered = []
        for mask_dict in mask_dicts:
            segmentation = self._to_bool_mask(mask_dict["segmentation"])
            if not np.any(segmentation):
                continue

            original_area = int(segmentation.sum())
            overlap = np.logical_and(segmentation, valid_region)
            overlap_area = int(overlap.sum())
            overlap_ratio = float(overlap_area) / float(original_area) if original_area > 0 else 0.0
            if overlap_ratio < valid_overlap_thresh:
                continue
            if clip_to_valid:
                segmentation = overlap
                if not np.any(segmentation):
                    continue

            bbox = mask_dict.get("bbox") or self._bbox_from_mask(segmentation)
            if bbox is None:
                continue

            updated = dict(mask_dict)
            updated["segmentation"] = segmentation
            updated["bbox"] = bbox
            updated["area"] = int(segmentation.sum())
            filtered.append(updated)

        if not filtered:
            return []

        if not resolve_overlaps:
            kept = []
            for mask_dict in filtered:
                bbox = mask_dict["bbox"]
                if bbox[2] * bbox[3] < min_mask_box_area:
                    continue
                if valid_area > 0 and (mask_dict["area"] / valid_area) > area_ratio_max:
                    continue
                kept.append(mask_dict)
            return kept

        areas = [mask["area"] for mask in filtered]
        order = np.argsort(areas)
        kept_masks = []
        kept_segs = []

        for idx in order:
            mask_dict = dict(filtered[idx])
            seg = mask_dict["segmentation"].copy()
            for kept_mask, kept_seg in zip(kept_masks, kept_segs):
                overlap = self._bbox_intersection(mask_dict["bbox"], kept_mask["bbox"])
                if overlap is None:
                    continue
                left, top, right, bottom = overlap
                seg[top:bottom, left:right] = np.logical_and(
                    seg[top:bottom, left:right],
                    np.logical_not(kept_seg[top:bottom, left:right]),
                )

            new_area = int(seg.sum())
            if new_area <= 0:
                continue
            if valid_area > 0 and (new_area / valid_area) > area_ratio_max:
                continue

            bbox = self._bbox_from_mask(seg)
            if bbox is None:
                continue
            if bbox[2] * bbox[3] < min_mask_box_area:
                continue

            mask_dict["segmentation"] = seg
            mask_dict["bbox"] = bbox
            mask_dict["area"] = new_area
            kept_masks.append(mask_dict)
            kept_segs.append(seg)

        return kept_masks

    def _mask_dicts_to_tensors(self, mask_dicts):
        masks_list = []
        confs_list = []
        boxes_list = []

        for mask_dict in mask_dicts:
            bbox = mask_dict.get("bbox")
            if bbox is None:
                continue
            x, y, w, h = bbox
            if w * h <= 0:
                continue

            masks_list.append(torch.from_numpy(mask_dict["segmentation"]))
            confs_list.append(float(mask_dict.get("predicted_iou", 0.0)))
            boxes_list.append([x, y, x + w, y + h])

        if not masks_list:
            return None, None, None

        masks = torch.stack(masks_list).unsqueeze(1).to(self.device)
        confs_filt = torch.tensor(confs_list, dtype=torch.float32, device=self.device)
        boxes_filt = torch.tensor(boxes_list, dtype=torch.float32, device=self.device)
        return masks, confs_filt, boxes_filt

    def _extract_clip_features(self, image_rgb, masks, boxes_filt):
        regions = []
        image_tensor = torch.tensor(image_rgb, device=self.device)
        for box_id, box in enumerate(boxes_filt):
            left, top, right, bottom = map(int, box.tolist())
            current_mask = masks[box_id, 0, top:bottom, left:right]
            row, col = torch.where(current_mask == False)
            region = image_tensor[top:bottom, left:right, :].clone()
            region[row, col, 0] = (region[row, col, 0] * 0.5).to(torch.uint8)
            region[row, col, 1] = (region[row, col, 1] * 0.5).to(torch.uint8)
            region[row, col, 2] = (region[row, col, 2] * 0.5).to(torch.uint8)
            regions.append(self.clip_preprocess(Image.fromarray(region.cpu().numpy())))

        imgs = torch.stack(regions).to(self.device)
        image_features = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            for img_batch in torch.split(imgs, 64, dim=0):
                image_feat = self.clip_adapter.encode_image(img_batch)
                image_feat /= image_feat.norm(dim=-1, keepdim=True)
                image_features.append(image_feat)
        return torch.cat(image_features, dim=0)

    def _save_visualizations(
        self,
        scene_id,
        frame_id,
        image_pil,
        image_rgb,
        masks,
        boxes_filt,
        confs_filt,
        cfg,
        vis_name=None,
    ):
        if vis_name is None:
            sambox_name = "color_sambox"
            sammask_name = "color_sammask"
        else:
            sambox_name = f"color_sambox_{vis_name}"
            sammask_name = f"color_sammask_{vis_name}"

        sambox_dir = os.path.join(
            cfg.exp.save_dir,
            cfg.exp.exp_name,
            "gen_2d",
            scene_id,
            sambox_name,
        )
        sammask_dir = os.path.join(
            cfg.exp.save_dir,
            cfg.exp.exp_name,
            "gen_2d",
            scene_id,
            sammask_name,
        )
        os.makedirs(sambox_dir, exist_ok=True)
        os.makedirs(sammask_dir, exist_ok=True)

        image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        if boxes_filt is not None and confs_filt is not None:
            for box, conf in zip(boxes_filt, confs_filt):
                left, top, right, bottom = map(int, box.detach().cpu().numpy())
                cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 2)
                text = f"{self.score_name}: {float(conf):.2f}"
                cv2.putText(
                    image_np,
                    text,
                    (left, bottom + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
        cv2.imwrite(os.path.join(sambox_dir, f"{frame_id}.png"), image_np)

        combined = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if masks is not None:
            mask_vis = np.zeros_like(image_rgb)
            for mask in masks.detach().cpu().numpy():
                mask_vis[mask[0]] = np.random.randint(0, 255, 3).tolist()

            mask_areas = np.any(mask_vis > 0, axis=2)
            combined[mask_areas] = cv2.addWeighted(
                combined[mask_areas],
                0.6,
                mask_vis[mask_areas],
                0.7,
                0,
            )
        cv2.imwrite(os.path.join(sammask_dir, f"{frame_id}.png"), combined)

    def _to_bool_mask(self, segmentation):
        if segmentation.dtype == np.bool_:
            return segmentation
        return segmentation.astype(np.bool_)

    def _bbox_from_mask(self, segmentation):
        ys, xs = np.where(segmentation)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())
        return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

    def _bbox_intersection(self, bbox_a, bbox_b):
        left = max(int(bbox_a[0]), int(bbox_b[0]))
        top = max(int(bbox_a[1]), int(bbox_b[1]))
        right = min(int(bbox_a[0] + bbox_a[2]), int(bbox_b[0] + bbox_b[2]))
        bottom = min(int(bbox_a[1] + bbox_a[3]), int(bbox_b[1] + bbox_b[3]))
        if right <= left or bottom <= top:
            return None
        return left, top, right, bottom
