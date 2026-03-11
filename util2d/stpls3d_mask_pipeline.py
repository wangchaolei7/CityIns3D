import os

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

    def gen_grounded_mask_and_feat(self, scene_id, class_names, cfg, gen_feat=True):
        """
        Generate 2D masks, keep the downstream save/feature flow unchanged,
        and only swap the mask backend.
        """
        scene_dir = os.path.join(cfg.data.datapath, scene_id)
        loader = build_dataset(root_path=scene_dir, cfg=cfg)

        pointcloud_mapper = PointCloudToImageMapper(
            image_dim=cfg.data.img_dim,
            cut_bound=cfg.data.cut_num_pixel_boundary,
        )

        points = torch.from_numpy(loader.read_pointcloud()).cuda()
        n_points = points.shape[0]
        grounded_data_dict = {}
        grounded_features = (
            torch.zeros((n_points, cfg.foundation_model.clip_dim)).cuda()
            if gen_feat
            else None
        )

        for i in trange(0, len(loader), cfg.data.img_interval):
            frame = loader[i]
            frame_id = frame["frame_id"]
            image_path = frame["image_path"]

            image_pil = Image.open(image_path).convert("RGB")
            image_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            crop_box = self._compute_nonwhite_crop(image_rgb, cfg)
            x0, y0, x1, y1 = crop_box
            cropped_rgb = image_rgb[y0:y1, x0:x1, :]
            cropped_pil = Image.fromarray(cropped_rgb)

            mask_dicts = self.generate_crop_masks(
                image_pil=cropped_pil,
                image_rgb=cropped_rgb,
                class_names=class_names,
                cfg=cfg,
            )
            mask_dicts = self._restore_masks_to_full_image(
                mask_dicts=mask_dicts,
                crop_box=crop_box,
                full_shape=image_rgb.shape[:2],
            )
            mask_dicts = self._postprocess_mask_dicts(
                mask_dicts=mask_dicts,
                image_rgb=image_rgb,
                cfg=cfg,
            )
            if not mask_dicts:
                continue

            masks, confs_filt, boxes_filt = self._mask_dicts_to_tensors(mask_dicts)
            if masks is None:
                continue

            image_features = self._extract_clip_features(
                image_rgb=image_rgb,
                masks=masks,
                boxes_filt=boxes_filt,
            )
            grounded_data_dict[frame_id] = {
                "masks": masks_to_rle(masks),
                "img_feat": image_features.cpu(),
                "conf": confs_filt.cpu(),
            }
            self._save_visualizations(
                scene_id=scene_id,
                frame_id=frame_id,
                image_pil=image_pil,
                image_rgb=image_rgb,
                masks=masks,
                boxes_filt=boxes_filt,
                confs_filt=confs_filt,
                cfg=cfg,
            )

            if gen_feat:
                pose = loader.read_pose(frame["pose_path"])
                depth = loader.read_depth(frame["depth_path"])
                intrinsic = loader.read_intrinsic(frame["intrinsic_path"])

                if "stpls3d" not in cfg.data.dataset_name:
                    raise ValueError(f"Unknown dataset: {cfg.data.dataset_name}")

                mapping = torch.ones([n_points, 4], dtype=int, device="cuda")
                mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(
                    pose,
                    points,
                    cfg.data.img_dim,
                    depth=depth,
                    intrinsic=intrinsic,
                    id_1=scene_id,
                    id_2=frame_id,
                )

                idx = torch.where(mapping[:, 3] == 1)[0]
                if len(idx) < 100:
                    continue

                pred_masks = BitMasks(masks.squeeze(1))
                final_feat = torch.einsum(
                    "qc,qhw->chw",
                    image_features.float(),
                    pred_masks.tensor.float(),
                )
                grounded_features[idx] += final_feat[
                    :, mapping[idx, 1], mapping[idx, 2]
                ].permute(1, 0)

        return grounded_data_dict, grounded_features

    def generate_crop_masks(self, image_pil, image_rgb, class_names, cfg):
        raise NotImplementedError

    def _compute_nonwhite_crop(self, image_rgb, cfg):
        height, width = image_rgb.shape[:2]
        white_thresh = int(getattr(cfg.foundation_model, "mask_white_thresh", 250))
        crop_pad = int(getattr(cfg.foundation_model, "mask_crop_pad", 10))
        non_white = np.any(image_rgb < white_thresh, axis=2)
        if not np.any(non_white):
            return (0, 0, width, height)

        ys, xs = np.where(non_white)
        x0 = max(0, int(xs.min()) - crop_pad)
        x1 = min(width, int(xs.max()) + 1 + crop_pad)
        y0 = max(0, int(ys.min()) - crop_pad)
        y1 = min(height, int(ys.max()) + 1 + crop_pad)
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

    def _postprocess_mask_dicts(self, mask_dicts, image_rgb, cfg):
        if not mask_dicts:
            return []

        white_mean_threshold = float(
            getattr(cfg.foundation_model, "mask_white_mean_threshold", 254.0)
        )
        min_mask_box_area = int(
            getattr(cfg.foundation_model, "mask_min_box_area", 100)
        )
        area_ratio_max = float(
            getattr(cfg.foundation_model, "mask_area_ratio_max", 0.7)
        )
        white_thresh = int(getattr(cfg.foundation_model, "mask_white_thresh", 250))

        non_white = np.any(image_rgb < white_thresh, axis=2)
        nonwhite_area = int(non_white.sum()) if np.any(non_white) else image_rgb.shape[0] * image_rgb.shape[1]

        filtered = []
        for mask_dict in mask_dicts:
            segmentation = self._to_bool_mask(mask_dict["segmentation"])
            if not np.any(segmentation):
                continue
            if np.mean(image_rgb[segmentation]) >= white_mean_threshold:
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
            if nonwhite_area > 0 and (new_area / nonwhite_area) > area_ratio_max:
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

        masks = torch.stack(masks_list).unsqueeze(1).cuda()
        confs_filt = torch.tensor(confs_list, dtype=torch.float32, device="cuda")
        boxes_filt = torch.tensor(boxes_list, dtype=torch.float32, device="cuda")
        return masks, confs_filt, boxes_filt

    def _extract_clip_features(self, image_rgb, masks, boxes_filt):
        regions = []
        for box_id, box in enumerate(boxes_filt):
            left, top, right, bottom = map(int, box.tolist())
            current_mask = masks[box_id, 0, top:bottom, left:right]
            row, col = torch.where(current_mask == False)
            region = torch.tensor(image_rgb)[top:bottom, left:right, :].cuda()
            region[row, col, 0] = (region[row, col, 0] * 0.5).to(torch.uint8)
            region[row, col, 1] = (region[row, col, 1] * 0.5).to(torch.uint8)
            region[row, col, 2] = (region[row, col, 2] * 0.5).to(torch.uint8)
            regions.append(self.clip_preprocess(Image.fromarray(region.cpu().numpy())))

        imgs = torch.stack(regions).cuda()
        image_features = []
        with torch.no_grad(), torch.cuda.amp.autocast():
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
    ):
        sambox_dir = os.path.join(
            cfg.exp.save_dir,
            cfg.exp.exp_name,
            "gen_2d",
            scene_id,
            "color_sambox",
        )
        sammask_dir = os.path.join(
            cfg.exp.save_dir,
            cfg.exp.exp_name,
            "gen_2d",
            scene_id,
            "color_sammask",
        )
        os.makedirs(sambox_dir, exist_ok=True)
        os.makedirs(sammask_dir, exist_ok=True)

        image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        for box, conf in zip(boxes_filt, confs_filt):
            left, top, right, bottom = map(int, box.cpu().numpy())
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

        mask_vis = np.zeros_like(image_rgb)
        for mask in masks.cpu().numpy():
            mask_vis[mask[0]] = np.random.randint(0, 255, 3).tolist()

        combined = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
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
