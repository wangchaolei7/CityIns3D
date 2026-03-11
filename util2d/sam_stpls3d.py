import torch
import numpy as np
import os
import cv2
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random

#### Foundation 2D
import clip
from util2d.openai_clip import CLIP_OpenAI
from util2d.segment_anything_hq import SAM_HQ

#### Grounding DINO
from detectron2.structures import BitMasks
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

#### Util
from util2d.util import show_mask, masks_to_rle 

#### Open3DIS util
# from open3dis.dataset import build_dataset
from open3dis.dataset_outdoor import build_dataset
from open3dis.dataset.scannet_loader import ScanNetReader, scaling_mapping
from open3dis.src.fusion_util import NMS_cuda
# from open3dis.src.mapper import PointCloudToImageMapper
from open3dis.src.mapper_zbuffer import PointCloudToImageMapper

#### SAM2
# from segmenter2d.sam2.sam2.build_sam import build_sam2, build_sam2_video_predictor
# from segmenter2d.sam2.sam2.sam2_image_predictor import SAM2ImagePredictor

# class SAM2:
#     def __init__(self, cfg):
#         self.sam_model = build_sam2(cfg.foundation_model.model_cfg, cfg.foundation_model.sam2_checkpoint)
#         self.sam_generator = SAM2ImagePredictor(self.sam_model)
#         print(f'------- Loaded Segment Anything 2 (SAM-2) Model:  -------')

class Sam_Stpls3d:
    ###################################################################
    #                              SAM                                #
    ###################################################################
    def __init__(self, cfg):
        # Load Foundation Model
        sam2d = SAM_HQ(cfg)  ### sam
        # sam2d = SAM2(cfg)  ### sam2
        clip2d = CLIP_OpenAI(cfg)
        self.sam_generator = sam2d.sam_generator
        self.clip_adapter, self.clip_preprocess = clip2d.clip_adapter, clip2d.clip_preprocess


    def gen_grounded_mask_and_feat(self, scene_id, class_names, cfg, gen_feat=False):
        """
        SAM (Automatic Mask Generation) + CLIP
            Generate 2D masks using SAM's automatic mode.
            Accmulate CLIP mask feature onto 3D point cloud.
        """
        scene_dir = os.path.join(cfg.data.datapath, scene_id)

        loader = build_dataset(root_path=scene_dir, cfg=cfg)
        # scannet_loader = ScanNetReader(root_path=scene_dir, cfg=cfg)

        # Pointcloud Image mapper
        img_dim = cfg.data.img_dim
        pointcloud_mapper = PointCloudToImageMapper(
            # image_dim=img_dim, intrinsics=loader.global_intrinsic, cut_bound=cfg.data.cut_num_pixel_boundary
            image_dim=img_dim, cut_bound=cfg.data.cut_num_pixel_boundary
        )

        points = loader.read_pointcloud()
        points = torch.from_numpy(points).cuda()
        n_points = points.shape[0]

        grounded_data_dict = {}

        # Accmulate CLIP mask feature onto 3D point cloud ?
        if gen_feat:
            grounded_features = torch.zeros((n_points, cfg.foundation_model.clip_dim)).cuda()
        else:
            grounded_features = None

        for i in trange(0, len(loader), cfg.data.img_interval):
            frame = loader[i]
            frame_id = frame["frame_id"]  # str
            image_path = frame["image_path"]  # str

            #### Segment Anything (Automatic Mask Generation) ####
            image_pil = Image.open(image_path).convert("RGB")
            image_sam = cv2.imread(image_path)
            image_sam = cv2.cvtColor(image_sam, cv2.COLOR_BGR2RGB)
            
            # 在全图上先定位非白背景的有效区域，裁剪后再调用 SAM，避免白底导致的超大掩码
            H_full, W_full = image_sam.shape[:2]
            white_thresh = 250  # 判定白色背景的阈值（任一通道 < 该值即认为是非纯白）
            non_white = np.any(image_sam < white_thresh, axis=2)
            nonwhite_area = int(non_white.sum()) if np.any(non_white) else (H_full * W_full)
            area_ratio_max = 0.7  # 单个掩码占非白区域像素的最大占比阈值
            if np.any(non_white):
                ys, xs = np.where(non_white)
                pad = 10
                y0 = max(0, int(ys.min()) - pad)
                y1 = min(H_full, int(ys.max()) + 1 + pad)
                x0 = max(0, int(xs.min()) - pad)
                x1 = min(W_full, int(xs.max()) + 1 + pad)
                # 防止空裁剪
                if (y1 - y0) < 2 or (x1 - x0) < 2:
                    y0, x0, y1, x1 = 0, 0, H_full, W_full
            else:
                y0, x0, y1, x1 = 0, 0, H_full, W_full

            cropped_img = image_sam[y0:y1, x0:x1, :]

            # 使用 SAM 的全自动模式生成掩码（在裁剪后的图上）
            sam_masks_data_cropped = self.sam_generator.generate(cropped_img)

            # 将裁剪坐标系下的结果还原为全图坐标与全图尺寸掩码
            sam_masks_data = []
            for md in sam_masks_data_cropped:
                seg_crop = md['segmentation']  # Hc x Wc, bool/uint8
                if seg_crop.dtype != np.bool_:
                    seg_crop = seg_crop.astype(np.bool_)
                seg_full = np.zeros((H_full, W_full), dtype=np.bool_)
                seg_full[y0:y1, x0:x1] = np.logical_or(seg_full[y0:y1, x0:x1], seg_crop)

                x, y, w, h = md['bbox']  # XYWH in cropped coord
                bbox_full = [x + x0, y + y0, w, h]

                # point_coords 偏移到全图坐标（若存在）
                if 'point_coords' in md and md['point_coords'] is not None:
                    pts = md['point_coords']
                    # 官方返回为 [[x, y]] 的列表格式
                    pts_shifted = []
                    for p in pts:
                        if isinstance(p, (list, tuple)):
                            # 兼容 [[x, y]] 或 [[x1, y1], [x2, y2], ...]
                            pts_shifted.append([p[0] + x0, p[1] + y0])
                        else:
                            # 若是更深层嵌套，如 [[ [x, y] ]]
                            try:
                                pts_shifted.append([p[0] + x0, p[1] + y0])
                            except Exception:
                                pts_shifted = md['point_coords']
                                break
                    point_coords_full = pts_shifted
                else:
                    point_coords_full = md.get('point_coords', [])

                md_full = md.copy()
                md_full['segmentation'] = seg_full
                md_full['bbox'] = bbox_full
                md_full['area'] = int(seg_full.sum())
                md_full['point_coords'] = point_coords_full
                sam_masks_data.append(md_full)
            # 过滤白色背景
            if sam_masks_data:
                filtered_masks_data = []
                for mask_dict in sam_masks_data:
                    segmentation = mask_dict['segmentation']
                    if np.mean(image_sam[segmentation]) < 254.0:
                        filtered_masks_data.append(mask_dict)
                sam_masks_data = filtered_masks_data

            # 面积升序排序，并对大掩码执行“裁切小掩码后保留剩余像素”的逻辑
            if sam_masks_data:
                areas = [md.get('area', int(np.sum(md['segmentation']))) for md in sam_masks_data]
                order = np.argsort(areas)  # small -> large

                kept_masks_data = []
                kept_segs = []  # 已保留小掩码的二值图

                for idx in order:
                    md = sam_masks_data[idx]
                    seg = md['segmentation']
                    # 保证布尔
                    if seg.dtype != np.bool_:
                        seg = seg.astype(np.bool_)

                    # 逐个从当前掩码中裁切掉已保留的小掩码区域
                    for kept_md, kept_seg in zip(kept_masks_data, kept_segs):
                        # 先看 bbox 是否相交，不相交就跳过
                        x1, y1, w1, h1 = md['bbox']
                        x2, y2, w2, h2 = kept_md['bbox']
                        l1, t1, r1, b1 = int(x1), int(y1), int(x1 + w1), int(y1 + h1)
                        l2, t2, r2, b2 = int(x2), int(y2), int(x2 + w2), int(y2 + h2)
                        l_int = max(l1, l2)
                        r_int = min(r1, r2)
                        t_int = max(t1, t2)
                        b_int = min(b1, b2)
                        if r_int <= l_int or b_int <= t_int:
                            continue
                        # 仅在相交区域做布尔裁切
                        seg[t_int:b_int, l_int:r_int] = np.logical_and(
                            seg[t_int:b_int, l_int:r_int],
                            np.logical_not(kept_seg[t_int:b_int, l_int:r_int])
                        )

                    # 裁切后，如果掩码为空，则跳过；否则更新其 bbox 与 area
                    new_area = int(seg.sum())
                    if new_area <= 0:
                        continue
                    # 过滤“超大掩码”：若掩码面积占非白区域面积超过阈值则丢弃
                    if nonwhite_area > 0 and (new_area / nonwhite_area) > area_ratio_max:
                        continue
                    ys, xs = np.where(seg)
                    x_min, x_max = int(xs.min()), int(xs.max())
                    y_min, y_max = int(ys.min()), int(ys.max())
                    new_bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

                    md['segmentation'] = seg
                    md['area'] = new_area
                    md['bbox'] = new_bbox

                    kept_masks_data.append(md)
                    kept_segs.append(seg)

                sam_masks_data = kept_masks_data
            
            if not sam_masks_data:  # No mask in that view
                continue
            
            # 将 SAM 的输出转换为与 GroundingDINO+SAM 版本一致的张量格式
            masks_list = []
            confs_list = []
            boxes_list = []
            
            for mask_dict in sam_masks_data:
                # SAM 的 bbox 是 [x, y, w, h] 格式
                x, y, w, h = mask_dict['bbox']
                
                # 可选：过滤掉面积过小的掩码以减少噪声
                if w * h < 100:
                    continue

                masks_list.append(torch.from_numpy(mask_dict['segmentation']))
                # 使用 'predicted_iou' 作为置信度
                confs_list.append(mask_dict['predicted_iou'])
                # 将 bbox 转换为 [x1, y1, x2, y2] 格式
                boxes_list.append([x, y, x + w, y + h])

            if not masks_list: # 如果所有掩码都被过滤掉了
                continue

            size = image_pil.size
            H, W = size[1], size[0]
            
            masks = torch.stack(masks_list).unsqueeze(1).cuda() # -> [N, 1, H, W]
            confs_filt = torch.tensor(confs_list).cuda()
            boxes_filt = torch.tensor(boxes_list, dtype=torch.float32).cuda() # -> [N, 4] XYXY format

            #### CLIP Feature Extraction ####
            regions = []
            for box_id, box in enumerate(boxes_filt):
                l, t, r, b = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
                
                # 从原始掩码中裁剪出对应bbox的部分
                current_mask_in_box = masks[box_id, 0, t:b, l:r]
                # 找到在裁剪区域内背景的像素位置（掩码为False的地方）
                row, col = torch.where(current_mask_in_box == False)
                
                # 裁剪图像
                tmp = torch.tensor(image_sam)[t:b, l:r, :].cuda()
                
                # Blurring background - 复制第一个代码中的技巧以提升CLIP特征质量
                tmp[row, col, 0] = (0 * 0.5 + tmp[row, col, 0] * (1 - 0.5)).to(torch.uint8)
                tmp[row, col, 1] = (0 * 0.5 + tmp[row, col, 1] * (1 - 0.5)).to(torch.uint8)
                tmp[row, col, 2] = (0 * 0.5 + tmp[row, col, 2] * (1 - 0.5)).to(torch.uint8)
        
                regions.append(self.clip_preprocess(Image.fromarray((tmp.cpu().numpy()))))

            imgs = torch.stack(regions).cuda()
            img_batches = torch.split(imgs, 64, dim=0)
            image_features = []

            # Batch forwarding CLIP
            with torch.no_grad(), torch.cuda.amp.autocast():
                for img_batch in img_batches:
                    image_feat = self.clip_adapter.encode_image(img_batch)
                    image_feat /= image_feat.norm(dim=-1, keepdim=True)
                    image_features.append(image_feat)
            image_features = torch.cat(image_features, dim=0)

            #### SAVING MASKS, CLIP FEATURES ####
            grounded_data_dict[frame_id] = {
                "masks": masks_to_rle(masks),
                "img_feat": image_features.cpu(),
                "conf": confs_filt.cpu(),
            }

            ### ------------------------------------------------------------------------------
            # 新增可视化保存代码 (与第一个代码段完全一致)
            sambox_dir = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, "gen_2d", scene_id, "color_sambox")
            sammask_dir = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, "gen_2d", scene_id, "color_sammask")
            os.makedirs(sambox_dir, exist_ok=True)
            os.makedirs(sammask_dir, exist_ok=True)

            # 保存检测框结果 (使用SAM生成的bbox)
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            for box, conf in zip(boxes_filt, confs_filt):
                l, t, r, b = map(int, box.cpu().numpy())
                cv2.rectangle(image_np, (l, t), (r, b), (0, 255, 0), 2)
                text = f"iou: {conf:.2f}"  # 将 'conf' 改为 'iou' 以反映其来源
                cv2.putText(image_np, text, (l, b + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(sambox_dir, f"{frame_id}.png"), image_np)

            # 保存分割结果
            mask_vis = np.zeros_like(image_sam)
            for mask in masks.cpu().numpy():
                color = np.random.randint(0, 255, 3).tolist()
                mask_vis[mask[0]] = color

            combined = cv2.cvtColor(image_sam, cv2.COLOR_RGB2BGR)  # 将RGB图像转为BGR以供cv2使用
            mask_areas = np.any(mask_vis > 0, axis=2)
            combined[mask_areas] = cv2.addWeighted(
                combined[mask_areas], 0.6,
                mask_vis[mask_areas], 0.7,
                0
            )
            cv2.imwrite(os.path.join(sammask_dir, f"{frame_id}.png"), combined)
            ### ------------------------------------------------------------------------------

            if gen_feat:
                pose = loader.read_pose(frame["pose_path"])
                depth = loader.read_depth(frame["depth_path"])
                instrinsic = loader.read_intrinsic(frame["intrinsic_path"])
                
                if "stpls3d" in cfg.data.dataset_name:  # Map on image resolution in Scannetpp only
                    rgb_dim_hw = cfg.data.img_dim # use actual image H,W for STPLS3D
                    mapping = torch.ones([n_points, 4], dtype=int, device='cuda')
                    mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, rgb_dim_hw, depth=depth, intrinsic=instrinsic, id_1=scene_id, id_2=frame_id)
                else:
                    raise ValueError(f"Unknown dataset: {cfg.data.dataset_name}")

                idx = torch.where(mapping[:, 3] == 1)[0]

                if False: # Visualize highlighted points
                    import pyviz3d.visualizer as viz
                    image = loader.read_image(image_path)
                    for tmp in mapping[idx]:
                        x, y = tmp[1].item(), tmp[2].item()
                        image = cv2.circle(image, (y,x), radius=0, color=(0, 0, 255), thickness=-5)
                    cv2.imwrite('../test.png', image)
                    vis = viz.Visualizer()
                    color = torch.zeros_like(points).cpu().numpy()
                    color[idx.cpu(),0] =  255
                    vis.add_points(f'pcl', points.cpu().numpy(), color, point_size=20, visible=True)
                    vis.save('../viz')

                if len(idx) < 100:  # No points corresponds to this image, visible points on 2D image
                    continue

                pred_masks = BitMasks(masks.squeeze(1))
                # Flood fill single CLIP feature for 2D mask
                final_feat = torch.einsum("qc,qhw->chw", image_features.float(), pred_masks.tensor.float())
                ### Summing features
                grounded_features[idx] += final_feat[:, mapping[idx, 1], mapping[idx, 2]].permute(1, 0)

        grounded_features = grounded_features
        return grounded_data_dict, grounded_features

    def get_grounding_output(self, image, caption, box_threshold, text_threshold, with_logits=True, device="cuda"):
        """
        Grounding DINO box generator
        Returning boxes and logits scores for each chunk in the caption with box & text threshoding
        """

        # Caption formatting
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."

        self.grounding_dino_model = self.grounding_dino_model.to(device)
        image = image.to(device)

        # Grounding DINO box generator
        with torch.no_grad():
            outputs = self.grounding_dino_model(image[None], captions=[caption])
            logits = outputs["pred_logits"].sigmoid()[0]  # (nqueries, 256)
        boxes = outputs["pred_boxes"][0]  # (nqueries, 4)

        # Filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        return boxes_filt, logits_filt.max(dim=1)[0]

    def init_segmenter2d_models(self, cfg):
        """
        Init Segmenter 2D
        """
        # Grounding DINO
        grounding_dino_model = self.load_model(
            cfg.foundation_model.grounded_config_file, cfg.foundation_model.grounded_checkpoint, device="cuda")
        print('------- Loaded Grounding DINO OGC SwinT -------')
        return grounding_dino_model

    def load_model(self, model_config_path, model_checkpoint_path, device):
        """
        Grounding DINO loader
        """
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cuda")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        model.cuda()
        return model

    def load_image(self, image_pil):
        """
        Grounding DINO preprocess
        """
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image
    
    def generate_random_color(self):
        """Generate a random color in BGR format."""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def visualize_masks(self, image, masks):
        """
        Visualize all masks on the image with different random colors.
        """
        vis_image = np.array(image).copy()

        # Generate a list of random colors for each mask
        colors = [self.generate_random_color() for _ in range(len(masks))]

        for idx, mask in enumerate(masks):
            segmentation = mask['segmentation']
            bbox = mask['bbox']
            point_coords = mask['point_coords']

            # 将二进制掩码转换为与原图相同的尺寸
            if isinstance(segmentation, np.ndarray):
                binary_mask = segmentation.astype(np.uint8) * 255
                colored_mask = np.zeros_like(vis_image)
                colored_mask[:, :] = colors[idx]  # Assign the random color to the mask

                # 在图像上绘制掩码
                vis_image = np.where(binary_mask[..., None], colored_mask, vis_image)

            # 绘制边界框
            x, y, w, h = map(int, bbox)
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色边界框

            # 绘制点坐标
            for point in point_coords:
                px, py = map(int, point)
                cv2.circle(vis_image, (px, py), 3, (0, 0, 255), -1)  # 蓝色点

        return vis_image
