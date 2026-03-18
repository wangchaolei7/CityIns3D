import sys
import time

import numpy as np
import torch

from util2d.stpls3d_mask_pipeline import BaseMaskStpls3d


class Sam3_Stpls3d(BaseMaskStpls3d):
    def __init__(self, cfg):
        super().__init__(cfg)

        sam3_repo_path = getattr(cfg.foundation_model, "sam3_repo_path", "/home/wangcl/sam3")
        if sam3_repo_path not in sys.path:
            sys.path.insert(0, sam3_repo_path)

        from sam3.model_builder import build_sam3_image_model
        from sam3.eval.postprocessors import PostProcessImage
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model.utils.misc import copy_data_to_device
        from sam3.train.data.collator import collate_fn_api as collate
        from sam3.train.data.sam3_image_dataset import (
            Datapoint,
            FindQueryLoaded,
            Image as SAMImage,
            InferenceMetadata,
        )
        from sam3.train.transforms.basic_for_api import (
            ComposeAPI,
            NormalizeAPI,
            RandomResizeAPI,
            ToTensorAPI,
        )

        sam3_checkpoint = cfg.foundation_model.sam3_checkpoint
        sam3_device = getattr(cfg.foundation_model, "device", "cuda")
        sam3_compile = bool(getattr(cfg.foundation_model, "sam3_compile", False))
        sam3_conf_threshold = float(
            getattr(cfg.foundation_model, "sam3_confidence_threshold", 0.5)
        )
        sam3_resolution = int(getattr(cfg.foundation_model, "sam3_resolution", 1008))
        sam3_inference_mode = getattr(
            cfg.foundation_model,
            "sam3_inference_mode",
            "batched",
        )
        if sam3_inference_mode not in {"batched", "sequential"}:
            raise ValueError(
                "foundation_model.sam3_inference_mode must be 'batched' or 'sequential'"
            )

        self.model = build_sam3_image_model(
            checkpoint_path=sam3_checkpoint,
            load_from_HF=False,
            device=sam3_device,
            enable_inst_interactivity=True,
            compile=sam3_compile,
        )
        self.processor = Sam3Processor(
            self.model,
            resolution=sam3_resolution,
            device=sam3_device,
            confidence_threshold=sam3_conf_threshold,
        )
        self.device = torch.device(sam3_device)
        self.collate = collate
        self.copy_data_to_device = copy_data_to_device
        self.datapoint_cls = Datapoint
        self.find_query_cls = FindQueryLoaded
        self.inference_metadata_cls = InferenceMetadata
        self.sam_image_cls = SAMImage
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=sam3_resolution,
                    max_size=sam3_resolution,
                    square=True,
                    consistent_transform=False,
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=sam3_conf_threshold,
            to_cpu=False,
        )
        self.prompt_template = getattr(
            cfg.foundation_model,
            "sam3_prompt_template",
            "{label}",
        )
        self.inference_mode = sam3_inference_mode
        self.log_timing = bool(getattr(cfg.foundation_model, "sam3_log_timing", False))
        self.query_id_counter = 1
        self.score_name = "score"
        print(
            f"------- Loaded Segment Anything 3 (SAM3) Model "
            f"[mode={self.inference_mode}] -------"
        )

    def generate_crop_masks(self, image_pil, image_rgb, class_names, cfg):
        if not class_names:
            return []

        start_time = time.perf_counter()
        self._sync_device()

        if self.inference_mode == "sequential":
            mask_dicts = self._generate_crop_masks_sequential(
                image_pil=image_pil,
                class_names=class_names,
            )
        else:
            mask_dicts = self._generate_crop_masks_batched(
                image_pil=image_pil,
                class_names=class_names,
            )

        self._sync_device()
        if self.log_timing:
            elapsed = time.perf_counter() - start_time
            print(
                f"[SAM3][{self.inference_mode}] prompts={len(class_names)} "
                f"masks={len(mask_dicts)} elapsed={elapsed:.3f}s"
            )
        return mask_dicts

    def generate_point_prompt_masks(self, image_pil, point_prompts, cfg):
        if not point_prompts:
            return []
        if not hasattr(self.model, "predict_inst") or getattr(self.model, "inst_interactive_predictor", None) is None:
            raise RuntimeError("Current SAM3 model build does not expose interactive point-prompt prediction.")

        start_time = time.perf_counter()
        self._sync_device()

        state = self.processor.set_image(image_pil)
        score_threshold = float(
            getattr(
                cfg.foundation_model,
                "sam3_stage2_confidence_threshold",
                getattr(cfg.foundation_model, "sam3_confidence_threshold", 0.5),
            )
        )

        mask_dicts = []
        width, height = image_pil.size
        for prompt_index, point in enumerate(point_prompts):
            x, y = int(point[0]), int(point[1])
            if x < 0 or x >= width or y < 0 or y >= height:
                continue

            masks, scores, _ = self.model.predict_inst(
                state,
                point_coords=np.asarray([[x, y]], dtype=np.float32),
                point_labels=np.asarray([1], dtype=np.int32),
                multimask_output=True,
                return_logits=False,
                normalize_coords=True,
            )
            if masks is None or scores is None or len(scores) == 0:
                continue

            containing_ids = np.where(masks[:, y, x] > 0)[0]
            if containing_ids.size > 0:
                best_local = int(containing_ids[np.argmax(scores[containing_ids])])
            else:
                best_local = int(np.argmax(scores))

            score = float(scores[best_local])
            if score < score_threshold:
                continue

            segmentation = masks[best_local].astype(np.bool_)
            if not np.any(segmentation):
                continue
            segmentation = self._keep_prompt_component(segmentation, [x, y])
            if not np.any(segmentation):
                continue

            ys, xs = np.where(segmentation)
            bbox = [
                int(xs.min()),
                int(ys.min()),
                int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1),
            ]
            mask_dicts.append(
                {
                    "segmentation": segmentation,
                    "bbox": bbox,
                    "area": int(segmentation.sum()),
                    "predicted_iou": score,
                    "point_coords": [[x, y]],
                    "prompt_index": int(prompt_index),
                    "prompt_type": "point",
                }
            )

        self._sync_device()
        if self.log_timing:
            elapsed = time.perf_counter() - start_time
            print(
                f"[SAM3][point] prompts={len(point_prompts)} "
                f"masks={len(mask_dicts)} elapsed={elapsed:.3f}s"
            )
        return mask_dicts

    def _generate_crop_masks_batched(self, image_pil, class_names):
        datapoint, query_id_to_prompt = self._build_text_datapoint(
            image_pil=image_pil,
            class_names=class_names,
        )
        datapoint = self.transform(datapoint)
        batch = self.collate([datapoint], dict_key="dummy")["dummy"]
        batch = self.copy_data_to_device(batch, self.device, non_blocking=True)

        with torch.inference_mode():
            output = self.model(batch)
        processed_results = self.postprocessor.process_results(
            output,
            batch.find_metadatas,
        )

        mask_dicts = []
        for query_id, prompt in query_id_to_prompt.items():
            result = processed_results.get(query_id)
            if result is None:
                continue

            masks = result.get("masks")
            scores = result.get("scores")
            if masks is None or scores is None or masks.numel() == 0:
                continue

            masks = masks.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()

            for mask, score in zip(masks, scores):
                segmentation = mask.squeeze(0).astype(np.bool_)
                if not np.any(segmentation):
                    continue

                ys, xs = np.where(segmentation)
                bbox = [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() - xs.min() + 1),
                    int(ys.max() - ys.min() + 1),
                ]
                mask_dicts.append(
                    {
                        "segmentation": segmentation,
                        "bbox": bbox,
                        "area": int(segmentation.sum()),
                        "predicted_iou": float(score),
                        "point_coords": [],
                        "text_prompt": prompt,
                    }
                )

        return mask_dicts

    def _generate_crop_masks_sequential(self, image_pil, class_names):
        state = self.processor.set_image(image_pil)
        mask_dicts = []

        for class_name in class_names:
            prompt = self.prompt_template.format(label=class_name)
            self.processor.reset_all_prompts(state)
            output = self.processor.set_text_prompt(prompt=prompt, state=state)

            masks = output.get("masks")
            scores = output.get("scores")
            if masks is None or scores is None or masks.numel() == 0:
                continue

            masks = masks.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()
            mask_dicts.extend(self._build_mask_dicts(masks, scores, prompt))

        return mask_dicts

    def _build_text_datapoint(self, image_pil, class_names):
        width, height = image_pil.size
        datapoint = self.datapoint_cls(find_queries=[], images=[])
        datapoint.images = [
            self.sam_image_cls(
                data=image_pil,
                objects=[],
                size=[height, width],
            )
        ]

        query_id_to_prompt = {}
        for class_name in class_names:
            prompt = self.prompt_template.format(label=class_name)
            query_id = self.query_id_counter
            self.query_id_counter += 1
            datapoint.find_queries.append(
                self.find_query_cls(
                    query_text=prompt,
                    image_id=0,
                    object_ids_output=[],
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=self.inference_metadata_cls(
                        coco_image_id=query_id,
                        original_image_id=query_id,
                        original_category_id=query_id,
                        original_size=[height, width],
                        object_id=0,
                        frame_index=0,
                    ),
                )
            )
            query_id_to_prompt[query_id] = prompt

        return datapoint, query_id_to_prompt

    def _build_mask_dicts(self, masks, scores, prompt):
        mask_dicts = []
        for mask, score in zip(masks, scores):
            segmentation = mask.squeeze(0).astype(np.bool_)
            if not np.any(segmentation):
                continue

            ys, xs = np.where(segmentation)
            bbox = [
                int(xs.min()),
                int(ys.min()),
                int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1),
            ]
            mask_dicts.append(
                {
                    "segmentation": segmentation,
                    "bbox": bbox,
                    "area": int(segmentation.sum()),
                    "predicted_iou": float(score),
                    "point_coords": [],
                    "text_prompt": prompt,
                }
            )
        return mask_dicts

    def _sync_device(self):
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
