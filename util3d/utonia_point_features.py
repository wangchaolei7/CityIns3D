import os
import sys
from typing import Dict, Tuple

import numpy as np
import open3d as o3d
import torch

from open3dis.dataset_outdoor.stpls3d_io import load_pointcloud_xyz_rgb

_SAVE_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class UtoniaPointFeatureExtractor:
    def __init__(self, cfg):
        self.cfg = cfg

        repo_path = os.path.abspath(
            os.path.expanduser(
                getattr(cfg.foundation_model, "utonia_repo_path", "./segmenter3d/Utonia")
            )
        )
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        import utonia

        self.utonia = utonia
        self.device = self._resolve_device(getattr(cfg.foundation_model, "device", "cuda"))
        self.scale = float(getattr(cfg.foundation_model, "utonia_scale", 0.2))
        self.apply_z_positive = bool(
            getattr(cfg.foundation_model, "utonia_apply_z_positive", False)
        )
        self.normalize_coord = bool(
            getattr(cfg.foundation_model, "utonia_normalize_coord", False)
        )
        self.upcast_levels = int(getattr(cfg.foundation_model, "utonia_upcast_levels", 4))
        self.superpoint_upcast_levels = int(
            getattr(cfg.foundation_model, "utonia_superpoint_upcast_levels", 2)
        )
        self.save_dtype = self._resolve_save_dtype(
            getattr(cfg.foundation_model, "utonia_output_dtype", "float16")
        )

        checkpoint = getattr(cfg.foundation_model, "utonia_checkpoint", "utonia")
        repo_id = getattr(cfg.foundation_model, "utonia_repo_id", "Pointcept/Utonia")
        download_root = getattr(cfg.foundation_model, "utonia_download_root", None)

        custom_config = None
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            fallback_patch_size = int(
                getattr(cfg.foundation_model, "utonia_fallback_patch_size", 1024)
            )
            custom_config = {
                "enc_patch_size": [fallback_patch_size for _ in range(5)],
                "enable_flash": False,
            }

        self.model = self.utonia.load(
            checkpoint,
            repo_id=repo_id,
            download_root=download_root,
            custom_config=custom_config,
        ).to(self.device)
        self.model.eval()
        self.transform = self.utonia.transform.default(
            scale=self.scale,
            apply_z_positive=self.apply_z_positive,
            normalize_coord=self.normalize_coord,
        )

    def extract_from_file(self, ply_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
        point_dict, num_points = self._load_point_cloud(ply_path)
        point_dict = self.transform(point_dict)
        input_grid_coord = point_dict["grid_coord"].detach().to(dtype=torch.int32).cpu().contiguous()
        input_inverse = point_dict["inverse"].detach().to(dtype=torch.int64).cpu().contiguous()

        with torch.inference_mode():
            for key, value in point_dict.items():
                if isinstance(value, torch.Tensor):
                    point_dict[key] = value.to(self.device, non_blocking=True)

            encoded_point = self.model(point_dict)
            grid_feat = self._extract_upcast_grid_feature(encoded_point, self.upcast_levels)
            if self.superpoint_upcast_levels == self.upcast_levels:
                spp_grid_feat = grid_feat
            else:
                spp_grid_feat = self._extract_upcast_grid_feature(
                    encoded_point,
                    self.superpoint_upcast_levels,
                )

            feat = grid_feat[input_inverse.to(self.device)]
            spp_feat = spp_grid_feat[input_inverse.to(self.device)]

            feature_pack = {
                "feat": feat.to(dtype=self.save_dtype).cpu().contiguous(),
                "grid_feat": grid_feat.to(dtype=self.save_dtype).cpu().contiguous(),
                "spp_feat": spp_feat.to(dtype=self.save_dtype).cpu().contiguous(),
                "spp_grid_feat": spp_grid_feat.to(dtype=self.save_dtype).cpu().contiguous(),
                "inverse": input_inverse,
                "grid_coord": input_grid_coord,
            }

        meta = {
            "num_points": int(num_points),
            "num_grid_points": int(feature_pack["grid_feat"].shape[0]),
            "feat_dim": int(feature_pack["feat"].shape[1]),
        }
        return feature_pack, meta

    def _resolve_device(self, requested_device: str) -> torch.device:
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested for Utonia, but CUDA is not available.")
        return device

    def _resolve_save_dtype(self, dtype_name: str) -> torch.dtype:
        if dtype_name not in _SAVE_DTYPES:
            valid = ", ".join(sorted(_SAVE_DTYPES))
            raise ValueError(f"Unsupported Utonia save dtype '{dtype_name}'. Valid: {valid}")
        return _SAVE_DTYPES[dtype_name]

    def _load_point_cloud(self, ply_path: str) -> Tuple[Dict[str, np.ndarray], int]:
        coord, color = load_pointcloud_xyz_rgb(ply_path)
        if coord.ndim != 2 or coord.shape[1] != 3:
            raise ValueError(f"Unexpected point cloud shape in '{ply_path}': {coord.shape}")

        if color.size > 0 and color.max() <= 1.0 + 1e-6:
            color = color * 255.0
        normal = np.zeros_like(coord, dtype=np.float32)

        point_dict = {
            "coord": coord.astype(np.float32, copy=False),
            "color": color.astype(np.float32, copy=False),
            "normal": normal.astype(np.float32, copy=False),
        }
        return point_dict, coord.shape[0]

    def _extract_upcast_grid_feature(self, point, concat_levels: int) -> torch.Tensor:
        current_point = point
        current_feat = point.feat
        current_concat_level = 0

        while "pooling_parent" in current_point.keys():
            if "pooling_inverse" not in current_point.keys():
                raise KeyError("Missing pooling_inverse while upcasting Utonia features.")

            parent = current_point.pooling_parent
            inverse = current_point.pooling_inverse
            if current_concat_level < concat_levels:
                current_feat = torch.cat([parent.feat, current_feat[inverse]], dim=-1)
            else:
                current_feat = current_feat[inverse]
            current_point = parent
            current_concat_level += 1

        return current_feat.detach()
