from open3dis.dataset_outdoor.stpls3d_loader import STPLS3DReader

__all__ = ["STPLS3DReader"]


def build_dataset(root_path, cfg):
    if cfg.data.dataset_name == 'stpls3d':
        return STPLS3DReader(root_path, cfg)
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.dataset_name}")