from util2d.segment_anything_hq import SAM_HQ
from util2d.stpls3d_mask_pipeline import BaseMaskStpls3d


class Sam_Stpls3d(BaseMaskStpls3d):
    def __init__(self, cfg):
        super().__init__(cfg)
        sam2d = SAM_HQ(cfg)  ### sam
        self.sam_generator = sam2d.sam_generator
        self.score_name = "iou"

    def generate_crop_masks(self, image_pil, image_rgb, class_names, cfg):
        return self.sam_generator.generate(image_rgb)
