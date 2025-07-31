# Copyright (c) Facebook, Inc. and its affiliates.
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# LoveDA数据集类别定义
_LOVEDA_CLASSES = [
    "nothing",
    "background",
    "building", 
    "road",
    "water",
    "barren",
    "forest", 
    "agricultural"
]

_LOVEDA_COLORS = [
    [0, 0, 0],        # nothing - black
    [255, 255, 255],  # background - white
    [255, 0, 0],      # building - red
    [255, 255, 0],    # road - yellow  
    [0, 0, 255],      # water - blue
    [159, 129, 183],  # barren - purple
    [0, 255, 0],      # forest - green
    [255, 195, 128],  # agricultural - orange
]

def _get_loveda_meta():
    return {
        "stuff_classes": _LOVEDA_CLASSES,
        "stuff_colors": _LOVEDA_COLORS,
    }

def register_all_loveda(root):
    root = os.path.join(root, "LoveDA")
    meta = _get_loveda_meta()
    
    for name, dirname in [("train", "Train"), ("val", "Val")]:
        # 为Rural和Urban分别注册数据集
        for area in ["Rural", "Urban"]:
            image_dir = os.path.join(root, dirname, area, "images_png")
            gt_dir = os.path.join(root, dirname, area, "masks_png")
            
            dataset_name = f"loveda_sem_seg_{name}_{area.lower()}"
            DatasetCatalog.register(
                dataset_name, 
                lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
            )
            MetadataCatalog.get(dataset_name).set(
                image_root=image_dir,
                sem_seg_root=gt_dir,
                evaluator_type="sem_seg",
                ignore_label=255,
                **meta,
            )
        
        # 注册合并的数据集（Rural + Urban）
        rural_image_dir = os.path.join(root, dirname, "Rural", "images_png")
        rural_gt_dir = os.path.join(root, dirname, "Rural", "masks_png")
        urban_image_dir = os.path.join(root, dirname, "Urban", "images_png")
        urban_gt_dir = os.path.join(root, dirname, "Urban", "masks_png")
        
        def load_combined_sem_seg(rural_gt, rural_img, urban_gt, urban_img):
            rural_data = load_sem_seg(rural_gt, rural_img, gt_ext="png", image_ext="png")
            urban_data = load_sem_seg(urban_gt, urban_img, gt_ext="png", image_ext="png")
            return rural_data + urban_data
        
        dataset_name = f"loveda_sem_seg_{name}"
        DatasetCatalog.register(
            dataset_name,
            lambda: load_combined_sem_seg(rural_gt_dir, rural_image_dir, urban_gt_dir, urban_image_dir)
        )
        MetadataCatalog.get(dataset_name).set(
            image_root=[rural_image_dir, urban_image_dir],
            sem_seg_root=[rural_gt_dir, urban_gt_dir],
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

# 注册数据集
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_loveda(_root)
