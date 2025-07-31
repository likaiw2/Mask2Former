# Copyright (c) Facebook, Inc. and its affiliates.
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# LoveDA数据集类别定义
_LOVEDA_CLASSES = [
    "background",
    "building", 
    "road",
    "water",
    "barren",
    "forest", 
    "agricultural"
]

_LOVEDA_COLORS = [
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
    
    for name, dirname in [("train", "Train"), ("val", "Val"), ("test", "Test")]:
        image_dir = os.path.join(root, dirname, "images_png")
        gt_dir = os.path.join(root, dirname, "masks_png")
        
        dataset_name = f"loveda_sem_seg_{name}"
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

# 注册数据集
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_loveda(_root)