import os
import glob
import json
import csv
import numpy as np
from collections import defaultdict
import tqdm
import cv2

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog

def setup_cfg(config_file, model_weights):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()
    return cfg

def calculate_miou(pred_mask, gt_mask, num_classes, ignore_label=255):
    """计算每个类别的IoU"""
    # 忽略ignore_label
    valid_mask = (gt_mask != ignore_label)
    pred_mask = pred_mask[valid_mask]
    gt_mask = gt_mask[valid_mask]
    
    # 计算混淆矩阵
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # 确保预测值在有效范围内
    pred_mask = np.clip(pred_mask, 0, num_classes - 1)
    gt_mask = np.clip(gt_mask, 0, num_classes - 1)
    
    # 计算混淆矩阵
    indices = num_classes * gt_mask + pred_mask
    conf_matrix = np.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
    
    # 计算每个类别的IoU
    ious = []
    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        
        if tp + fp + fn == 0:
            ious.append(np.nan)
        else:
            ious.append(tp / (tp + fp + fn))
    
    return np.array(ious), conf_matrix

def main():
    config_file = "configs/loveda/semantic-segmentation/maskformer2_R50_bs16_90k.yaml"
    model_weights = "/home/liw324/code/Segment/Mask2Former/output/train_1/model_0029999.pth"
    
    cfg = setup_cfg(config_file, model_weights)
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    
    class_names = metadata.stuff_classes
    num_classes = len(class_names)
    ignore_label = metadata.ignore_label
    
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    
    # 验证集图像路径
    val_images = glob.glob("datasets/LoveDA/Val/*/images_png/*.png")
    
    # 累积混淆矩阵
    total_conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_image_results = []
    
    for img_path in tqdm.tqdm(val_images):
        # 构建对应的GT路径
        gt_path = img_path.replace("images_png", "masks_png")
        
        if not os.path.exists(gt_path):
            print(f"GT not found: {gt_path}")
            continue
            
        # 读取图像和GT
        img = cv2.imread(img_path)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # 获取预测结果
        outputs = predictor(img)
        pred_mask = outputs["sem_seg"].argmax(dim=0).cpu().numpy()
        
        # 计算当前图像的IoU
        ious, conf_matrix = calculate_miou(pred_mask, gt_mask, num_classes, ignore_label)
        total_conf_matrix += conf_matrix
        
        # 保存每张图像的结果
        image_result = {
            "image_name": os.path.basename(img_path),
            "per_class_iou": {class_names[i]: float(ious[i]) if not np.isnan(ious[i]) else None 
                             for i in range(num_classes)},
            "mean_iou": float(np.nanmean(ious))
        }
        per_image_results.append(image_result)
    
    # 计算总体IoU
    total_ious = []
    for i in range(num_classes):
        tp = total_conf_matrix[i, i]
        fp = total_conf_matrix[:, i].sum() - tp
        fn = total_conf_matrix[i, :].sum() - tp
        
        if tp + fp + fn == 0:
            total_ious.append(np.nan)
        else:
            total_ious.append(tp / (tp + fp + fn))
    
    total_ious = np.array(total_ious)
    mean_iou = np.nanmean(total_ious)
    
    # 准备结果
    results = {
        "overall_results": {
            "mean_iou": float(mean_iou),
            "per_class_iou": {class_names[i]: float(total_ious[i]) if not np.isnan(total_ious[i]) else None 
                             for i in range(num_classes)},
            "confusion_matrix": total_conf_matrix.tolist()
        },
        "per_image_results": per_image_results
    }
    
    # # 保存JSON结果
    # with open("loveda_miou_results.json", "w") as f:
    #     json.dump(results, f, indent=2)
    
    # 保存CSV结果
    with open("loveda_miou_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        
        # 写入总体结果
        writer.writerow(["Overall Results"])
        writer.writerow(["Class", "IoU"])
        for i, class_name in enumerate(class_names):
            iou_val = total_ious[i] if not np.isnan(total_ious[i]) else "N/A"
            writer.writerow([class_name, iou_val])
        writer.writerow(["Mean IoU", mean_iou])
        writer.writerow([])
        
        # 写入每张图像的结果
        writer.writerow(["Per Image Results"])
        header = ["Image"] + class_names + ["Mean IoU"]
        writer.writerow(header)
        
        for result in per_image_results:
            row = [result["image_name"]]
            for class_name in class_names:
                iou_val = result["per_class_iou"][class_name]
                row.append(iou_val if iou_val is not None else "N/A")
            row.append(result["mean_iou"])
            writer.writerow(row)
    
    # 打印结果
    print("\n=== Overall Results ===")
    print(f"Mean IoU: {mean_iou:.4f}")
    print("\nPer-class IoU:")
    for i, class_name in enumerate(class_names):
        if not np.isnan(total_ious[i]):
            print(f"{class_name}: {total_ious[i]:.4f}")
        else:
            print(f"{class_name}: N/A")
    
    print(f"\nResults saved to:")
    print(f"- loveda_miou_results.json")
    print(f"- loveda_miou_results.csv")

if __name__ == "__main__":
    main()