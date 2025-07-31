import os
import glob
import numpy as np
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from demo.predictor import VisualizationDemo
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
import cv2
import tqdm

def setup_cfg(config_file, model_weights):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.freeze()
    return cfg

def visualize_gt(image, gt_mask, metadata):
    """可视化ground truth"""
    visualizer = Visualizer(image[:, :, ::-1], metadata, instance_mode=ColorMode.IMAGE)
    vis_output = visualizer.draw_sem_seg(gt_mask)
    return vis_output.get_image()

def main():
    config_file = "configs/loveda/semantic-segmentation/maskformer2_R50_bs16_90k.yaml"
    model_weights = "/home/liw324/code/Segment/Mask2Former/output/train_1/model_0029999.pth"
    
    cfg = setup_cfg(config_file, model_weights)
    demo = VisualizationDemo(cfg, instance_mode=ColorMode.IMAGE)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    
    # 验证集图像路径
    val_images = glob.glob("datasets/LoveDA/Val/*/images_png/*.png")
    output_dir = "visualization_output/temp"
    os.makedirs(output_dir, exist_ok=True)
    
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
        predictions, vis_output = demo.run_on_image(img)
        pred_vis = vis_output.get_image()
        
        # 可视化GT
        gt_vis = visualize_gt(img, gt_mask, metadata)
        
        # 确保三张图片高度一致
        h = img.shape[0]
        img_resized = cv2.resize(img, (img.shape[1], h))
        pred_resized = cv2.resize(pred_vis[:, :, ::-1], (pred_vis.shape[1], h))
        gt_resized = cv2.resize(gt_vis[:, :, ::-1], (gt_vis.shape[1], h))
        
        # 水平拼接：原图 | 预测 | GT
        combined = np.hstack([img_resized, pred_resized, gt_resized])
        
        # 保存拼接结果
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, combined)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()