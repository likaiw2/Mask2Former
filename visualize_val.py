import os
import glob
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from demo.predictor import VisualizationDemo
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog
from detectron2.utils.file_io import PathManager
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

def main():
    config_file = "configs/loveda/semantic-segmentation/maskformer2_R50_bs16_90k.yaml"
    model_weights = "/home/liw324/code/Segment/Mask2Former/output/train_1/model_0029999.pth"  # 替换为你的模型路径
    
    cfg = setup_cfg(config_file, model_weights)
    demo = VisualizationDemo(cfg, instance_mode=ColorMode.IMAGE)
    
    # 验证集图像路径
    val_images = glob.glob("datasets/LoveDA/Val/*/images_png/*.png")
    output_dir = "visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in tqdm.tqdm(val_images):
        img = cv2.imread(img_path)
        predictions, vis_output = demo.run_on_image(img)
        
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        vis_output.save(output_path)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()