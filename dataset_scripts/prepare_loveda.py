#!/usr/bin/env python3

import os
import numpy as np
from PIL import Image
import argparse

def convert_loveda_labels(input_dir, output_dir):
    """
    转换LoveDA标签格式（如果需要）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # 读取标签图像
            label_img = np.array(Image.open(input_path))
            
            # 如果需要标签映射，在这里处理
            # label_img = map_labels(label_img)
            
            # 保存处理后的标签
            Image.fromarray(label_img).save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    convert_loveda_labels(args.input_dir, args.output_dir)