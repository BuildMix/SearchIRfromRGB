# main.py
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# --- 导入自定义模块 ---
from core.feature_extractor import VGGBlock1Extractor
from core.image_process import process_image
from core.matcher import sparse_ransac_smart
from utils.io_utils import load_yolo_labels, increment_path, append_results_to_summary
from utils.visualizer import get_distinct_colors, calculate_errors, visualize_batch_results

def main():
    # --- 配置区域 ---
    root_dir = './Datasets'
    vi_dir = os.path.join(root_dir, 'vi')
    ir_dir = os.path.join(root_dir, 'ir')
    labels_dir = os.path.join(root_dir, 'labels')
    
    # 保存配置
    save_dir = increment_path(Path('runs') / 'exp', exist_ok=False, mkdir=True)
    results_txt = os.path.join(save_dir, 'results.txt')

    print(f"=== High-Res Sparse SCN Registration (Refactored) ===")
    print(f"Output: {save_dir}")

    # 1. 加载模型
    model = VGGBlock1Extractor()
    model.eval()

    # 2. 遍历文件
    if not os.path.exists(vi_dir):
        print(f"Error: Dataset directory not found at {vi_dir}")
        return

    file_list = [f for f in os.listdir(vi_dir) if f.endswith(('.jpg', '.png'))]
    file_ids = [os.path.splitext(f)[0] for f in file_list]
    file_ids.sort()

    for img_id in tqdm(file_ids, desc="Processing"):
        vi_p = os.path.join(vi_dir, img_id + ".png")
        ir_p = os.path.join(ir_dir, img_id + ".png")
        lbl_p = os.path.join(labels_dir, img_id + ".txt")

        # 读取与预处理
        rgb_tensor, rgb_sobel, rgb_viz = process_image(vi_p)
        ir_tensor, _, ir_viz = process_image(ir_p)
        all_boxes = load_yolo_labels(lbl_p)

        if rgb_tensor is None or ir_tensor is None or len(all_boxes) == 0:
            continue

        # 特征提取
        with torch.no_grad():
            rgb_feat = model(rgb_tensor) 
            ir_feat = model(ir_tensor)

        # 归一化特征
        rgb_feat = F.normalize(rgb_feat, p=2, dim=1)
        ir_feat = F.normalize(ir_feat, p=2, dim=1)
        
        # 展平 IR 特征
        B, C, Hf, Wf = ir_feat.shape
        ir_feat_flat = ir_feat.view(B, C, -1)

        # 准备数据
        colors = get_distinct_colors(len(all_boxes))
        final_results = []
        feat_size = (Hf, Wf)
        orig_size = rgb_viz.shape[:2]

        # 循环处理每个框
        for i, box in enumerate(all_boxes):
            trans_corners, _ = sparse_ransac_smart(
                rgb_feat, ir_feat_flat, box, rgb_sobel, feat_size, orig_size
            )
            
            if trans_corners is not None:
                metrics = calculate_errors(box, trans_corners, orig_size[1], orig_size[0])
                final_results.append({
                    'orig_box': box,
                    'trans_box': trans_corners,
                    'color': colors[i],
                    'metrics': metrics
                })

        # 保存结果
        visualize_batch_results(rgb_viz, ir_viz, final_results, save_dir, img_id)
        append_results_to_summary(results_txt, img_id, final_results)

    print(f"Done. Check {results_txt}")

if __name__ == "__main__":
    main()