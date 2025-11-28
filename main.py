import os
import torch
from pathlib import Path
from tqdm import tqdm

# === 引入模块 ===
from core.extractor import VGGBlock3Extractor
from utils.image_ops import process_image
from utils.file_ops import increment_path, load_yolo_labels, append_results_to_summary
from utils.visualizer import visualize_batch_results, get_distinct_colors
from core.matcher import correlation_layer, ransac_smart
from core.metrics import calculate_errors

def main():
    # 1. 配置参数
    root_dir = './Datasets'
    vi_dir = os.path.join(root_dir, 'vi')
    ir_dir = os.path.join(root_dir, 'ir')
    labels_dir = os.path.join(root_dir, 'labels')
    
    project = 'runs'
    name = 'exp'
    
    # 准备保存路径
    save_dir = increment_path(Path(project) / name, exist_ok=False, mkdir=True)
    results_txt_path = os.path.join(save_dir, 'results.txt')

    print(f"==========================================")
    print(f"  批量配准任务启动")
    print(f"  输出目录: {save_dir}")
    print(f"==========================================\n")

    # 2. 初始化模型
    print("加载特征提取模型...", end='')
    model = VGGBlock3Extractor()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    print(" 完成\n")

    # 3. 扫描文件
    supported_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    if not os.path.exists(vi_dir):
        print(f"Error: {vi_dir} 不存在")
        return

    file_list = [f for f in os.listdir(vi_dir) if f.lower().endswith(supported_exts)]
    file_ids = sorted([os.path.splitext(f)[0] for f in file_list])
    
    print(f"待处理图像组数: {len(file_ids)}\n")

    # 初始化日志文件
    with open(results_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"批量配准测试报告\nTotal: {len(file_ids)}\n{'='*60}\n")

    # 4. 主循环
    for img_id in tqdm(file_ids, desc="Processing"):
        # 路径匹配
        vi_path = next((os.path.join(vi_dir, img_id + ext) for ext in supported_exts if os.path.exists(os.path.join(vi_dir, img_id + ext))), None)
        ir_path = next((os.path.join(ir_dir, img_id + ext) for ext in supported_exts if os.path.exists(os.path.join(ir_dir, img_id + ext))), None)
        label_path = os.path.join(labels_dir, img_id + ".txt")

        if not (vi_path and ir_path and os.path.exists(label_path)):
            continue

        # 图片处理
        rgb_tensor, rgb_sobel, rgb_viz = process_image(vi_path)
        ir_tensor, _, ir_viz = process_image(ir_path)
        
        if rgb_tensor is None or ir_tensor is None: continue

        # 模型推理
        if torch.cuda.is_available():
            rgb_tensor = rgb_tensor.cuda()
            ir_tensor = ir_tensor.cuda()

        with torch.no_grad():
            rgb_feat = model(rgb_tensor)
            ir_feat = model(ir_tensor)
        
        # 核心计算
        corr_matrix = correlation_layer(rgb_feat, ir_feat)
        feat_size = (rgb_feat.shape[2], rgb_feat.shape[3])
        orig_size = rgb_viz.shape[:2]
        
        all_boxes = load_yolo_labels(label_path)
        colors = get_distinct_colors(len(all_boxes))
        
        final_results = []
        
        # 遍历所有目标框
        for i, box in enumerate(all_boxes):
            # 将 Tensor 转回 CPU 进行 RANSAC 计算
            cpu_corr = corr_matrix.cpu()
            trans_corners, _ = ransac_smart(cpu_corr, box, rgb_sobel, feat_size, orig_size)
            
            if trans_corners is not None:
                metrics = calculate_errors(box, trans_corners, orig_size[1], orig_size[0])
                final_results.append({
                    'orig_box': box,
                    'trans_box': trans_corners,
                    'color': colors[i],
                    'metrics': metrics
                })
        
        # 结果保存
        visualize_batch_results(rgb_viz, ir_viz, final_results, save_dir, img_id)
        append_results_to_summary(results_txt_path, img_id, final_results)

    print(f"\n任务完成。查看报告: {results_txt_path}")

if __name__ == "__main__":
    main()