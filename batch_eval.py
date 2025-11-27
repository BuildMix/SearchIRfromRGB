import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import math
from pathlib import Path

# ==========================================
# Part 1: 模型与预处理 (保持不变)
# ==========================================
class VGGBlock3Extractor(nn.Module):
    def __init__(self):
        super(VGGBlock3Extractor, self).__init__()
        full_vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = full_vgg16.features[:17]
        self.inst_norm = nn.InstanceNorm2d(256, affine=False)
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        feat = self.features(x)
        feat = self.inst_norm(feat)
        return feat

def apply_sobel_algorithm(gray_img):
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    abs_x = cv2.convertScaleAbs(sobel_x)
    abs_y = cv2.convertScaleAbs(sobel_y)
    return cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

def process_image(image_path):
    if not os.path.exists(image_path):
        print(f"错误: 找不到文件 {image_path}")
        return None, None, None, None
    orig_img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    sobel_img = apply_sobel_algorithm(gray_img) 
    sobel_rgb = cv2.merge([sobel_img, sobel_img, sobel_img])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(sobel_rgb).unsqueeze(0)
    return img_tensor, sobel_img, cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

def correlation_layer(feat_rgb, feat_ir):
    B, C, H, W = feat_rgb.shape
    N = H * W
    feat_rgb_norm = F.normalize(feat_rgb, p=2, dim=1)
    feat_ir_norm = F.normalize(feat_ir, p=2, dim=1)
    feat_rgb_flat = feat_rgb_norm.view(B, C, N)
    feat_ir_flat = feat_ir_norm.view(B, C, N)
    correlation_matrix = torch.bmm(feat_rgb_flat.transpose(1, 2), feat_ir_flat)
    return correlation_matrix

# ==========================================
# Part 2: Step 4 智能缩放版 (保持不变)
# ==========================================
def step4_ransac_smart(corr_matrix, norm_box, sobel_map, feat_size, orig_size):
    B, N, N_dim = corr_matrix.shape
    H_feat, W_feat = feat_size
    H_orig, W_orig = orig_size
    
    ncx, ncy, nw, nh = norm_box
    pcx, pcy = int(ncx * W_orig), int(ncy * H_orig)
    pw, ph = int(nw * W_orig), int(nh * H_orig)
    x_min, y_min = pcx - pw//2, pcy - ph//2
    x_max, y_max = pcx + pw//2, pcy + ph//2

    grid_size = 6
    xs = np.linspace(x_min + pw*0.1, x_max - pw*0.1, grid_size)
    ys = np.linspace(y_min + ph*0.1, y_max - ph*0.1, grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points_rgb = np.vstack((grid_x.flatten(), grid_y.flatten())).T

    valid_matches_src = []
    valid_matches_dst = []
    
    sobel_threshold = 15 
    search_radius = W_orig // 3 
    feat_search_radius = search_radius * (W_feat / W_orig)

    for pt in points_rgb:
        px, py = int(pt[0]), int(pt[1])
        if px < 0 or px >= W_orig or py < 0 or py >= H_orig: continue
        if sobel_map[py, px] < sobel_threshold: continue
            
        fx = int(px * (W_feat / W_orig))
        fy = int(py * (H_feat / H_orig))
        fx, fy = min(max(fx, 0), W_feat - 1), min(max(fy, 0), H_feat - 1)
        query_idx = fy * W_feat + fx
        
        corr_row = corr_matrix[0, query_idx, :]
        mask = torch.ones_like(corr_row) * float('-inf')
        win_x_min = max(0, int(fx - feat_search_radius))
        win_x_max = min(W_feat, int(fx + feat_search_radius))
        win_y_min = max(0, int(fy - feat_search_radius))
        win_y_max = min(H_feat, int(fy + feat_search_radius))
        
        mask_2d = mask.view(H_feat, W_feat)
        mask_2d[win_y_min:win_y_max, win_x_min:win_x_max] = 0
        masked_corr = corr_row + mask_2d.view(-1)
        
        max_idx = torch.argmax(masked_corr).item()
        match_fy = max_idx // W_feat
        match_fx = max_idx % W_feat
        match_px = int(match_fx * (W_orig / W_feat))
        match_py = int(match_fy * (H_orig / H_feat))
        
        valid_matches_src.append([px, py])
        valid_matches_dst.append([match_px, match_py])

    valid_matches_src = np.array(valid_matches_src, dtype=np.float32)
    valid_matches_dst = np.array(valid_matches_dst, dtype=np.float32)
    
    if len(valid_matches_src) < 3: return None, None

    M_affine, inliers = cv2.estimateAffinePartial2D(valid_matches_src, valid_matches_dst, method=cv2.RANSAC, ransacReprojThreshold=10.0)
    
    use_fallback = True
    final_M = None

    if M_affine is not None:
        a = M_affine[0, 0]
        b = M_affine[1, 0]
        scale = math.sqrt(a**2 + b**2)
        if 0.8 <= scale <= 1.2:
            final_M = M_affine
            use_fallback = False
    
    if use_fallback:
        deltas = valid_matches_dst - valid_matches_src
        dx = np.median(deltas[:, 0])
        dy = np.median(deltas[:, 1])
        final_M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)

    box_corners = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv2.transform(box_corners, final_M)
    
    return transformed_corners, inliers

# ==========================================
# Part 3: 辅助函数 (标签读取 & 误差计算 & 路径管理)
# ==========================================
def load_yolo_labels(txt_path):
    boxes = []
    if not os.path.exists(txt_path):
        print(f"错误: 找不到标签文件 {txt_path}")
        return boxes
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                box = [float(x) for x in parts[1:5]]
                boxes.append(box)
    return boxes

def get_distinct_colors(num_colors):
    cmap = plt.get_cmap('hsv')
    colors = [cmap(i) for i in np.linspace(0, 0.9, num_colors)]
    return colors

def calculate_errors(orig_box_norm, trans_corners, img_w, img_h):
    """
    计算原标签与匹配标签的误差
    """
    ncx, ncy, nw, nh = orig_box_norm
    gt_cx = ncx * img_w
    gt_cy = ncy * img_h
    gt_w = nw * img_w
    gt_h = nh * img_h
    
    pts = trans_corners.reshape(4, 2)
    pred_cx = np.mean(pts[:, 0])
    pred_cy = np.mean(pts[:, 1])
    
    top_w = np.linalg.norm(pts[1] - pts[0])
    bottom_w = np.linalg.norm(pts[2] - pts[3])
    pred_w = (top_w + bottom_w) / 2.0
    
    left_h = np.linalg.norm(pts[3] - pts[0])
    right_h = np.linalg.norm(pts[2] - pts[1])
    pred_h = (left_h + right_h) / 2.0
    
    center_error = math.sqrt((gt_cx - pred_cx)**2 + (gt_cy - pred_cy)**2)
    width_error = pred_w - gt_w
    height_error = pred_h - gt_h
    
    return {
        'gt_center': (gt_cx, gt_cy),
        'pred_center': (pred_cx, pred_cy),
        'gt_size': (gt_w, gt_h),
        'pred_size': (pred_w, pred_h),
        'center_error': center_error,
        'width_error': width_error,
        'height_error': height_error
    }

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """
    递增路径，例如: runs/exp -> runs/exp1 -> runs/exp2 ...
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        
        # 查找下一个可用的索引
        for n in range(1, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                path = Path(p)
                break
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return str(path)

def visualize_batch_results(rgb_img, ir_img, results, save_dir, img_id):
    """
    可视化并保存结果，文件名包含 img_id
    """
    H, W = rgb_img.shape[:2]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.imshow(rgb_img)
    ax1.set_title(f"Visible Input ({len(results)} targets)")
    
    ax2.imshow(ir_img)
    ax2.set_title("Infrared Result")

    for item in results:
        color = item['color']
        
        # 左图
        ncx, ncy, nw, nh = item['orig_box']
        pcx, pcy = int(ncx * W), int(ncy * H)
        pw, ph = int(nw * W), int(nh * H)
        rect = patches.Rectangle((pcx - pw//2, pcy - ph//2), pw, ph, 
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax1.add_patch(rect)

        # 右图
        if item['trans_box'] is not None:
            pts = item['trans_box'].reshape(-1, 2)
            poly = patches.Polygon(pts, linewidth=2, edgecolor=color, facecolor='none')
            ax2.add_patch(poly)
    
    ax1.axis('off')
    ax2.axis('off')
    plt.tight_layout()
    
    # 动态构建文件名
    save_name = f"batch_result_eval_{img_id}.png"
    save_path = os.path.join(save_dir, save_name)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"\n对比图已保存: {save_path}")
    # plt.show() # 如果跑批处理可以注释掉这一行，避免弹出窗口

# ==========================================
# 主流程
# ==========================================
def main():
    # 1. 设置路径
    rgb_path = './Datasets/01396_vi.png'
    ir_path = './Datasets/01396_ir.png'
    label_path = './Datasets/01396.txt' 

    # 提取 ID: 从 './Datasets/01396_vi.png' 提取 '01396'
    # 假设文件名格式为 "ID_xx.png" 或 "ID.png"
    filename = os.path.basename(rgb_path)
    img_id = filename.split('_')[0] if '_' in filename else filename.split('.')[0]

    # 2. 设置保存目录 (runs/exp, runs/exp1, ...)
    project = 'runs'
    name = 'exp'
    save_dir = increment_path(Path(project) / name, exist_ok=False, mkdir=True)
    print(f"结果将保存至: {save_dir}")

    print("3. 加载模型与图片...")
    model = VGGBlock3Extractor()
    model.eval()
    
    rgb_tensor, rgb_sobel, rgb_viz = process_image(rgb_path)
    ir_tensor, _, ir_viz = process_image(ir_path)
    if rgb_tensor is None: return

    print(f"4. 读取标签: {label_path}")
    all_boxes = load_yolo_labels(label_path)
    
    print("5. 计算全局特征...")
    with torch.no_grad():
        rgb_feat = model(rgb_tensor)
        ir_feat = model(ir_tensor)
    
    corr_matrix = correlation_layer(rgb_feat, ir_feat)
    feat_size = (rgb_feat.shape[2], rgb_feat.shape[3])
    orig_size = rgb_viz.shape[:2]
    img_h, img_w = orig_size

    print("\n" + "="*60)
    print(f"{'ID':<4} | {'GT Center':<15} | {'Pred Center':<15} | {'Diff(px)':<8} | {'Size Diff(W,H)':<15}")
    print("-" * 60)

    colors = get_distinct_colors(len(all_boxes))
    final_results = []
    total_center_error = 0
    valid_count = 0

    for i, box in enumerate(all_boxes):
        trans_corners, _ = step4_ransac_smart(
            corr_matrix, box, rgb_sobel, feat_size, orig_size
        )
        
        if trans_corners is not None:
            # 计算误差
            metrics = calculate_errors(box, trans_corners, img_w, img_h)
            
            # 打印信息
            gt_c_str = f"({metrics['gt_center'][0]:.0f}, {metrics['gt_center'][1]:.0f})"
            pred_c_str = f"({metrics['pred_center'][0]:.0f}, {metrics['pred_center'][1]:.0f})"
            diff_str = f"{metrics['center_error']:.2f}"
            size_diff_str = f"({metrics['width_error']:.1f}, {metrics['height_error']:.1f})"
            
            print(f"{i:<4} | {gt_c_str:<15} | {pred_c_str:<15} | {diff_str:<8} | {size_diff_str:<15}")
            
            total_center_error += metrics['center_error']
            valid_count += 1
            
            final_results.append({
                'orig_box': box,
                'trans_box': trans_corners,
                'color': colors[i]
            })
        else:
            print(f"{i:<4} | {'[Failed]':<15} | {'-':<15} | {'-':<8} | {'-':<15}")

    print("="*60)
    if valid_count > 0:
        avg_error = total_center_error / valid_count
        print(f"平均中心点误差: {avg_error:.2f} 像素")
    else:
        print("无有效配准目标。")

    # 传入保存目录和图片ID
    visualize_batch_results(rgb_viz, ir_viz, final_results, save_dir, img_id)

if __name__ == "__main__":
    main()