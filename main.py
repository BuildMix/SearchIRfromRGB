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
# Part 2: Step 4 智能缩放版 (Smart Scale RANSAC)
# ==========================================
def step4_ransac_smart(corr_matrix, norm_box, sobel_map, feat_size, orig_size):
    B, N, N_dim = corr_matrix.shape
    H_feat, W_feat = feat_size
    H_orig, W_orig = orig_size
    
    # 1. 解析 YOLO 框
    ncx, ncy, nw, nh = norm_box
    pcx, pcy = int(ncx * W_orig), int(ncy * H_orig)
    pw, ph = int(nw * W_orig), int(nh * H_orig)
    x_min, y_min = pcx - pw//2, pcy - ph//2
    x_max, y_max = pcx + pw//2, pcy + ph//2

    # 2. 采样 (网格点)
    grid_size = 6
    xs = np.linspace(x_min + pw*0.1, x_max - pw*0.1, grid_size)
    ys = np.linspace(y_min + ph*0.1, y_max - ph*0.1, grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points_rgb = np.vstack((grid_x.flatten(), grid_y.flatten())).T

    valid_matches_src = []
    valid_matches_dst = []
    
    # 低阈值 + 空间约束
    sobel_threshold = 15 
    search_radius = W_orig // 3 
    feat_search_radius = search_radius * (W_feat / W_orig)

    print(f"\n--- RANSAC 过程 (智能缩放版) ---")
    
    for pt in points_rgb:
        px, py = int(pt[0]), int(pt[1])
        if px < 0 or px >= W_orig or py < 0 or py >= H_orig: continue
        if sobel_map[py, px] < sobel_threshold: continue
            
        fx = int(px * (W_feat / W_orig))
        fy = int(py * (H_feat / H_orig))
        fx, fy = min(max(fx, 0), W_feat - 1), min(max(fy, 0), H_feat - 1)
        query_idx = fy * W_feat + fx
        
        # 空间掩码
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
    
    print(f"有效匹配点数: {len(valid_matches_src)}")
    if len(valid_matches_src) < 3: return None, None

    # --- 3. 核心改进：带检查的仿射变换 ---
    
    # A. 尝试计算仿射矩阵 (允许平移+旋转+缩放)
    M_affine, inliers = cv2.estimateAffinePartial2D(valid_matches_src, valid_matches_dst, method=cv2.RANSAC, ransacReprojThreshold=10.0)
    
    use_fallback = True
    final_M = None

    if M_affine is not None:
        # B. 提取缩放系数 (Scale)
        # M = [[a, -b, tx], [b, a, ty]]
        # scale = sqrt(a^2 + b^2)
        a = M_affine[0, 0]
        b = M_affine[1, 0]
        scale = math.sqrt(a**2 + b**2)
        
        print(f"RANSAC 计算出的缩放系数: {scale:.4f}")

        # C. 熔断检查 (Sanity Check)
        # 我们允许 0.8 ~ 1.2 的缩放。超出这个范围认为是不合理的漂移。
        if 0.8 <= scale <= 1.2:
            print(">> 缩放系数合理，采用 RANSAC 结果。")
            final_M = M_affine
            use_fallback = False
        else:
            print(">> 警告：缩放系数异常 (过大或过小)！触发熔断机制。")
    
    # D. 自动降级 (Fallback): 中值流 (仅平移)
    if use_fallback:
        print(">> 启动 B 计划：仅计算平移 (Scale=1.0)")
        deltas = valid_matches_dst - valid_matches_src
        dx = np.median(deltas[:, 0])
        dy = np.median(deltas[:, 1])
        final_M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        # 伪造一个全 1 的 inliers，因为中值对所有点都有效
        inliers = np.ones((len(valid_matches_src), 1), dtype=np.uint8)

    # 4. 变换原始框
    box_corners = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv2.transform(box_corners, final_M)
    
    return transformed_corners, inliers

# ==========================================
# Part 3: 可视化 (保持不变)
# ==========================================
def visualize_final_comparison(rgb_img, ir_img, trans_corners, norm_box):
    H, W = rgb_img.shape[:2]
    ncx, ncy, nw, nh = norm_box
    pcx, pcy = int(ncx * W), int(ncy * H)
    pw, ph = int(nw * W), int(nh * H)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    ax1.imshow(rgb_img)
    ax1.set_title("Original Input (Visible)")
    rect_rgb = patches.Rectangle((pcx - pw//2, pcy - ph//2), pw, ph, linewidth=3, edgecolor='yellow', facecolor='none')
    ax1.add_patch(rect_rgb)
    ax1.axis('off')

    ax2.imshow(ir_img)
    ax2.set_title("Robust Registration Result (Infrared)")
    pts = trans_corners.reshape(-1, 2)
    poly = patches.Polygon(pts, linewidth=3, edgecolor='#00FF00', facecolor='none')
    ax2.add_patch(poly)
    ax2.axis('off')

    plt.tight_layout()
    save_path = "final_result.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"对比结果已保存: {save_path}")
    plt.show()

# ==========================================
# 主流程
# ==========================================
def main():
    rgb_path = './Datasets/visible.png'
    ir_path = './Datasets/infrared.png'
    input_box_norm = [0.5546875, 0.5716145833333333, 0.142578125, 0.18489583333333331]

    print("加载模型...")
    model = VGGBlock3Extractor()
    model.eval()

    print("Step 1-3...")
    rgb_tensor, rgb_sobel, rgb_viz = process_image(rgb_path)
    ir_tensor, _, ir_viz = process_image(ir_path)
    if rgb_tensor is None: return

    with torch.no_grad():
        rgb_feat = model(rgb_tensor)
        ir_feat = model(ir_tensor)

    corr_matrix = correlation_layer(rgb_feat, ir_feat)

    print("Step 4: RANSAC (智能缩放版)...")
    feat_size = (rgb_feat.shape[2], rgb_feat.shape[3])
    orig_size = rgb_viz.shape[:2]
    
    trans_corners, inliers = step4_ransac_smart(
        corr_matrix, input_box_norm, rgb_sobel, feat_size, orig_size
    )
    
    if trans_corners is not None:
        visualize_final_comparison(rgb_viz, ir_viz, trans_corners, input_box_norm)
    else:
        print("配准失败。")

if __name__ == "__main__":
    main()