import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

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
        return None, None, None
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

# ==========================================
# Part 2: Step 3 相关度计算
# ==========================================
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
# Part 3: 三联图可视化
# ==========================================
def save_visualizations(corr_matrix, norm_box, feat_size, orig_size, rgb_viz, ir_viz):
    """
    保存三联图: 1.RGB原图框  2.IR热力图框  3.IR原图框(干净)
    """
    B, N, N_dim = corr_matrix.shape
    H_feat, W_feat = feat_size
    H_orig, W_orig = orig_size

    # --- 1. 坐标转换 ---
    ncx, ncy, nw, nh = norm_box
    pcx, pcy = int(ncx * W_orig), int(ncy * H_orig)
    pw, ph  = int(nw * W_orig), int(nh * H_orig)
    
    # RGB 框左上角
    px_min = pcx - pw // 2
    py_min = pcy - ph // 2

    # --- 2. 匹配计算 ---
    fx = int(pcx * (W_feat / W_orig))
    fy = int(pcy * (H_feat / H_orig))
    fx = min(max(fx, 0), W_feat - 1)
    fy = min(max(fy, 0), H_feat - 1)
    query_idx = fy * W_feat + fx

    correlation_map_flat = corr_matrix[0, query_idx, :]
    correlation_map = correlation_map_flat.view(H_feat, W_feat).cpu().numpy()
    heatmap_resized = cv2.resize(correlation_map, (W_orig, H_orig), interpolation=cv2.INTER_CUBIC)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap_resized)
    match_cx, match_cy = max_loc

    # IR 框左上角
    match_x_min = match_cx - pw // 2
    match_y_min = match_cy - ph // 2

    print(f"RGB中心: ({pcx}, {pcy}) -> IR匹配中心: ({match_cx}, {match_cy})")

    # --- 3. 绘图 (三联) ---
    # 设置画布宽度更宽一些 (21, 7) 适应三张图
    fig1 = plt.figure(figsize=(21, 7))
    
    # [左图] RGB Input
    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title("1. Visible Input (Label)")
    ax1.imshow(rgb_viz)
    rect_rgb = patches.Rectangle((px_min, py_min), pw, ph, linewidth=3, edgecolor='r', facecolor='none')
    ax1.add_patch(rect_rgb)
    ax1.plot(pcx, pcy, 'r+', markersize=15, markeredgewidth=2)
    ax1.axis('off')

    # [中图] IR Heatmap + Box
    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title(f"2. IR Heatmap Analysis")
    ax2.imshow(ir_viz, cmap='gray')
    ax2.imshow(heatmap_resized, cmap='jet', alpha=0.5)
    # 注意: Matplotlib 的 patch 对象不能重复使用，必须新建一个
    rect_ir_heat = patches.Rectangle((match_x_min, match_y_min), pw, ph, linewidth=3, edgecolor='#00FF00', facecolor='none', linestyle='--')
    ax2.add_patch(rect_ir_heat)
    ax2.plot(match_cx, match_cy, 'go', markersize=10)
    ax2.axis('off')

    # [右图] IR Result (Clean) --- 新增的部分 ---
    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title(f"3. Final Result (Infrared)")
    ax3.imshow(ir_viz, cmap='gray') # 只显示原图，不显示热力图
    # 新建一个绿框 Patch
    rect_ir_clean = patches.Rectangle((match_x_min, match_y_min), pw, ph, linewidth=3, edgecolor='#00FF00', facecolor='none', linestyle='--')
    ax3.add_patch(rect_ir_clean)
    ax3.axis('off')

    plt.tight_layout()
    save_path1 = "result_visual_comparison.png"
    plt.savefig(save_path1, bbox_inches='tight', dpi=150)
    print(f"三联对比图已保存: {save_path1}")
    
    # --- 4. 绘图 2: 全局矩阵 (单独保存，保持不变) ---
    full_matrix = corr_matrix[0].cpu().numpy()
    fig2 = plt.figure(figsize=(10, 10))
    plt.title(f"Global Correlation Matrix ({N_dim}x{N_dim})")
    im = plt.imshow(full_matrix, cmap='viridis', aspect='auto', origin='upper')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    save_path2 = "result_global_matrix.png"
    plt.savefig(save_path2, bbox_inches='tight', dpi=150)
    print(f"全局矩阵图已保存: {save_path2}")

# ==========================================
# 主流程
# ==========================================
def main():
    rgb_path = 'visible.png'
    ir_path = 'infrared.png'
    input_box_norm = [0.8251953125, 0.52734375, 0.23828125, 0.30729166666666663]

    print("加载模型...")
    model = VGGBlock3Extractor()
    model.eval()

    print("Step 1 & 2: 预处理与特征提取...")
    rgb_tensor, _, rgb_orig = process_image(rgb_path)
    ir_tensor, _, ir_orig = process_image(ir_path)
    
    if rgb_tensor is None or ir_tensor is None: return

    with torch.no_grad():
        rgb_feat = model(rgb_tensor)
        ir_feat = model(ir_tensor)

    print("Step 3: 计算相关性与可视化...")
    corr_matrix = correlation_layer(rgb_feat, ir_feat)
    
    save_visualizations(
        corr_matrix, 
        input_box_norm,
        feat_size=(rgb_feat.shape[2], rgb_feat.shape[3]), 
        orig_size=rgb_orig.shape[:2], 
        rgb_viz=rgb_orig, 
        ir_viz=ir_orig
    )

if __name__ == "__main__":
    main()