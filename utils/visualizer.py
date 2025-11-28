# utils/visualizer.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import math

def get_distinct_colors(num_colors):
    cmap = plt.get_cmap('hsv')
    return [cmap(i) for i in np.linspace(0, 0.9, num_colors)]

def calculate_errors(orig_box_norm, trans_corners, img_w, img_h):
    """计算中心点误差和尺寸误差"""
    ncx, ncy, nw, nh = orig_box_norm
    gt_cx, gt_cy = ncx * img_w, ncy * img_h
    gt_w, gt_h = nw * img_w, nh * img_h
    
    pts = trans_corners.reshape(4, 2)
    pred_cx = np.mean(pts[:, 0])
    pred_cy = np.mean(pts[:, 1])
    
    pred_w = (np.linalg.norm(pts[1]-pts[0]) + np.linalg.norm(pts[2]-pts[3])) / 2.0
    pred_h = (np.linalg.norm(pts[3]-pts[0]) + np.linalg.norm(pts[2]-pts[1])) / 2.0
    
    center_error = math.sqrt((gt_cx - pred_cx)**2 + (gt_cy - pred_cy)**2)
    
    return {
        'gt_center': (gt_cx, gt_cy),
        'pred_center': (pred_cx, pred_cy),
        'center_error': center_error,
        'width_error': pred_w - gt_w,
        'height_error': pred_h - gt_h
    }

def visualize_batch_results(rgb_img, ir_img, results, save_dir, img_id):
    """画图并保存"""
    H, W = rgb_img.shape[:2]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.imshow(rgb_img)
    ax1.set_title(f"Visible ({img_id})")
    ax2.imshow(ir_img)
    ax2.set_title(f"Infrared Result")

    for item in results:
        color = item['color']
        # 画左图 (原图框)
        ncx, ncy, nw, nh = item['orig_box']
        pcx, pcy = int(ncx * W), int(ncy * H)
        pw, ph = int(nw * W), int(nh * H)
        rect = patches.Rectangle((pcx - pw//2, pcy - ph//2), pw, ph, linewidth=2, edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        # 画右图 (变换框)
        if item['trans_box'] is not None:
            pts = item['trans_box'].reshape(-1, 2)
            poly = patches.Polygon(pts, linewidth=2, edgecolor=color, facecolor='none')
            ax2.add_patch(poly)
    
    ax1.axis('off'); ax2.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{img_id}_result.png"), bbox_inches='tight', dpi=150)
    plt.close(fig)