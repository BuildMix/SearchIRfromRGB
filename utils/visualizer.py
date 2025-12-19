import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def get_distinct_colors(num_colors):
    cmap = plt.get_cmap('hsv')
    colors = [cmap(i) for i in np.linspace(0, 0.9, num_colors)]
    return colors

def visualize_batch_results(rgb_img, ir_img, results, save_dir, img_id):
    """可视化并保存结果"""
    H, W = rgb_img.shape[:2]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.imshow(rgb_img)
    ax1.set_title(f"Visible ({img_id}) - {len(results)} targets")
    
    ax2.imshow(ir_img)
    ax2.set_title(f"Infrared ({img_id}) - Result")

    for item in results:
        color = item['color']
        
        # 左图：GT Box
        ncx, ncy, nw, nh = item['orig_box']
        pcx, pcy = int(ncx * W), int(ncy * H)
        pw, ph = int(nw * W), int(nh * H)
        rect = patches.Rectangle((pcx - pw//2, pcy - ph//2), pw, ph, 
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax1.add_patch(rect)

        # 右图：预测 Box
        if item['trans_box'] is not None:
            pts = item['trans_box'].reshape(-1, 2)
            poly = patches.Polygon(pts, linewidth=2, edgecolor=color, facecolor='none')
            ax2.add_patch(poly)
    
    ax1.axis('off')
    ax2.axis('off')
    plt.tight_layout()
    
    save_name = f"{img_id}_results.png"
    save_path = os.path.join(save_dir, save_name)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)