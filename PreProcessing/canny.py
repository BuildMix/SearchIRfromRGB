import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 核心算法: 自适应阈值 Canny (保持不变) ---
def apply_auto_canny(gray_img, sigma=0.33):
    """
    根据图像的中值亮度自动计算高低阈值，应用 Canny。
    这样可以同时适应对比度较低的红外图和对比度较高的可见光图。
    """
    v = np.median(gray_img)
    
    # 限制阈值在 [0, 255] 之间
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    print(f"  [Canny自动阈值] 亮度中值:{v:.1f} -> 低阈值:{lower}, 高阈值:{upper}")
    
    # Canny 输出的是二值图像 (0或255)
    edged_img = cv2.Canny(gray_img, lower, upper)
    return edged_img

# --- 2. 图片加载与处理流程 ---
def load_and_process_images(rgb_path, ir_path):
    if not os.path.exists(rgb_path) or not os.path.exists(ir_path):
        print("错误: 找不到输入图片文件。请确保 visible.png 和 infrared.png 存在。")
        return None

    # 读取原图
    orig_rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    orig_ir_gray = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

    if orig_rgb_bgr is None or orig_ir_gray is None:
        print("错误: 图片解码失败。")
        return None

    # 准备展示用的数据 (BGR转RGB)
    display_rgb = cv2.cvtColor(orig_rgb_bgr, cv2.COLOR_BGR2RGB)
    display_ir = orig_ir_gray

    # 准备处理用的数据 (转灰度)
    rgb_gray_for_proc = cv2.cvtColor(orig_rgb_bgr, cv2.COLOR_BGR2GRAY)
    
    print("正在处理可见光图像...")
    canny_rgb_result = apply_auto_canny(rgb_gray_for_proc)
    
    print("正在处理红外图像...")
    canny_ir_result = apply_auto_canny(orig_ir_gray)

    return display_rgb, display_ir, canny_rgb_result, canny_ir_result

# --- 3. 主函数 (包含保存逻辑) ---
def main():
    # --- 配置部分 ---
    rgb_filename = 'visible.png'
    ir_filename = 'infrared.png'
    
    # 四宫格大图的输出文件名
    output_grid_filename = 'output_canny.png' 
    # ----------------

    results = load_and_process_images(rgb_filename, ir_filename)
    if results is None:
        return
    
    disp_rgb, disp_ir, res_rgb, res_ir = results

    # --- 保存单独的结果图 ---
    cv2.imwrite('output_canny_visible.png', res_rgb)
    cv2.imwrite('output_canny_infrared.png', res_ir)
    print("\n[1/2] 已保存单独的 Canny 结果图。")

    # --- Matplotlib 四宫格绘制 ---
    fig = plt.figure(figsize=(12, 10)) # 设置画布大小
    
    # 子图 1: 可见光原图
    plt.subplot(2, 2, 1)
    plt.title("Original Visible (RGB)")
    plt.imshow(disp_rgb)
    plt.axis('off')

    # 子图 2: 红外原图
    plt.subplot(2, 2, 2)
    plt.title("Original Infrared")
    plt.imshow(disp_ir, cmap='gray')
    plt.axis('off')

    # 子图 3: 可见光 Canny 结果
    plt.subplot(2, 2, 3)
    plt.title("Canny Edges (Visible)")
    plt.imshow(res_rgb, cmap='gray') 
    plt.axis('off')

    # 子图 4: 红外 Canny 结果
    plt.subplot(2, 2, 4)
    plt.title("Canny Edges (Infrared)")
    plt.imshow(res_ir, cmap='gray')
    plt.axis('off')

    plt.tight_layout() # 自动调整间距

    # --- 保存四宫格大图 ---
    print(f"正在保存四宫格对比图到 {output_grid_filename} ...")
    plt.savefig(output_grid_filename, format='png', bbox_inches='tight', dpi=150)
    print(f"[2/2] 四宫格对比图已成功保存！")

    # (可选) 显示窗口，如果不想弹窗可注释掉
    # plt.show() 

if __name__ == "__main__":
    main()