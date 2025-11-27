import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 核心算法: Sobel (保持不变) ---
def apply_sobel_to_gray(gray_img):
    """
    接收一个灰度图像数组，应用 Sobel 算子，返回处理后的图像数组。
    """
    # 1. 计算 Sobel (使用 float64 防止负梯度截断)
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

    # 2. 取绝对值并转回 uint8 (0-255)
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)

    # 3. 融合 X 和 Y 方向的梯度
    sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    
    return sobel_combined

# --- 2. 图片加载与处理流程 ---
def load_and_process_images(rgb_path, ir_path):
    # 检查文件
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
    
    print("正在计算 Sobel 梯度 (可见光)...")
    sobel_rgb_result = apply_sobel_to_gray(rgb_gray_for_proc)
    
    print("正在计算 Sobel 梯度 (红外)...")
    sobel_ir_result = apply_sobel_to_gray(orig_ir_gray)

    return display_rgb, display_ir, sobel_rgb_result, sobel_ir_result

# --- 3. 主函数 (包含保存逻辑) ---
def main():
    # --- 配置部分 ---
    rgb_filename = 'visible.png'
    ir_filename = 'infrared.png'
    
    # 四宫格大图的输出文件名
    output_grid_filename = 'output_sobel.png'
    # ----------------

    results = load_and_process_images(rgb_filename, ir_filename)
    if results is None:
        return
    
    disp_rgb, disp_ir, res_rgb, res_ir = results

    # --- 保存单独的结果图 ---
    cv2.imwrite('output_sobel_visible.png', res_rgb)
    cv2.imwrite('output_sobel_infrared.png', res_ir)
    print("\n[1/2] 已保存单独的 Sobel 结果图。")

    # --- Matplotlib 四宫格绘制 ---
    # 设置画布大小
    fig = plt.figure(figsize=(12, 10))
    
    # 子图 1: 可见光原图
    plt.subplot(2, 2, 1)
    plt.title("Original Visible (RGB)")
    plt.imshow(disp_rgb) # 彩色图不需要 cmap='gray'
    plt.axis('off')

    # 子图 2: 红外原图
    plt.subplot(2, 2, 2)
    plt.title("Original Infrared")
    plt.imshow(disp_ir, cmap='gray') # 灰度图需要指定 cmap
    plt.axis('off')

    # 子图 3: 可见光 Sobel 结果
    plt.subplot(2, 2, 3)
    plt.title("Sobel Result (Visible)")
    plt.imshow(res_rgb, cmap='gray')
    plt.axis('off')

    # 子图 4: 红外 Sobel 结果
    plt.subplot(2, 2, 4)
    plt.title("Sobel Result (Infrared)")
    plt.imshow(res_ir, cmap='gray')
    plt.axis('off')

    # 自动调整布局
    plt.tight_layout()
    
    # --- 保存四宫格大图 ---
    print(f"正在保存四宫格对比图到 {output_grid_filename} ...")
    plt.savefig(output_grid_filename, format='png', bbox_inches='tight', dpi=150)
    print(f"[2/2] 四宫格对比图已成功保存！")

    # (可选) 显示窗口
    # plt.show()

if __name__ == "__main__":
    main()