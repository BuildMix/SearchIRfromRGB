import cv2
import numpy as np

def get_ir_gradient(image_path):
    # 1. 读取红外图像 (灰度模式)
    img = cv2.imread(image_path, 0)
    if img is None:
        print("未找到图像")
        return

    # 2. 预处理：强烈建议先降噪
    # 高斯模糊适合一般情况，双边滤波(Bilateral)适合保留强边缘
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # ---------------------------------------------------------
    # 方法 A: 改进的 Sobel 梯度法 (适合细节提取)
    # ---------------------------------------------------------
    # 使用 CV_64F 保留负梯度信息
    grad_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=-1)
    grad_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=-1)

    # 计算梯度幅值 (Magnitude)
    grad_magnitude = cv2.magnitude(grad_x, grad_y)

    # 归一化到 0-255 (这对红外图像非常重要，否则很暗)
    grad_sobel_norm = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    grad_sobel_uint8 = np.uint8(grad_sobel_norm)

    # ---------------------------------------------------------
    # 方法 B: 形态学梯度法 (适合整体轮廓，抗噪性好)
    # ---------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 梯度 = 膨胀 - 腐蚀
    grad_morph = cv2.morphologyEx(img_blur, cv2.MORPH_GRADIENT, kernel)
    
    # 同样增强对比度
    grad_morph_norm = cv2.normalize(grad_morph, None, 0, 255, cv2.NORM_MINMAX)
    grad_morph_uint8 = np.uint8(grad_morph_norm)

    # ---------------------------------------------------------
    # 可视化结果
    # ---------------------------------------------------------
    cv2.imwrite('./Preprocessing/result_sobel.png', grad_sobel_uint8)
    cv2.imwrite('./Preprocessing/result_morph.png', grad_morph_uint8)
    
    print("处理完成！图片已保存到 Preprocessing 文件夹下。")

# 使用示例
get_ir_gradient('./Preprocessing/infrared.png')