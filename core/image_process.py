# core/image_process.py
import cv2
import os
from torchvision import transforms

def apply_sobel_algorithm(gray_img):
    """计算梯度图"""
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    abs_x = cv2.convertScaleAbs(sobel_x)
    abs_y = cv2.convertScaleAbs(sobel_y)
    return cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

def process_image(image_path):
    """读取图像并转换为模型输入 Tensor"""
    if not os.path.exists(image_path):
        return None, None, None
    orig_img = cv2.imread(image_path)
    if orig_img is None: 
        return None, None, None
        
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    sobel_img = apply_sobel_algorithm(gray_img) 
    
    # 构造伪彩色输入适配 VGG
    sobel_rgb = cv2.merge([sobel_img, sobel_img, sobel_img])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(sobel_rgb).unsqueeze(0)
    
    # 返回: Tensor(VGG输入), Sobel图(筛选用), 原图(可视化用)
    return img_tensor, sobel_img, cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)