# core/feature_extractor.py
import torch.nn as nn
from torchvision import models

class VGGBlock1Extractor(nn.Module):
    def __init__(self):
        super(VGGBlock1Extractor, self).__init__()
        # 加载预训练权重
        full_vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # 结构: Conv64 -> ReLU -> Conv64 -> ReLU -> MaxPool
        self.features = full_vgg16.features[:5]
        
        # InstanceNorm 去除对比度差异
        self.inst_norm = nn.InstanceNorm2d(64, affine=False)
        
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        feat = self.features(x)
        feat = self.inst_norm(feat)
        return feat