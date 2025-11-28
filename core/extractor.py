import torch.nn as nn
from torchvision import models

class VGGBlock3Extractor(nn.Module):
    def __init__(self):
        super(VGGBlock3Extractor, self).__init__()
        # 加载预训练模型
        full_vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # 截取前17层（对应 block3_pool 之前）
        self.features = full_vgg16.features[:17]
        # 实例归一化，去除风格/光照影响
        self.inst_norm = nn.InstanceNorm2d(256, affine=False)
        # 冻结参数
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        feat = self.features(x)
        feat = self.inst_norm(feat)
        return feat