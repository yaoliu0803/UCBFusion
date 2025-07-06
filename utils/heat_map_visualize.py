# import python package
import cv2
import numpy as np
import matplotlib.pyplot as plt

# import external package
import torch
from torchvision import transforms
'''
# -----------------------------------------------------------
heat map visualize(pixel color in the heat map)
weight of warm color is larger than that of cold color.
# -----------------------------------------------------------
'''


# 假设你已经有了一个1通道的权重图，这里我们用一个随机生成的张量来模拟
# 权重图的形状应该是 [1, H, W]，其中 H 和 W 是图像的高度和宽度
img_path = "../test/feature_map/feature2_128_0.jpg"
img_cv = cv2.imread(img_path, 0)
tran = transforms.ToTensor()
img_tensor = tran(img_cv)
img_tensor = img_tensor.squeeze(0)

print("img_tensor=", img_tensor.shape)
# img_tensor_W, img_tensor_H = img_tensor.shape[1], img_tensor.shape[2]

# 假设heatmap是一个单通道的2D张量，可以是模型输出的某个特征图
# 这里使用随机生成的数据作为示例
# heatmap = torch.rand((256, 256))
heatmap = img_tensor
# 将张量的数据转换为numpy数组
heatmap_np = heatmap.numpy()

# 使用matplotlib绘制热力图
plt.imshow(heatmap_np, cmap='hot', interpolation='bilinear')
plt.colorbar()  # 添加颜色条
plt.show()
print('saved')
