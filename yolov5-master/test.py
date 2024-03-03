import torch
from models.yolo import Model  # 确保这个路径是你 YOLOv5 模型定义文件的路径
# 加载模型权重
weights_path = 'I:/yolov5-master/yolov5-master/runs/train/exp97-WAIPIN_pram/weights/best.pt'  # 替换为你的权重文件路径
model = torch.load(weights_path)['model']  # 加载模型

print(model)
