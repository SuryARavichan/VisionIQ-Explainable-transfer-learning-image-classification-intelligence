# modeling.py
import torch
import torch.nn as nn
from torchvision import models




def get_model(model_name='efficientnet_b0', num_classes=2, pretrained=True):
"""Return a model with a custom classification head."""
model_name = model_name.lower()
if 'efficientnet' in model_name:
# use torchvision efficientnet
backbone = getattr(models, model_name)(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
in_features = backbone.classifier[1].in_features
# replace head
backbone.classifier = nn.Sequential(
nn.Dropout(p=0.3),
nn.Linear(in_features, num_classes)
)
return backbone
elif 'resnet' in model_name:
backbone = getattr(models, model_name)(pretrained=pretrained)
in_features = backbone.fc.in_features
backbone.fc = nn.Sequential(
nn.BatchNorm1d(in_features),
nn.Dropout(0.4),
nn.Linear(in_features, num_classes)
)
return backbone
else:
# fallback: simple small CNN
model = nn.Sequential(
nn.Conv2d(3, 32, 3, padding=1),
nn.ReLU(),
nn.MaxPool2d(2),
nn.Conv2d(32, 64, 3, padding=1),
nn.ReLU(),
nn.AdaptiveAvgPool2d(1),
nn.Flatten(),
nn.Linear(64, num_classes)
)
return model