import torch
import numpy as np
import cv2
from torchvision import transforms


# Simple Grad-CAM for torchvision models
class GradCAM:
def __init__(self, model, target_layer):
self.model = model.eval()
self.gradients = None
self.activations = None
target_layer.register_forward_hook(self.save_activation)
target_layer.register_backward_hook(self.save_gradient)


def save_activation(self, module, input, output):
self.activations = output.detach()


def save_gradient(self, module, grad_input, grad_output):
self.gradients = grad_output[0].detach()


def __call__(self, input_tensor, class_idx=None):
self.model.zero_grad()
output = self.model(input_tensor)
if class_idx is None:
class_idx = output.argmax(dim=1).item()
score = output[0, class_idx]
score.backward()


weights = self.gradients.mean(dim=(2,3), keepdim=True)
cam = (weights * self.activations).sum(dim=1, keepdim=True)