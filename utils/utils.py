import torch
import numpy as np

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def crop_like(data, like):
    """crop like"""
    if data.shape[-2:] != like.shape[-2:]:
        with torch.no_grad():
            dx, dy = data.shape[-2] - like.shape[-2], data.shape[-1] - like.shape[-1]
            data = data[:, :, dx // 2: - dx // 2, dy // 2: - dy // 2]
    return data

def tensor2img(image_tensor):

    def tonemap(matrix, gamma=2.2):
        return np.clip(matrix ** (1.0 / gamma), 0, 1)
    
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = tonemap(image_numpy) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    return image_numpy.astype(np.uint8)