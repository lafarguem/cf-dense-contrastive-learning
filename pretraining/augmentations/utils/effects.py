import math
import random
from torchvision.transforms import functional as TF
import torch

class GaussianBlur(object):
    def __call__(self, img):
        sigma = random.uniform(0.1, 2.0)
        radius = math.ceil(2 * sigma)
        kernel_size = 2 * radius + 1
        return TF.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
    
class RandomSolarizeFloat:
    def __init__(self, threshold=0.5, p=0.5):
        self.threshold = threshold
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = torch.where(img >= self.threshold, 1.0 - img, img)
        return img