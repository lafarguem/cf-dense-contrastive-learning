"""Code inspired by PixPro: https://github.com/zdaxie/PixPro
paper: https://ieeexplore.ieee.org/document/9578721
"""
from __future__ import division
import torch
import math
import random
from PIL import Image
import warnings
from pretraining.utils.debug import save_mask
from torchvision.transforms import functional as TF


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}

def _get_image_size(img):
    if TF._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() >= 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask=None):
        M = torch.eye(3, dtype=torch.float32)
        for t in self.transforms:
            class_name = t.__class__.__name__
            if 'RandomResizedCropCoord' in class_name or \
            'ResizeCoord' in class_name or \
            'CenterCropCoord' in class_name or \
            'RandomRotateCoord' in class_name:
                img, mask, M = t(img, M, mask)
            elif 'FlipCoord' in class_name:
                img, mask, M = t(img, M, mask)
            elif 'ToTensor' in class_name:
                img = t(img)
                if mask is not None:
                    mask = t(mask)
            else:
                img = t(img)

        return img, mask, M

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomHorizontalFlipCoord(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, M, mask=None):
        if random.random() < self.p:
            T_flip = torch.tensor([
                [-1, 0, 1],
                [0, 1, 0],
                [0, 0, 1]
            ], dtype=torch.float32)

            M = T_flip @ M
            img = TF.hflip(img)
            if mask is not None:
                mask = TF.hflip(mask)

        return img, mask, M

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlipCoord(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, M, mask=None):
        if random.random() < self.p:
            T_flip = torch.tensor([
                [1, 0, 0],
                [0, -1, 1],
                [0, 0, 1]
            ], dtype=torch.float32)

            M = T_flip @ M
            img = TF.vflip(img)
            if mask is not None:
                mask = TF.vflip(mask)

        return img, mask, M

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomResizedCropCoord:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.), interpolation=Image.BILINEAR):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def get_params(self, img, scale, ratio):
        width, height = _get_image_size(img)

        area = height * width
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round((target_area * aspect_ratio) ** 0.5))
            h = int(round((target_area / aspect_ratio) ** 0.5))

            if w <= width and h <= height:
                i = torch.randint(0, height - h + 1, (1,)).item()
                j = torch.randint(0, width - w + 1, (1,)).item()
                return i, j, h, w, height, width

        in_ratio = width / height
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w, height, width

    def __call__(self, img, M, mask=None):
        i, j, h, w, height, width = self.get_params(img, self.scale, self.ratio)

        left = j / width
        top = i / height
        right = (j + w) / width
        bottom = (i + h) / height

        T_crop = torch.tensor([
            [1, 0, -left],
            [0, 1, -top],
            [0, 0, 1]
        ], dtype=torch.float32)

        sx = 1.0 / (right - left)
        sy = 1.0 / (bottom - top)
        T_scale = torch.tensor([
            [sx, 0, 0],
            [0, sy, 0],
            [0,  0, 1]
        ], dtype=torch.float32)

        M_new = T_scale @ T_crop @ M

        img = TF.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        if mask is not None:
            mask = TF.resized_crop(mask, i, j, h, w, self.size, Image.NEAREST)

        return img, mask, M_new

class ResizeCoord:
    def __init__(self, size, interpolation=Image.BILINEAR):
        if isinstance(size, int):
            self.size = size
        else:
            assert len(size) == 2
            self.size = size
        self.interpolation = interpolation

    def __call__(self, img, M, mask=None):
        width, height = _get_image_size(img)

        if isinstance(self.size, int):
            if width < height:
                new_width = self.size
                new_height = int(height * self.size / width)
            else:
                new_height = self.size
                new_width = int(width * self.size / height)
        else:
            new_width, new_height = self.size

        img = TF.resize(img, [new_height, new_width], interpolation=self.interpolation)
        if mask is not None:
            mask = TF.resize(mask, [new_height, new_width], interpolation=Image.NEAREST)

        return img, mask, M

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


class CenterCropCoord(object):
    """Crops the given PIL Image at the center to the given size, returns transformation matrix M in normalized coords."""

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2, "Size should be int or tuple/list of length 2"
            self.size = size

    def __call__(self, img, M, mask=None):
        orig_width, orig_height = _get_image_size(img)
        crop_w, crop_h = self.size

        left = (orig_width - crop_w) / 2.0
        top = (orig_height - crop_h) / 2.0
        right = left + crop_w
        bottom = top + crop_h

        img = TF.crop(img, int(top), int(left), crop_h, crop_w)
        if mask is not None:
            mask = TF.crop(mask, int(top), int(left), crop_h, crop_w)

        left_norm = left / orig_width
        top_norm = top / orig_height
        right_norm = right / orig_width
        bottom_norm = bottom / orig_height

        w_crop = right_norm - left_norm
        h_crop = bottom_norm - top_norm

        T_scale = torch.tensor([
            [1/w_crop, 0, -left_norm/w_crop],
            [0, 1/h_crop, -top_norm/h_crop],
            [0, 0, 1]
        ])
        M_new = T_scale@M

        return img, mask, M_new

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"

class RandomRotateCoord(object):
    """
    Randomly rotate the image by an angle between (-degrees, +degrees).
    Rotation is counter-clockwise. Updates transformation matrix M.
    """

    def __init__(self, degrees):
        if isinstance(degrees, (tuple, list)):
            assert len(degrees) == 2, "degrees must be (min, max)"
            self.degrees = degrees
        else:
            self.degrees = (-degrees, degrees)

    def get_params(self):
        """Get a random angle in degrees."""
        return random.uniform(self.degrees[0], self.degrees[1])

    def __call__(self, img, M, mask=None):
        angle = self.get_params()
        angle_rad = math.radians(angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)

        T1 = torch.tensor([
            [1, 0, -0.5],
            [0, 1, -0.5],
            [0, 0, 1]
        ], dtype=torch.float32)

        T2 = torch.tensor([
            [1, 0, 0.5],
            [0, 1, 0.5],
            [0, 0, 1]
        ], dtype=torch.float32)

        R = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        M_norm = T2 @ R @ T1

        M = M_norm @ M

        img = TF.rotate(img, angle, interpolation=Image.BILINEAR, expand=False)
        if mask is not None:
            mask = TF.rotate(mask, angle, interpolation=Image.NEAREST, expand=False)

        return img, mask, M

    def __repr__(self):
        return f"{self.__class__.__name__}(degrees={self.degrees})"