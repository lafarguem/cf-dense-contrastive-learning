from pretraining.augmentations.base_transform import BaseTransform
import torchvision.transforms as transforms

from pretraining.augmentations.utils.rand_augment import rand_augment_transform
import pretraining.augmentations.utils.transform_coord as transform_coord
from pretraining.augmentations.utils.effects import GaussianBlur, RandomSolarizeFloat

class ModularTransform(BaseTransform):
    def __init__(self, image_size, normalize, **kwargs):
        self.normalize = transforms.Normalize(**normalize)
        self.image_size = image_size
        self.kwargs = kwargs

    def view_transform(self, view, view2=False):
        transform_list = []
        if (angle := view.get('rotate')) is not None and angle > 0.:
            if isinstance(angle, bool):
                angle = 15
            transform_list.append(transform_coord.RandomRotateCoord(angle))
        if (crop := view.get('crop')) is not None and crop > 0.:
            if isinstance(crop, bool):
                crop = 0.08
            transform_list.append(transform_coord.RandomResizedCropCoord(self.image_size, scale=(crop, 1.)))
        else:
            transform_list.append(transform_coord.ResizeCoord(self.image_size))
            transform_list.append(transform_coord.CenterCropCoord(self.image_size))
        if (p := view.get('horizontal_flip')) is not None and p > 0.:
            if isinstance(p, bool):
                p = 0.5
            transform_list.append(transform_coord.RandomHorizontalFlipCoord(p=p))
        if (p := view.get('jitter')) is not None and p > 0.:
            if isinstance(p, bool):
                p = 0.8
            transform_list.append(transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=p))
        if (p := view.get('grayscale')) is not None and p > 0.:
            if isinstance(p, bool):
                p = 0.2
            transform_list.append(transforms.RandomGrayscale(p=p))
        if (p := view.get('gaussian_blur')) is not None and p > 0.:
            if isinstance(p, bool):
                p = 0.5
            transform_list.append(transforms.RandomApply([GaussianBlur()], p=p))
        if view2 and (p := view.get('solarize')) is not None and p > 0.:
            if isinstance(p, bool):
                p = 0.2
            transform_list.append(RandomSolarizeFloat(threshold=0.5, p=p))
        transform_list.append(self.normalize)
        return transform_coord.Compose(transform_list)

    def get_transform(self):
        transform1 = self.view_transform(self.kwargs, False)
        transform2 = self.view_transform(self.kwargs, True)
        return transform1, transform2