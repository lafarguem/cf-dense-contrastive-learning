from pretraining.augmentations.base_transform import BaseTransform
import torchvision.transforms as transforms
import pretraining.augmentations.utils.transform_coord as transform_coord

class ValTransform(BaseTransform):
    def __init__(self, image_size, normalize):
        self.normalize = transforms.Normalize(**normalize)
        self.image_size = image_size

    def get_transform(self):
        transform = transform_coord.Compose([
            transform_coord.ResizeCoord(self.image_size + 32),
            transform_coord.CenterCropCoord(self.image_size),
            self.normalize,
        ])
        return transform, transform