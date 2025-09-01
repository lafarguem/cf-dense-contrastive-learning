import os
import torchvision.transforms as TF
from torchvision.io import decode_image
from torchvision.utils import save_image
import torch
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
from pretraining.datasets.padchest import prepare_padchest_csv
from pretraining.utils.common import rle_decode
import psutil
import numpy as np

PADCHEST_ROOT = Path("/vol/biodata/data/chest_xray/BIMCV-PADCHEST")
PADCHEST_IMAGES = PADCHEST_ROOT / "images"

COUNTERFACTUAL_IMAGES = Path("/vol/biomedic3/bglocker/mscproj/mal224/data/counterfactuals")
CF_SUFFIXES = ['sc', 'pe', 'sc_pe']

DATA_SAVE = Path('/vol/biomedic3/bglocker/mscproj/mal224/data/manual_padchest/preprocessed/base')
CF_SAVE = Path('/vol/biomedic3/bglocker/mscproj/mal224/data/manual_padchest/preprocessed/counterfactuals')

MANUAL_ROOT = Path('/vol/biomedic3/bglocker/mscproj/mal224/data/manual_padchest')

print("reading csv")
df = prepare_padchest_csv(
    PADCHEST_ROOT / "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
)

images = []
names = []
masks = []
counterfactuals = {key: [] for key in CF_SUFFIXES}

directory = os.fsencode(str(MANUAL_ROOT / 'images'))

for idx,f in enumerate(os.listdir(directory)):
    filename = os.fsdecode(f)
    image = decode_image(MANUAL_ROOT / 'images' / filename)[0].unsqueeze(0)
    mask = decode_image(MANUAL_ROOT / 'labels' / filename)
    out = torch.zeros(224, 224)
    out[mask[2] > 0] = 1
    out[mask[1] > 0] = 2
    mask = out.unsqueeze(0).to(torch.uint8)
    for suffix in CF_SUFFIXES:
        cf_filename = COUNTERFACTUAL_IMAGES / f"{filename[:-4]}_{suffix}.png"
        cf_image = decode_image(cf_filename)[0].unsqueeze(0)
        counterfactuals[suffix].append(cf_image.numpy())

    if image.shape[0] != 1:
        print(filename)
        continue
    images.append(image.numpy())
    masks.append(mask)
    names.append(filename)

    if (idx+1) % 5000 == 0:
        process = psutil.Process() 
        mem_info = process.memory_info()

        print(f"RSS (Resident Set Size): {mem_info.rss / (1024 * 1024):.2f} MB")
        print("saving", idx)
        image_tensor = np.stack(images, axis=0)
        mask_tensor = torch.stack(masks, dim=0).to(torch.uint8)
        np.save(str(DATA_SAVE / 'images.npy'), image_tensor)
        torch.save(mask_tensor, str(DATA_SAVE / 'masks.pt'))
        np.save(str(DATA_SAVE / 'names.npy'), names)
        for key in counterfactuals:
            cf_tensor = np.stack(counterfactuals[key], axis=0)
            np.save(str(CF_SAVE / f'{key}.npy'), cf_tensor)

image_tensor = np.stack(images, axis=0)
mask_tensor = torch.stack(masks, dim=0).to(torch.uint8)
np.save(str(DATA_SAVE / 'images.npy'), image_tensor)
torch.save(mask_tensor, str(DATA_SAVE / 'masks.pt'))
np.save(str(DATA_SAVE / 'names.npy'), names)
for key in counterfactuals:
    cf_tensor = np.stack(counterfactuals[key], axis=0)
    np.save(str(CF_SAVE / f'{key}.npy'), cf_tensor)