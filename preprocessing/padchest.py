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
from pretraining.utils.debug import save_mask
import psutil
import time
import numpy as np

PADCHEST_ROOT = Path("/vol/biodata/data/chest_xray/BIMCV-PADCHEST")
PADCHEST_IMAGES = PADCHEST_ROOT / "images"

SEG_CSV = '/vol/biodata/data/chest_xray/CheXmask/v0.4/OriginalResolution/Padchest.csv'

COUNTERFACTUAL_IMAGES = Path("/vol/biomedic3/bglocker/mscproj/mal224/data/counterfactuals")
CF_SUFFIXES = ['sc', 'pe', 'sc_pe']

DATA_SAVE = Path('/vol/biomedic3/bglocker/mscproj/mal224/data/padchest')
CF_SAVE = Path('/vol/biomedic3/bglocker/mscproj/mal224/data/padchest/counterfactuals')

print("reading csv")
df = prepare_padchest_csv(
    PADCHEST_ROOT / "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
    SEG_CSV
)

counterfactuals = {key: [] for key in CF_SUFFIXES}
transform = TF.Compose([TF.Resize(224), TF.CenterCrop(224)])
mask_transform = TF.Compose(
    [TF.Resize(224, interpolation=TF.InterpolationMode.NEAREST), TF.CenterCrop(224)]
)
left_lung = df["Left Lung"].values
right_lung = df["Right Lung"].values
widths = df["Width"].values
heights = df["Height"].values

for idx in tqdm(range(len(df))):
    filename = df.iloc[idx]['ImageID']
    image = decode_image(PADCHEST_IMAGES / filename)
    if image.shape[0] != 1:
        print(filename)
        continue
    mask = rle_decode((widths[idx], heights[idx]), [left_lung[idx], right_lung[idx]])
    image = transform(image)
    mask = mask_transform(mask).to(torch.uint8)
    np.save(DATA_SAVE / 'images' / f'{filename[:-4]}.npy', image.numpy())
    torch.save(mask, DATA_SAVE / 'labels' / f'{filename[:-4]}.pt')
    for suffix in CF_SUFFIXES:
        cf_filename = COUNTERFACTUAL_IMAGES / f"{filename[:-4]}_{suffix}.png"
        cf_image = decode_image(cf_filename)[0].unsqueeze(0)
        np.save(CF_SAVE / f'{filename[:-4]}_{suffix}.npy', cf_image.numpy())