from pretraining.datasets.base_dataset import CounterfactualContrastiveDataset
from pathlib import Path
import pandas as pd
import torch
from datetime import datetime
from pretraining.datasets.caching import SharedCache
import numpy as np
from torchvision.io import decode_image
from pretraining.utils.common import rle_decode
import os
import time

def prepare_padchest_csv(df_path, seg_path=None, manual_path=None):
    df = pd.read_csv(
        df_path
    )
    def process(x, target):
        if isinstance(x, str):
            list_labels = x[1:-1].split(",")
            list_labels = [label.replace("'", "").strip() for label in list_labels]
            return target in list_labels
        else:
            return False

    for label in [
        "exclude",
        "suboptimal study",
    ]:
        df[label] = df.Labels.astype(str).apply(lambda x: process(x, label))

    df['pleural effusion'] = df.Labels.astype(str).apply(lambda x: process(x, 'pleural effusion')) | df.labelCUIS.astype(str).apply(lambda x: process(x, 'C2073625'))
    df['disease'] = (df.Labels == "['pleural effusion']") | (df.labelCUIS == "['C2073625']")
    df['no finding'] = (df.Labels == "[]") | (df.Labels == "['unchanged']") | (df.Labels == "['normal']") | (df.labelCUIS == "[]")

    df = df.loc[df.Projection == "PA"]
    df = df.loc[df.Pediatric == "No"]

    df = df.loc[~df.exclude]
    df = df.loc[~df["suboptimal study"]]
    df["Manufacturer"] = df.Manufacturer_DICOM.apply(
        lambda x: "Phillips" if x == "PhilipsMedicalSystems" else "Imaging"
    )
    df = df.loc[df["PatientSex_DICOM"].isin(["M", "F"])]
    df["PatientAge"] = (
        df.StudyDate_DICOM.apply(lambda x: datetime.strptime(str(x), "%Y%M%d").year)
        - df.PatientBirth
    )
    invalid_filenames = [
        "216840111366964013829543166512013353113303615_02-092-190.png",
        "216840111366964013962490064942014134093945580_01-178-104.png",
        "216840111366964012989926673512011151082430686_00-157-045.png",
        "216840111366964012558082906712009327122220177_00-102-064.png",
        "216840111366964012959786098432011033083840143_00-176-115.png",
        "216840111366964012373310883942009152114636712_00-102-045.png",
        "216840111366964012487858717522009280135853083_00-075-001.png",
        "216840111366964012819207061112010307142602253_04-014-084.png",
        "216840111366964012989926673512011074122523403_00-163-058.png",
        "216840111366964013590140476722013058110301622_02-056-111.png",
        "216840111366964012339356563862009072111404053_00-043-192.png",
        "216840111366964013590140476722013043111952381_02-065-198.png",
        "216840111366964012819207061112010281134410801_00-129-131.png",
        "216840111366964013686042548532013208193054515_02-026-007.png",
        "216840111366964012989926673512011083134050913_00-168-009.png"
        # '216840111366964013590140476722013058110301622_02-056-111.png'
    ]
    if manual_path is not None:
        manual_names = []
        for file in os.listdir(str(Path(manual_path) / 'images')):
            filename = os.fsdecode(file)
            manual_names.append(filename)
        invalid_filenames.extend(manual_names)
    
    df = df.loc[~df.ImageID.isin(invalid_filenames)]
    df.disease = df['pleural effusion']

    if seg_path is not None:
        seg_df = pd.read_csv(seg_path)
        merged_df = pd.merge(df, seg_df, on='ImageID')
        df = merged_df[merged_df['Dice RCA (Mean)'] > 0.7]

    return df.reset_index(drop=True)

class CounterfactualContrastivePadChestDataset(CounterfactualContrastiveDataset):
    def __init__(
        self,
        csv_path,
        label: str,
        transform = None,
        cache: bool = False,
        max_cache_size = 24,
        cache_dim = [1,224,224],
        cf_suffixes = ['sc', 'pe', 'sc_pe'],
        multi_crop=False, 
        return_coord=False,
        cf_dir = "/vol/biomedic3/bglocker/mscproj/mal224/data/mini-padchest/padchest_cf_images_v0",
        target_transform = None,
        seg_path = None,
        manual_path = None,
        parents=['scanner', 'sex', 'disease'],
    ):
        super().__init__(transform, cf_suffixes, multi_crop, return_coord, target_transform)
        self.csv_path = Path(csv_path)
        seg_path = Path(seg_path) if seg_path is not None else None
        df = prepare_padchest_csv(self.csv_path, seg_path, manual_path)
        self.data_dir = self.csv_path.parent / "images"
        self.cf_dir = cf_dir
        self.label_col = label
        self.patients = df.PatientID
        self.disease = df.disease.astype(int).values
        self.img_paths = df.ImageID.values
        self.genders = df.PatientSex_DICOM.values
        self.ages = df.PatientAge.values
        self.manufacturers = df.Manufacturer.values
        self.cache = cache
        self.left_lung = df["Left Lung"].values
        self.right_lung = df["Right Lung"].values
        self.widths = df["Width"].values
        self.heights = df["Height"].values
        self.heart = df.Heart.values
        self.parents = parents

        if cache:
            self.cache = SharedCache(
                size_limit_gib=max_cache_size,
                dataset_len=self.img_paths.shape[0],
                data_dims=cache_dim,
                dtype=torch.float32,
            )
        else:
            self.cache = None

    def __len__(self):
        return len(self.img_paths)

    def read_image(self, idx):
        path = self.data_dir / self.img_paths[idx]
        img = decode_image(str(path))  

        if img.shape[0] == 3:  
            img = img.float()
            img = ((img[0] + img[1] + img[2]) / 3).unsqueeze(0)
        else:
            img = img.float()
        return img/(img.max() + 1e-6)  

    def get_image(self, idx, return_image=True):
        start = time.time()
        if return_image:
            if self.cache is not None:
                image = self.cache.get_slot(idx)
                if image is None:
                    image = self.read_image(idx)
                    self.cache.set_slot(idx, image, allow_overwrite=True)
            else:
                image = self.read_image(idx)
        time_ckpt = time.time()
        get_image_time = time_ckpt - start
        sample = {}
        sample["index"] = idx
        sample["age"] = self.ages[idx] / 100
        sample["sex"] = 0 if self.genders[idx] == "M" else 1
        sample["scanner"] = 0 if self.manufacturers[idx] == "Phillips" else 1
        sample["disease"] = 0 if not self.disease[idx] else 1
        sample["y"] = sample[self.label_col]
        sample["shortpath"] = self.img_paths[idx]
        sample["patient"] = self.patients[idx]
        start = time.time()
        get_other_time = start-time_ckpt
        if return_image:
            sample["mask"] = rle_decode((self.widths[idx], self.heights[idx]), [self.left_lung[idx], self.right_lung[idx]])
            sample["x"] = image
        time_ckpt = time.time()
        get_mask_time = time_ckpt - start
        sample['get_image_time'] = get_image_time
        sample['get_other_time'] = get_other_time
        sample['get_mask_time'] = get_mask_time
        if self.parents is not None:
            sample["pa"] = torch.cat(
                [
                    sample[c]
                    if isinstance(sample[c], torch.Tensor)
                    else torch.tensor([sample[c]])
                    for c in self.parents
                ]
            ).detach()
        
        return sample

    def get_counterfactual_image(self, idx, cf_suffix):
        cf_dir = Path(self.cf_dir)
        short_path = self.img_paths[idx][:-4]
        filename = cf_dir / f"{short_path}_{cf_suffix}.png"
        if os.path.exists(str(filename)):
            img = decode_image(str(filename)) / 255.0
            if img.shape[0] == 3:
                img = ((img[0] + img[1] + img[2]) / 3).unsqueeze(0)
            img = img / (img.max() + 1e-12)
            return img
        else:
            raise FileNotFoundError(f'Counterfactual not found {str(filename)}')
    
    def _split(self, train_prop, idxs=None, balance_disease=False, **kwargs):
        if idxs is None:
            idxs = np.arange(len(self))
        
        df = pd.DataFrame({
            'PatientID': self.patients[idxs],
            'disease': self.disease[idxs]
        })

        all_patients = df['PatientID'].unique()
        n_test = int(len(all_patients) * (1-train_prop))

        if not balance_disease:
            test_patients_all = np.random.choice(all_patients, size=n_test, replace=False)
        else:
            selected_df = df[df['disease'] == 1]
            unselected_df = df[df['disease'] == 0]

            n_half = n_test // 2

            selected_sample = selected_df['PatientID'].drop_duplicates()
            unselected_sample = unselected_df['PatientID'].drop_duplicates()

            selected_sample = selected_sample.sample(min(n_half, len(selected_sample)), random_state=42)
            unselected_sample = unselected_sample.sample(min(n_half, len(unselected_sample)), random_state=42)

            test_patients_all = pd.unique(pd.concat([selected_sample, unselected_sample]))

        complement_patients = np.setdiff1d(all_patients, test_patients_all)

        complement_idx = self.patients[self.patients.isin(complement_patients)].index.to_numpy()
        test_idx = self.patients[self.patients.isin(test_patients_all)].index.to_numpy()

        return complement_idx, test_idx

class CounterfactualContrastivePreProcessedPadChestDataset(CounterfactualContrastiveDataset):
    def __init__(
        self,
        csv_path,
        preprocess_root,
        label: str,
        transform = None,
        cf_suffixes = ['sc', 'pe', 'sc_pe'],
        multi_crop=False, 
        return_coord=False,
        target_transform = None,
        seg_path = None,
        parents=['scanner', 'sex', 'disease'],
        manual_path=None,
    ):
        super().__init__(transform, cf_suffixes, multi_crop, return_coord, target_transform)
        self.csv_path = Path(csv_path)
        seg_path = Path(seg_path) if seg_path is not None else None
        df = prepare_padchest_csv(self.csv_path, seg_path, manual_path)
        self.preprocess_root = Path(preprocess_root)
        self.label_col = label
        self.patients = df.PatientID
        self.disease = df.disease.astype(int).values
        self.img_paths = df.ImageID.values
        self.genders = df.PatientSex_DICOM.values
        self.ages = df.PatientAge.values
        self.manufacturers = df.Manufacturer.values
        self.left_lung = df["Left Lung"].values
        self.right_lung = df["Right Lung"].values
        self.widths = df["Width"].values
        self.heights = df["Height"].values
        self.heart = df.Heart.values
        self.parents = parents

    def __len__(self):
        return len(self.img_paths)
    
    def read_image(self, idx):
        filename = f"{self.img_paths[idx][:-4]}.npy"
        img = np.load(self.preprocess_root / 'images' / filename)
        img = torch.from_numpy(img)
        
        if img.shape[0] == 3:  
            img = img.float()
            img = ((img[0] + img[1] + img[2]) / 3).unsqueeze(0)
        else:
            img = img.float()
        return img/(img.max() + 1e-6)  

    def read_mask(self, idx):
        filename = f"{self.img_paths[idx][:-4]}.pt"
        mask = torch.load(self.preprocess_root / 'labels' / filename, weights_only=True)
        return mask.long()
    
    def get_image(self, idx, return_image=True):
        start = time.time()
        if return_image:
            image = self.read_image(idx)
        time_ckpt = time.time()
        get_image_time = time_ckpt - start
        sample = {}
        sample["index"] = idx
        sample["age"] = self.ages[idx] / 100
        sample["sex"] = 0 if self.genders[idx] == "M" else 1
        sample["scanner"] = 0 if self.manufacturers[idx] == "Phillips" else 1
        sample["disease"] = 0 if not self.disease[idx] else 1
        sample["y"] = sample[self.label_col]
        sample["shortpath"] = self.img_paths[idx]
        start = time.time()
        get_other_time = start-time_ckpt
        if return_image:
            sample["mask"] = self.read_mask(idx)
            sample["x"] = image
        time_ckpt = time.time()
        get_mask_time = time_ckpt - start
        sample['get_image_time'] = get_image_time
        sample['get_other_time'] = get_other_time
        sample['get_mask_time'] = get_mask_time
        if self.parents is not None:
            sample["pa"] = torch.cat(
                [
                    sample[c]
                    if isinstance(sample[c], torch.Tensor)
                    else torch.tensor([sample[c]])
                    for c in self.parents
                ]
            ).detach()
        
        return sample

    def get_counterfactual_image(self, idx, cf_suffix):
        filename = f"{self.img_paths[idx][:-4]}_{cf_suffix}.npy"
        img = np.load(self.preprocess_root / 'counterfactuals' / filename)
        img = torch.from_numpy(img)
        
        if img.shape[0] == 3:  
            img = img.float()
            img = ((img[0] + img[1] + img[2]) / 3).unsqueeze(0)
        else:
            img = img.float()
        return img/(img.max() + 1e-6)  
    
    def _split(self, train_prop, idxs=None, balance_disease=False, **kwargs):
        if idxs is None:
            idxs = np.arange(len(self))
        
        df = pd.DataFrame({
            'PatientID': self.patients[idxs],
            'disease': self.disease[idxs]
        })

        all_patients = df['PatientID'].unique()
        n_test = int(len(all_patients) * (1-train_prop))

        if not balance_disease:
            test_patients_all = np.random.choice(all_patients, size=n_test, replace=False)
        else:
            selected_df = df[df['disease'] == 1]
            unselected_df = df[df['disease'] == 0]

            n_half = n_test // 2

            selected_sample = selected_df['PatientID'].drop_duplicates()
            unselected_sample = unselected_df['PatientID'].drop_duplicates()

            selected_sample = selected_sample.sample(min(n_half, len(selected_sample)), random_state=42)
            unselected_sample = unselected_sample.sample(min(n_half, len(unselected_sample)), random_state=42)

            test_patients_all = pd.unique(pd.concat([selected_sample, unselected_sample]))

        complement_patients = np.setdiff1d(all_patients, test_patients_all)

        complement_idx = self.patients[self.patients.isin(complement_patients)].index.to_numpy()
        test_idx = self.patients[self.patients.isin(test_patients_all)].index.to_numpy()

        return complement_idx, test_idx