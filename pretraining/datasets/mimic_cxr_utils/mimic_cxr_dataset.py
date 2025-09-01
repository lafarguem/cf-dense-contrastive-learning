"""
This module contains the MIMIC_Dataset class, which is used to load the dataset.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd
from PIL import Image

from pretraining.datasets.mimic_cxr_utils.mimic_cxr_modifiers import Modifier, Pathology
from torchvision.io import decode_image


class MIMICDataset:
    """
    Dataset class for MIMIC-CXR-JPG dataset.
    Each datum is a tuple of an image and a pandas Series containing the labels.
    """

    def __init__(
        self,
        root: str,
        split_path: str,
        modifiers: Optional[List[Modifier]] = None,
        meta_csv = None,
        seg_csv = None,
    ):
        self.root = Path(root)
        self.split_path = Path(split_path)
        self.meta_csv = meta_csv
        self.seg_csv = seg_csv
        labels = self.get_labels()

        labels = labels.set_index("dicom_id")

        if modifiers:
            for modifier in modifiers:
                labels = modifier.apply(labels)

        labels["Path"] = labels.apply(self.get_path, axis=1)

        self.labels = labels

    def get_labels(self) -> pd.DataFrame:
        """
        Retrieves the labels from the metadata and chexpert csv files and merges them.
        """
        metadata_labels = pd.read_csv(self.root / "mimic-cxr-2.0.0-metadata.csv")
        chexpert_labels = pd.read_csv(
            self.root / "mimic-cxr-2.0.0-chexpert.csv",
            index_col=["subject_id", "study_id"],
        )
        splits = pd.read_csv(self.split_path)
        meta = pd.read_csv(self.meta_csv)
        seg = pd.read_csv(self.seg_csv)
        labels = metadata_labels.merge(
            chexpert_labels,
            on="study_id",
            how="left",
        ).dropna(subset=["subject_id"])
        labels = labels.merge(
            splits,
            on="dicom_id",
            suffixes=("", "_right"),
            how="left",
        ).merge(meta).merge(seg, on='dicom_id')
        return labels

    def get_path(self, row: pd.Series):
        """
        Returns the path of the image file corresponding to the row.
        """
        dicom_id = str(row.name)
        subject = "p" + str(int(row["subject_id"]))
        study = "s" + str(int(row["study_id"]))
        image_file = dicom_id + ".jpg"
        return self.root / "files" / subject[:3] / subject / study / image_file

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        path = row["Path"]
        img = decode_image(path)
        if img.shape[0] == 3:
            img = img.float()
            img = (0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]).unsqueeze(0)
        else:
            img = img.float()
        img = img/(img.max() + 1e-6)
    
        return (img, row)
    
    def get_row(self,idx):
        return self.labels.iloc[idx]