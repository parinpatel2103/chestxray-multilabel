import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax", "Hernia"
]


def list_image_folders(img_root):
    """
    Returns list of folders that contain an 'images' subfolder, like:
    <img_root>/images_001/images, <img_root>/images_002/images, ...
    """
    if isinstance(img_root, (list, tuple)):
        return list(img_root)

    img_root = Path(img_root)
    img_dirs = []
    if not img_root.exists():
        return img_dirs

    for name in sorted(os.listdir(img_root)):
        p = img_root / name / "images"
        if p.is_dir():
            img_dirs.append(str(p))
    return img_dirs


def filter_df_to_existing_images(df, img_root):
    """
    Filters df to rows where Image Index exists in any of the image folders.
    img_root can be a string path or a list of image directories.
    """
    img_dirs = list_image_folders(img_root)

    existing = set()
    for d in img_dirs:
        try:
            for f in os.listdir(d):
                if f.endswith(".png"):
                    existing.add(f)
        except FileNotFoundError:
            pass

    return df[df["Image Index"].isin(existing)].reset_index(drop=True)


def _labels_to_multihot(label_str: str):
    y = np.zeros(len(CLASS_NAMES), dtype=np.float32)

    if pd.isna(label_str) or label_str.strip() == "" or label_str.strip() == "No Finding":
        return y

    parts = [p.strip().replace(" ", "_") for p in label_str.split("|")]
    for p in parts:
        if p in CLASS_NAMES:
            y[CLASS_NAMES.index(p)] = 1.0
    return y


def default_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class ChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_root: str, train=True, tfm=None):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.img_dirs = list_image_folders(img_root)
        self.tfm = tfm if tfm is not None else default_transforms(train=train)

        self.y = np.stack([_labels_to_multihot(x) for x in self.df["Finding Labels"].values])

    def __len__(self):
        return len(self.df)

    def _find_image(self, fname: str):
        for folder in self.img_dirs:
            path = os.path.join(folder, fname)
            if os.path.exists(path):
                return path
        return None

    def __getitem__(self, idx):
        fname = self.df.iloc[idx]["Image Index"]

        path = self._find_image(fname)
        if path is None:
            raise FileNotFoundError(f"Could not find image {fname} under {self.img_root}")

        img = Image.open(path).convert("RGB")
        x = self.tfm(img)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y
