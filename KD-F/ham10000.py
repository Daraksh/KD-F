import os
from PIL import Image

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class HAM10000Dataset(Dataset):
    def __init__(
        self,
        data_path,
        split="train",
        sensitive_attr="sex",
        transform=None,
        max_benign=5000,        # new arg → keep ≤ 5 000 benign samples
        random_state=42,
    ):
        self.data_path = data_path
        self.split = split
        self.sensitive_attr = sensitive_attr
        self.transform = transform
        self.max_benign = max_benign
        self.random_state = random_state

        # Load metadata and create train / val / test indices
        self.metadata = self._load_metadata()
        self._prepare_split()

    # -----------------------------------------------------------------
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        row = self.metadata.iloc[real_idx]

        # image ----------------------------------------------------------------
        img_path = os.path.join(self.data_path, "images", f"{row['image_id']}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            default_transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                ]
            )
            image = default_transform(image)

        label = torch.tensor(row["binaryLabel"], dtype=torch.long)
        group = torch.tensor(row[self.sensitive_attr], dtype=torch.long)
        return {"image": image, "label": label, "group": group}

    # -----------------------------------------------------------------
    def _load_metadata(self):
        meta_path = os.path.join(self.data_path, "HAM10000_subset.csv")
        df = pd.read_csv(meta_path)

        # binary label ---------------------------------------------------------
        binary_mapping = {
            "mel": 1,
            "bcc": 1,
            "akiec": 1,
            "nv": 0,
            "bkl": 0,
            "vasc": 0,
            "df": 0,
        }
        df["binaryLabel"] = df["dx"].map(binary_mapping)

        # gender encoding ------------------------------------------------------
        if self.sensitive_attr == "sex":
            df["sex"] = df["sex"].map({"male": 0, "female": 1})
        elif self.sensitive_attr == "age":
            df["age"] = pd.cut(df["age"], bins=[0, 50, 70, 100], labels=[0, 1, 2])

        # ------------- down-sample benign to max_benign ------------------
        if self.max_benign is not None and self.max_benign > 0:
            benign_df = df[df["binaryLabel"] == 0]
            malignant_df = df[df["binaryLabel"] == 1]

            if len(benign_df) > self.max_benign:
                benign_df = benign_df.sample(
                    n=self.max_benign, random_state=self.random_state
                )

            df = pd.concat([benign_df, malignant_df]).reset_index(drop=True)

        return df

    # -----------------------------------------------------------------
    def _prepare_split(self):
        train_idx, test_idx = train_test_split(
            self.metadata.index,
            test_size=0.2,
            stratify=self.metadata[["binaryLabel", self.sensitive_attr]],
            random_state=self.random_state,
        )
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=0.1,
            stratify=self.metadata.loc[train_idx, ["binaryLabel", self.sensitive_attr]],
            random_state=self.random_state,
        )

        if self.split == "train":
            self.indices = train_idx
        elif self.split == "val":
            self.indices = val_idx
        else:
            self.indices = test_idx
