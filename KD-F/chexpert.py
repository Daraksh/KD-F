import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CheXpertDataset(Dataset):
    def __init__(self, dataframe, data_path, sensitive_attr='sex', transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.data_path = data_path
        self.sensitive_attr = sensitive_attr
        self.transform = transform
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_path, row['Path'])
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}")
            return {
                "image": torch.zeros(3, 224, 224),
                "label": torch.tensor(-1),
                "group": torch.tensor(-1)
            }
        
        image = self.transform(image)
        label = torch.tensor(row['No Finding'], dtype=torch.long)
        group = torch.tensor(row[self.sensitive_attr], dtype=torch.long)
        
        # return {'image': image, 'label': label, 'group': group,  'path': row['Path']}
        return {'image': image, 'label': label, 'group': group,  'sex': row['Sex']}
