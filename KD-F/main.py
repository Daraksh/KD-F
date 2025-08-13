import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os
import pickle
from run_full_pipeline import train_fairdi_complete

# Disable warnings and multiprocessing
import warnings
warnings.filterwarnings('ignore')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class HAM10000DatasetFromDF(Dataset):
    def __init__(self, dataframe, data_path, sensitive_attr='Sex_binary', transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.data_path = data_path
        self.sensitive_attr = sensitive_attr
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_path, row['Path'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(row['binaryLabel'], dtype=torch.long)
        group = torch.tensor(row[self.sensitive_attr], dtype=torch.long)
        
        return {'image': image, 'label': label, 'group': group}

def preprocess_ham10000(data_path):
    demo_data = pd.read_csv(os.path.join(data_path, 'HAM10000_subset.csv'))
    demo_data['Path'] = demo_data['image_id'].apply(lambda x: f'images/{x}.jpg')
    demo_data = demo_data.dropna(subset=['age', 'sex'])
    demo_data['Sex_binary'] = (demo_data['sex'] == 'male').astype(int)
    
    print('Sex distribution in original data:')
    print(demo_data['sex'].value_counts())
    print('Sex_binary distribution:')
    print(demo_data['Sex_binary'].value_counts())
    
    sex = demo_data['sex'].values
    sex[sex == 'male'] = 'M'
    sex[sex == 'female'] = 'F'
    demo_data['Sex'] = sex
    
    ages = demo_data['age'].astype(int)
    demo_data['Age_binary'] = np.where(ages < 60, 0, 1)
    
    labels = demo_data['dx'].copy()
    labels[labels != 'akiec'] = '0'
    labels[labels == 'akiec'] = '1'
    demo_data['binaryLabel'] = labels.astype(int)
    
    return demo_data

def split_data(demo_data):
    demo_data = demo_data.copy()
    demo_data['stratify_col'] = demo_data['Sex_binary'].astype(str) + '_' + demo_data['binaryLabel'].astype(str)
    
    train_meta, temp_meta = train_test_split(
        demo_data,
        test_size=0.2,
        random_state=0,
        stratify=demo_data['stratify_col']
    )
    
    val_meta, test_meta = train_test_split(
        temp_meta,
        test_size=0.5,
        random_state=0,
        stratify=temp_meta['stratify_col']
    )
    
    for df in (train_meta, val_meta, test_meta):
        df.drop(columns=['stratify_col'], inplace=True)
    
    print('Train split Sex_binary distribution:')
    print(train_meta['Sex_binary'].value_counts())
    print('Val split Sex_binary distribution:')
    print(val_meta['Sex_binary'].value_counts())
    print('Test split Sex_binary distribution:')
    print(test_meta['Sex_binary'].value_counts())
    
    return train_meta, val_meta, test_meta

def save_models_and_results(teachers, student, results, save_dir='saved_models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    print("Saving teacher models...")
    for gid, teacher in teachers.items():
        torch.save(teacher.state_dict(), os.path.join(save_dir, f'teacher_group_{gid}.pth'))
        print(f"Saved teacher for group {gid}")
    
    print("Saving student model...")
    torch.save(student.state_dict(), os.path.join(save_dir, 'student.pth'))
    
    print("Saving evaluation results...")
    with open(os.path.join(save_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"All models and results saved to: {save_dir}")
    return save_dir

def main():
    data_path = 'D:\\HAM10000\\'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using GPU" if device == 'cuda' else "Using CPU")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("Preprocessing HAM10000 metadata...")
    demo_data = preprocess_ham10000(data_path)
    print("Splitting data...")
    train_meta, val_meta, test_meta = split_data(demo_data)
    
    print("Creating datasets...")
    train_dataset = HAM10000DatasetFromDF(train_meta, data_path, transform=transform)
    val_dataset = HAM10000DatasetFromDF(val_meta, data_path, transform=transform)
    test_dataset = HAM10000DatasetFromDF(test_meta, data_path, transform=transform)
    
    print("Starting FairDI pipeline...")
    teachers, student, results = train_fairdi_complete(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        sensitive_attr='Sex_binary',
        num_classes=2,
        device=device,
        backbone='vit_base_patch16_224'
    )
    
    print("Training completed!")
    save_dir = save_models_and_results(teachers, student, results)
    print(f"Pipeline completed successfully! Models saved in: {save_dir}")
    
    return teachers, student, results

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    torch.set_num_threads(1)
    main()
