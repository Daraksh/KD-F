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
from chexpert import CheXpertDataset


import warnings
warnings.filterwarnings('ignore')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# def preprocess_chexpert(data_path):
#     train_df = pd.read_csv(os.path.join(data_path, 'new_train.csv'))
#     valid_df = pd.read_csv(os.path.join(data_path, 'new_val.csv'))
#     test_df = pd.read_csv(os.path.join(data_path,'new_val.csv'))
#     demo_data = pd.concat([train_df, valid_df], ignore_index=True)
    
#     demo_data['Path'] = demo_data['Path']
#     demo_data['binaryLabel'] = (demo_data['No Finding'] == 0).astype(int)
#     #demo_data['FitzCategory'] = demo_data['FitzCategory'].map({'Male': 0, 'Female': 1})
    
#     demo_data = demo_data[demo_data['Frontal/Lateral'] == 'Frontal']
#     demo_data = demo_data.dropna(subset=['FitzCategory'])
    
#     print('FitzCategory distribution in data:')
#     print(demo_data['FitzCategory'].value_counts())
#     print('FitzCategory binary distribution:')
#     print(demo_data['FitzCategory'].value_counts())
#     print('Label distribution:')
#     print(demo_data['binaryLabel'].value_counts())
#     print('Total samples:', len(demo_data))
    
#     return demo_data

# def split_data(demo_data):
#     demo_data = demo_data.copy()
#     demo_data['stratify_col'] = demo_data['FitzCategory'].astype(str) + '_' + demo_data['binaryLabel'].astype(str)
    
#     train_meta, temp_meta = train_test_split(
#         demo_data,
#         test_size=0.2,
#         random_state=0,
#         stratify=demo_data['stratify_col']
#     )
    
#     val_meta, test_meta = train_test_split(
#         temp_meta,
#         test_size=0.5,
#         random_state=0,
#         stratify=temp_meta['stratify_col']
#     )
    
#     for df in (train_meta, val_meta, test_meta):
#         df.drop(columns=['stratify_col'], inplace=True)
    
#     print('Train split FitzCategory distribution:')
#     print(train_meta['FitzCategory'].value_counts())
#     print('Val split FitzCategory distribution:')
#     print(val_meta['FitzCategory'].value_counts())
#     print('Test split FitzCategory distribution:')
#     print(test_meta['FitzCategory'].value_counts())
    
#     return train_meta, val_meta, test_meta
# def split_data(demo_data, save_dir=None):
#     demo_data = demo_data.copy()
#     demo_data['stratify_col'] = demo_data['FitzCategory'].astype(str) + '_' + demo_data['binaryLabel'].astype(str)
    
#     train_meta, temp_meta = train_test_split(
#         demo_data,
#         test_size=0.2,
#         random_state=0,
#         stratify=demo_data['stratify_col']
#     )
    
#     val_meta, test_meta = train_test_split(
#         temp_meta,
#         test_size=0.5,
#         random_state=0,
#         stratify=temp_meta['stratify_col']
#     )
    
#     for df in (train_meta, val_meta, test_meta):
#         df.drop(columns=['stratify_col'], inplace=True)

#     # Optional: Save to CSV
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         train_meta.to_csv(os.path.join(save_dir, 'train_split.csv'), index=False)
#         val_meta.to_csv(os.path.join(save_dir, 'val_split.csv'), index=False)
#         test_meta.to_csv(os.path.join(save_dir, 'test_split.csv'), index=False)
#         print(f"Saved train/val/test splits to: {save_dir}")

#     print('Train split FitzCategory distribution:')
#     print(train_meta['FitzCategory'].value_counts())
#     print('Val split FitzCategory distribution:')
#     print(val_meta['FitzCategory'].value_counts())
#     print('Test split FitzCategory distribution:')
#     print(test_meta['FitzCategory'].value_counts())
    
#     return train_meta, val_meta, test_meta


def save_models_and_results(teachers, student, results, save_dir='saved_models_chexpert'):
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
    data_path = '/DATA2/fairune'
    train_meta = pd.read_csv(os.path.join(data_path, 'new_train.csv'))
    val_meta = pd.read_csv(os.path.join(data_path, 'new_val.csv'))
    test_meta = pd.read_csv(os.path.join(data_path,'new_val.csv'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using GPU" if device == 'cuda' else "Using CPU")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    #print("Preprocessing CheXpert metadata...")
    #demo_data = preprocess_chexpert(data_path)
    #print("Splitting data...")
    # train_meta, val_meta, test_meta = split_data(demo_data)
    #train_meta, val_meta, test_meta = split_data(demo_data, save_dir=os.path.join(data_path, 'splits'))

    
    print("Creating datasets...")
    train_dataset = CheXpertDataset(train_meta, data_path, sensitive_attr='FitzCategory', transform=transform)
    val_dataset = CheXpertDataset(val_meta, data_path, sensitive_attr='FitzCategory', transform=transform)
    test_dataset = CheXpertDataset(test_meta, data_path, sensitive_attr='FitzCategory', transform=transform)
    
    print("Starting FairDI pipeline...")
    teachers, student, results = train_fairdi_complete(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        sensitive_attr='FitzCategory',
        num_classes=2,
        device=device,
        #backbone='vit_base_patch16_224'
        backbone='rad_dino'
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
