import torch
from teacher import TeacherModel
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


# class TeacherTrainer:
#     def train_teachers(self, backbone,grouped_datasets,original_dataset,num_epochs = 50,device = 'cuda'):
        
#         # Freeze backbone for teacher training
#         backbone.freeze_backbone()
        
#         teachers = {}
#         for group_id, group_indices in grouped_datasets.items():
#             print(f"Training teacher for group {group_id}....")
#             teacher = TeacherModel(group_id,backbone)
            
#             for param in teacher.backbone.parameters():
#                 param.requires_grad = False
            
#             group_subset = torch.utils.data.Subset(original_dataset,group_indices)
#             group_loader = DataLoader(group_subset,batch_size = 64, shuffle = True)
            
#             optimizer = optim.SGD(teacher.classifier.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
#             criterion = nn.CrossEntropyLoss()
            
#             teacher.classifier.train()
            
#             for epoch in range(num_epochs):
#                 epoch_loss = 0
#                 for batch in group_loader:
#                     data = batch['image'].to(device)
#                     targets = batch['label'].to(device)
#                     optimizer.zero_grad()
#                     outputs = teacher(data)
#                     targets = targets.to(outputs.device)
#                     loss = criterion(outputs,targets)
#                     loss.backward()
                    
#                     # Gradient clipping
#                     torch.nn.utils.clip_grad_norm_(teacher.classifier.parameters(), max_norm=1.0)
                    
#                     optimizer.step()
#                     epoch_loss+=loss.item()
                
#                 if epoch%10 == 0:
#                     print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(group_loader):.4f}')
            
#             teachers[group_id]=teacher
        
#         return teachers
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

class TeacherTrainer:
    def train_teachers(self, backbone, grouped_datasets, original_dataset, num_epochs=50, device='cuda'):
        
        # Freeze backbone for teacher training
        backbone.freeze_backbone()
        
        teachers = {}
        for group_id, group_indices in grouped_datasets.items():
            print(f"\n🧑‍🏫 Training teacher for group {group_id}...")

            teacher = TeacherModel(group_id, backbone)
            
            # Freeze backbone parameters
            for param in teacher.backbone.parameters():
                param.requires_grad = False
            
            group_subset = torch.utils.data.Subset(original_dataset, group_indices)
            group_loader = DataLoader(group_subset, batch_size=150, shuffle=True)

            optimizer = optim.SGD(teacher.classifier.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            teacher.to(device)

            for epoch in range(num_epochs):
                teacher.train()
                epoch_loss = 0.0

                pbar = tqdm(group_loader, desc=f"[Group {group_id} | Epoch {epoch+1}/{num_epochs}]")
                for batch in pbar:
                    data = batch['image'].to(device)
                    targets = batch['label'].to(device)

                    optimizer.zero_grad()
                    outputs = teacher(data)
                    loss = criterion(outputs, targets)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(teacher.classifier.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    avg_loss = epoch_loss / (pbar.n + 1)
                    pbar.set_postfix(loss=avg_loss)

                print(f"✅ [Group {group_id}] Epoch {epoch + 1}/{num_epochs} completed. Avg Loss: {epoch_loss / len(group_loader):.4f}")

                # 🔍 Train AUC calculation
                teacher.eval()
                all_probs, all_labels = [], []

                with torch.no_grad():
                    for batch in group_loader:
                        x = batch['image'].to(device)
                        y = batch['label'].to(device)
                        logits = teacher(x)
                        probs = F.softmax(logits, dim=1)[:, 1]
                        all_probs.extend(probs.cpu().numpy())
                        all_labels.extend(y.cpu().numpy())

                try:
                    auc = roc_auc_score(all_labels, all_probs)
                    print(f"📊 [Group {group_id}] Train AUC after Epoch {epoch + 1}: {auc:.4f}")
                except ValueError:
                    print(f"⚠️ [Group {group_id}] Skipping AUC (Epoch {epoch + 1}) due to single-class labels")

            teachers[group_id] = teacher

        return teachers
