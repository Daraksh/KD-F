import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import Config

from sklearn.metrics import roc_auc_score
from tqdm import tqdm
class Student(nn.Module):
    def __init__(self, backbone, config=False):
        super().__init__()
        self.config = Config.student_config if not config else config
        self.backbone = backbone

        feature_dim = backbone.feature_dim if hasattr(backbone, 'feature_dim') else backbone.fc.in_features
        num_classes = backbone.num_classes if hasattr(backbone, 'num_classes') else 2
        self.classifier = nn.Linear(feature_dim, num_classes)

        self.tau = self.config['tau']
        self.lambda_kd = self.config['lambda_kd']
        self.num_epochs = self.config['num_epochs']

    # ------------------------------------------------------------------
    def forward(self, x):
        features = self.backbone.get_features(x) if hasattr(self.backbone, 'get_features') else self.backbone(x)
        return self.classifier(features)

    # ------------------------------------------------------------------
    # def train_student(self, teachers, full_dataloader, device='cuda'):
    #     self.backbone.freeze_backbone()
    #     for p in self.backbone.parameters():
    #         p.requires_grad = False

    #     self.optimizer = optim.Adam(self.classifier.parameters(), lr=1e-4)

    #     #  KLDivLoss with log_target=True
    #     kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
    #     ce_loss = nn.CrossEntropyLoss()

    #     for t in teachers.values():
    #         t.eval()

    #     self.classifier.train()
    #     for epoch in range(self.num_epochs):
    #         epoch_loss = 0.0
    #         for batch in full_dataloader:
    #             data = batch['image'].to(device)
    #             targets = batch['label'].to(device)
    #             group_ids = batch['group'].to(device)

    #             self.optimizer.zero_grad()
    #             student_logits = self.forward(data)
    #             student_log_probs = F.log_softmax(student_logits / self.tau, dim=1)

    #             # build teacher log-probabilities
    #             teacher_log_probs = []
    #             for i, gid in enumerate(group_ids):
    #                 #teacher = teachers[gid.item()]
    #                 teacher = teachers[gid.item()].to(device)
    #                 with torch.no_grad():
    #                     t_logits = teacher(data[i:i + 1])
    #                 teacher_log_probs.append(F.log_softmax(t_logits / self.tau, dim=1))
    #             teacher_log_probs = torch.cat(teacher_log_probs, dim=0)
    #             teacher_log_probs = teacher_log_probs.to(student_log_probs.device)
    #             distill = kl_loss(student_log_probs, teacher_log_probs)
    #             supervise = ce_loss(student_logits, targets)
    #             total = self.lambda_kd * (self.tau ** 2) * distill + (1 - self.lambda_kd) * supervise

    #             total.backward()
    #             torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
    #             self.optimizer.step()

    #             epoch_loss += total.item()

    #         if (epoch + 1) % 20 == 0:
    #             print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(full_dataloader):.4f}')

    #     return self
    def train_student(self, teachers, full_dataloader, device='cuda'):
        self.backbone.freeze_backbone()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.classifier.parameters(), lr=1e-4)
        kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
        ce_loss = nn.CrossEntropyLoss()

        for t in teachers.values():
            t.eval()

        self.classifier.train()
        log_every = 1  # set to 1 for every epoch, or change to 10/20 etc.

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            all_probs = []
            all_labels = []

            pbar = tqdm(full_dataloader, desc=f"[Student | Epoch {epoch+1}/{self.num_epochs}]")
            for batch in pbar:
                data = batch['image'].to(device)
                targets = batch['label'].to(device)
                group_ids = batch['group'].to(device)

                self.optimizer.zero_grad()
                student_logits = self.forward(data)
                student_log_probs = F.log_softmax(student_logits / self.tau, dim=1)

                teacher_log_probs = []
                for i, gid in enumerate(group_ids):
                    teacher = teachers[gid.item()].to(device)
                    with torch.no_grad():
                        t_logits = teacher(data[i:i + 1])
                    teacher_log_probs.append(F.log_softmax(t_logits / self.tau, dim=1))
                teacher_log_probs = torch.cat(teacher_log_probs, dim=0).to(student_log_probs.device)

                distill = kl_loss(student_log_probs, teacher_log_probs)
                supervise = ce_loss(student_logits, targets)
                total = self.lambda_kd * (self.tau ** 2) * distill + (1 - self.lambda_kd) * supervise

                total.backward()
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += total.item()
                avg_loss = epoch_loss / (pbar.n + 1)
                pbar.set_postfix(loss=avg_loss)

                # collect for AUC
                probs = F.softmax(student_logits, dim=1)[:, 1]
                all_probs.extend(probs.detach().cpu().numpy())
                all_labels.extend(targets.detach().cpu().numpy())

            print(f"✅ Epoch {epoch + 1}, Loss: {epoch_loss / len(full_dataloader):.4f}")

            try:
                auc = roc_auc_score(all_labels, all_probs)
                print(f"📊 Student Train AUC after Epoch {epoch + 1}: {auc:.4f}")
            except ValueError:
                print(f"⚠️ Skipping AUC (Epoch {epoch + 1}) due to single-class labels")

        return self