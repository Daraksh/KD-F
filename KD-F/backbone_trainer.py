import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from fis_loss import FISLoss

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# class BackboneTrainer:
#     def _set_trainable_layers(self, model):
#         backbone_name = type(model.backbone).__name__
        
#         if 'EfficientNet' in backbone_name:
#             for name, param in model.backbone.named_parameters():
#                 if ('blocks.5' in name) or ('blocks.6' in name) or ('classifier' in name):
#                     param.requires_grad = True
#                 else:
#                     param.requires_grad = False
#         elif 'ViTModel' in backbone_name:
#             for name, param in model.backbone.named_parameters():
#                 if 'encoder.layer.10' in name or 'encoder.layer.11' in name:
#                     param.requires_grad = True
#                 else:
#                     param.requires_grad = False
        
#         for p in model.classifier.parameters():
#             p.requires_grad = True

class BackboneTrainer:
    def _set_trainable_layers(self, model):
        backbone_name = type(model.backbone).__name__

        if 'EfficientNet' in backbone_name:
            # Unfreeze last two blocks and classifier
            for name, param in model.backbone.named_parameters():
                if ('blocks.5' in name) or ('blocks.6' in name) or ('classifier' in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        elif 'ViTModel' in backbone_name:
            # Unfreeze last two transformer encoder layers
            for name, param in model.backbone.named_parameters():
                if 'encoder.layer.10' in name or 'encoder.layer.11' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        elif 'DinoVisionTransformer' in backbone_name or 'CLIPVisionTransformer' in backbone_name:
            # RAD-DINO: unfreeze last two transformer blocks
            for name, param in model.backbone.named_parameters():
                if 'encoder.layer.10' in name or 'encoder.layer.11' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # Always unfreeze classifier head
        for p in model.classifier.parameters():
            p.requires_grad = True


    # def train(self, model, dataloader, groups, config=False, device='cuda'):
    #     self.config = Config.backbone_trainer_config if config is False else config
    #     num_epochs = self.config['num_epochs']

    #     self._set_trainable_layers(model)

    #     self.optimizer = optim.AdamW(
    #         filter(lambda p: p.requires_grad, model.parameters()),
    #         lr=self.config['optimizer_lr'],
    #         weight_decay=self.config['optimizer_weight_decay']
    #     )

    #     fis_loss = FISLoss(warmup_epochs=max(1, num_epochs // 10))
    #     model.to(device)
    #     model.train()

    #     for epoch in range(num_epochs):
    #         fis_loss.set_epoch(epoch)
    #         epoch_loss = 0.0
    #         for batch in dataloader:
    #             data = batch['image'].to(device)
    #             labels = batch['label'].to(device)
    #             group_ids = batch['group'].to(device)

    #             self.optimizer.zero_grad()
    #             outputs = model(data)
    #             loss = fis_loss(outputs, labels, group_ids)
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #             self.optimizer.step()
    #             epoch_loss += loss.item()

    #         print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}')

    #     return model
    def train(self, model, dataloader, groups, config=False, device='cuda'):
        self.config = Config.backbone_trainer_config if config is False else config
        num_epochs = self.config['num_epochs']

        self._set_trainable_layers(model)

        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config['optimizer_lr'],
            weight_decay=self.config['optimizer_weight_decay']
        )
        fis_loss = FISLoss(warmup_epochs=max(1, num_epochs // 10))
        model.to(device)

        for epoch in range(num_epochs):
            model.train()
            fis_loss.set_epoch(epoch)
            epoch_loss = 0.0

            pbar = tqdm(dataloader, desc=f"[Backbone Epoch {epoch + 1}/{num_epochs}]")
            for batch in pbar:
                data = batch['image'].to(device)
                labels = batch['label'].to(device)
                group_ids = batch['group'].to(device)

                self.optimizer.zero_grad()
                outputs = model(data)
                loss = fis_loss(outputs, labels, group_ids)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                avg_loss = epoch_loss / (pbar.n + 1)
                pbar.set_postfix(loss=avg_loss)

            print(f"✅ Epoch {epoch + 1}/{num_epochs} completed. Avg Loss: {epoch_loss / len(dataloader):.4f}")

        return model