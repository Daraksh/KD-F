import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

from transformers import AutoModel

from transformers import AutoImageProcessor

# class BackboneModel(nn.Module):
#     def __init__(self, num_classes: int = 2, backbone: str = 'efficientnet_b3'):
#         super().__init__()
        
#         if backbone == 'efficientnet_b3':
#             import timm
#             self.backbone = timm.create_model('efficientnet_b3', pretrained=True)
#             self.feature_dim = self.backbone.classifier.in_features
#             self.backbone.classifier = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
#         elif backbone == 'vit_base_patch16_224':
#             self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224', add_pooling_layer=False)
#             self.feature_dim = self.backbone.config.hidden_size
#         else:
#             raise ValueError(f'Unknown backbone: {backbone}')
            
        
#         self.num_classes = num_classes
#         self.classifier = nn.Linear(self.feature_dim, num_classes)

#     def forward(self, x):
#         if hasattr(self.backbone, 'classifier'):
#             features = self.backbone(x)
#         else:
#             outputs = self.backbone(pixel_values=x)
#             features = outputs.last_hidden_state[:, 0]
#         return self.classifier(features)

#     def get_features(self, x):
#         if hasattr(self.backbone, 'classifier'):
#             return self.backbone(x)
#         else:
#             outputs = self.backbone(pixel_values=x)
#             return outputs.last_hidden_state[:, 0]

#     def freeze_backbone(self):
#         for param in self.backbone.parameters():
#             param.requires_grad = False

from transformers import AutoModel, AutoImageProcessor
import torch.nn as nn

class BackboneModel(nn.Module):
    def __init__(self, num_classes: int = 2, backbone: str = 'efficientnet_b3'):
        super().__init__()

        if backbone == 'efficientnet_b3':
            import timm
            self.backbone = timm.create_model('efficientnet_b3', pretrained=True)
            self.feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'vit_base_patch16_224':
            from transformers import ViTModel
            self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224', add_pooling_layer=False)
            self.feature_dim = self.backbone.config.hidden_size
        elif backbone == 'rad_dino':
            self.backbone = AutoModel.from_pretrained("microsoft/rad-dino")
            self.feature_dim = self.backbone.config.hidden_size
        else:
            raise ValueError(f'Unknown backbone: {backbone}')

        self.num_classes = num_classes
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        if hasattr(self.backbone, 'classifier'):
            features = self.backbone(x)
        else:
            # for ViT & RAD-DINO
            outputs = self.backbone(pixel_values=x)
            features = outputs.pooler_output  # CLS embedding
        return self.classifier(features)

    def get_features(self, x):
        if hasattr(self.backbone, 'classifier'):
            return self.backbone(x)
        else:
            outputs = self.backbone(pixel_values=x)
            return outputs.pooler_output

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
