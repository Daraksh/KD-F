import torch
import torch.nn as nn
from backbone import BackboneModel
from config import Config

class TeacherModel(nn.Module):
    def __init__(self, group_id, backbone, config=False):
        super().__init__()
        self.config = Config.teacher_config if not config else config
        self.backbone = backbone
        
        # FIRST: Check what attributes the backbone has
        if hasattr(backbone, 'feature_dim'):
            feature_dim = backbone.feature_dim
            num_classes = backbone.num_classes
        else:
            feature_dim = backbone.fc.in_features
            num_classes = 2  # or get from config
            
        # THEN: Create classifier using the determined dimensions
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        # Check if backbone is BackboneModel or raw ResNet
        if hasattr(self.backbone, 'get_features'):
            features = self.backbone.get_features(x)  # Get raw features (32, 512)
        else:
            features = self.backbone(x)  # For raw ResNet
        features = features.to(self.classifier.weight.device)
        output = self.classifier(features)
        return output

        
    def get_soft_predictions(self, x, temperature=1.5):
        logits = self.forward(x)
        scaled_logits = logits / temperature
        soft_predictions = torch.nn.functional.softmax(scaled_logits, dim=1)
        return soft_predictions
