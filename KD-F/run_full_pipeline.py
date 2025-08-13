import torch
from torch.utils.data import DataLoader
from dataset_factory import prepare_grouped_datasets, GroupBalancedSampler
from backbone import BackboneModel
from backbone_trainer import BackboneTrainer
from teacher_trainer import TeacherTrainer
from student_trainer import Student
from fairness_metrics import evaluate_fairness, print_evaluation_results



from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def train_fairdi_complete(train_dataset, test_dataset, sensitive_attr,
                         num_classes=2, device='cuda', backbone='DinoVisionTransformer'):
    print("Preparing grouped datasets...")
    groups = prepare_grouped_datasets(train_dataset, sensitive_attr)
    
    sampler = GroupBalancedSampler(train_dataset, sensitive_attr, batch_size=150, groups=groups)
    train_loader = DataLoader(train_dataset, batch_size=150, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=150, shuffle=False, num_workers=0)

    print("\nStep 0: Training backbone with Fair Identity Scaling...")
    backbone_model = BackboneModel(num_classes=num_classes, backbone=backbone).to(device)
    trained_backbone = BackboneTrainer().train(backbone_model, train_loader, groups, device=device)

    print("\nStep 1: Training specialized teachers...")
    teachers = TeacherTrainer().train_teachers(trained_backbone, groups, train_dataset, num_epochs=50, device=device)

    print("\nStep 2: Training student with knowledge distillation...")
    student_model = Student(trained_backbone).to(device)
    student = student_model.train_student(teachers, train_loader, device=device)

    print("\nEvaluating final model...")
    results = evaluate_fairness(student, test_loader, groups, device)
    print_evaluation_results(results)

    return teachers, student, results
