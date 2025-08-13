class Config:
    backbone_trainer_config = {
        'num_epochs': 100,
        'optimizer_lr': 3e-5,
        'optimizer_weight_decay': 1e-4,
        'fis_warmup': 6
    }

    teacher_config = {
        'num_epochs': 20,
        'optimizer_lr': 1e-2,
        'optimizer_weight_decay': 1e-4
    }

    student_config = {
        'tau': 1.5,
        'lambda_kd': 0.95,
        'num_epochs': 20
    }

    fis_params = {
        'eps': 1e-6,
        'alpha_start': 1.0,
        'alpha_end': 0.2,
        'tau': 1.0
    }
