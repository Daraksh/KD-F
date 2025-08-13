import torch
import torch.nn as nn

class FISLoss(nn.Module):
    def __init__(self, base_loss_fn=None, initial_alpha=0.8, final_alpha=0.2,
                 warmup_epochs=3, eps=1e-6):
        super().__init__()
        self.base_loss_fn = base_loss_fn if base_loss_fn else nn.CrossEntropyLoss(reduction='none')
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.momentum = 0.6
        self.eps = eps
        self.running_group_stats = None

    def reset_running_stats(self):
        self.running_group_stats = None

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _get_adaptive_alpha(self):
        if self.current_epoch < self.warmup_epochs:
            return 1.0
        progress = min(1.0, (self.current_epoch - self.warmup_epochs) / 30.0)
        return self.initial_alpha + progress * (self.final_alpha - self.initial_alpha)

    def _compute_group_weights_running(self, groups, losses):
        unique_groups = torch.unique(groups)
        current_stats = {}
        for g in unique_groups:
            mask = groups == g
            if mask.any():
                current_stats[g.item()] = {
                    'loss': losses[mask].mean().detach()
                }

        if self.running_group_stats is None:
            self.running_group_stats = current_stats
        else:
            for gid, stats in current_stats.items():
                if gid in self.running_group_stats:
                    prev = self.running_group_stats[gid]['loss']
                    self.running_group_stats[gid]['loss'] = (
                        self.momentum * prev + (1 - self.momentum) * stats['loss']
                    )
                else:
                    self.running_group_stats[gid] = stats

        group_weights = torch.ones_like(groups, dtype=torch.float, device=groups.device)
        max_loss = max([s['loss'] for s in self.running_group_stats.values()]) + self.eps
        for g in unique_groups:
            weight = self.running_group_stats[g.item()]['loss'] / max_loss
            group_weights[groups == g] = weight

        return group_weights

    def forward(self, outputs, targets, groups):
        groups = groups.to(outputs.device)
        losses = self.base_loss_fn(outputs, targets)

        if self.current_epoch < self.warmup_epochs:
            return losses.mean()

        individual_w = torch.sigmoid(losses - losses.mean())
        individual_w = torch.clamp(individual_w, 0.1, 10.0)
        group_w = self._compute_group_weights_running(groups, losses)
        alpha = self._get_adaptive_alpha()
        final_w = alpha * individual_w + (1 - alpha) * group_w
        final_w = torch.clamp(final_w, 0.01, 10.0)
        weighted = final_w * losses
        return (weighted.mean() / final_w.mean()).clamp(max=10.0)
