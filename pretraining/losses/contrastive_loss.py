import torch.nn as nn
from pretraining.models.contrastive.output import DualBranchContrastiveOutput
import torch
from pretraining.utils.distributed import is_main_process, get_world_size, reduce_tensor
import wandb

class DenseContrastiveLoss(nn.Module):
    def __init__(self, dense_loss, global_loss=None, global_loss_weight = 0, dual_branch=True, always_compute_global=False, always_compute_dense=False):
        super(DenseContrastiveLoss, self).__init__()
        self.dense_loss = dense_loss
        self.global_loss = global_loss
        self.global_loss_weight = global_loss_weight
        self.dual_branch = dual_branch
        self.always_compute_global = always_compute_global
        self.always_compute_dense = always_compute_dense
    
    def forward(self, contrastive_output: DualBranchContrastiveOutput, data=None, return_sub_losses=False):
        if data is not None:
            labels = data.get("mask", None)
        else:
            labels = None
        loss = 0
        if (self.global_loss_weight < (1.0 - 1e-6)) or self.always_compute_dense:
            dense_loss = self.dense_loss(contrastive_output, self.dual_branch, labels)
            loss += (1-self.global_loss_weight) * dense_loss
        else:
            dense_loss = None

        if (self.global_loss_weight > 1e-6) or self.always_compute_global:
            qs = torch.unbind(contrastive_output.projected_instance_embeddings.clone(),dim=1)
            if self.dual_branch:
                ks = torch.unbind(contrastive_output.projected_teacher_instance_embeddings.clone(),dim=1)
            else:
                ks = torch.unbind(contrastive_output.projected_instance_embeddings.clone(), dim=1)
            global_loss_1 = [self.global_loss(
                qs[0],
                ks[i],
            ) for i in range(1, contrastive_output.num_views)]
            global_loss_2 = [self.global_loss(
                qs[i],
                ks[0],
            ) for i in range(1, contrastive_output.num_views)]
            global_loss = sum(global_loss_1) + sum(global_loss_2)
            loss += self.global_loss_weight * global_loss
        else:
            global_loss = None
        
        if return_sub_losses:
            return loss, dense_loss, global_loss