import torch
import torch.nn as nn
import torch.nn.functional as F
from pretraining.utils.distributed import get_world_size
from torch.distributed.nn.functional import all_gather
from pretraining.utils.similarities import CosineSimilarity
import math

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, similarity=CosineSimilarity(), eps=1e-6):
        super().__init__()
        self.temperature = temperature
        self.world_size = get_world_size()
        self.similarity = similarity
        self.eps = eps

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1, eps=1e-8)
        z2 = F.normalize(z2, dim=1, eps=1e-8)

        if self.world_size > 1:
            z1_dist = all_gather(z1.contiguous())
            z2_dist = all_gather(z2.contiguous())
        else:
            z1_dist = z1
            z2_dist = z2

        z = torch.cat([z1, z2], dim=0)
        z_dist = torch.cat([z1_dist, z2_dist], dim=0)

        cov = torch.mm(z, z_dist.t().contiguous())
        sim = torch.exp(cov / self.temperature)
        neg = sim.sum(dim=-1)

        row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / self.temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=self.eps)

        pos = torch.exp(torch.sum(z1 * z2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        return -torch.log(pos / (neg + self.eps)).mean()