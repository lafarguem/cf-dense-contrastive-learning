import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x1 = F.normalize(x1, dim=-2, eps=1e-8)
        x2 = F.normalize(x2, dim=-2, eps=1e-8)
        
        if x1.dim() == 2:
            return torch.matmul(x1.T, x2)
        elif x1.dim() == 3:
            return torch.bmm(x1.transpose(1, 2), x2)
        else:
            raise ValueError("Input tensors must be 2D or 3D")

class DotProduct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        if x1.dim() == 2:
            return torch.matmul(x1.T, x2)
        elif x1.dim() == 3:
            return torch.bmm(x1.transpose(1, 2), x2)
        else:
            raise ValueError("Input tensors must be 2D or 3D")

class EuclideanDistanceSquared(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        if x1.dim() == 2:
            x2 = x2.T
            x1 = x1.T

            x1_norm = (x1 ** 2).sum(1).view(-1, 1)
            x2_norm = (x2 ** 2).sum(1).view(1, -1)
            dist = x1_norm + x2_norm - 2 * torch.matmul(x1, x2.T)
        elif x1.dim() == 3:
            x1 = x1.transpose(1, 2)
            x2 = x2.transpose(1, 2)

            x1_norm = (x1 ** 2).sum(-1).unsqueeze(2)
            x2_norm = (x2 ** 2).sum(-1).unsqueeze(1)
            dist = x1_norm + x2_norm - 2 * torch.bmm(x1, x2.transpose(1, 2))
        else:
            raise ValueError("Input tensors must be 2D or 3D")

        dist = torch.clamp(dist, min=1e-8)

class EuclideanDistance(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = EuclideanDistanceSquared()

    def forward(self, x1, x2):
        dist = self.l2(x1,x2)
        return torch.sqrt(dist)

SIMILARITY_REGISTRY = {
    "cosine": CosineSimilarity,
    "dot": DotProduct,
    "euclidean": EuclideanDistance,
    "euclidean squared": EuclideanDistanceSquared
}

def get_similarity(name):
    if name not in SIMILARITY_REGISTRY:
        raise ValueError(f"Unknown similarity: {name}")
    return SIMILARITY_REGISTRY[name]()