import torch
import torch.nn as nn
import torch.nn.functional as F
from pretraining.utils.common import compute_pairwise_dist, compute_centers
from pretraining.utils.samplers import get_sampler
from pretraining.utils.similarities import get_similarity
from pretraining.utils.debug import debug_image

class PixProLoss(nn.Module):
    def __init__(self, similarity, pos_pixel_dist, sampler=None):
        super().__init__()
        self.similarity = get_similarity(similarity)
        self.sampler = get_sampler(sampler)
        self.pos_pixel_dist = pos_pixel_dist

    def forward(self, out, dual, labels=None):
        qs = torch.unbind(out.projected_dense_embeddings, dim=1)
        if dual:
            ks = torch.unbind(out.projected_teacher_dense_embeddings, dim=1)
        else:
            ks = qs.clone()
        coords = out.coords
        num_views = out.num_views

        dense_loss = 0
        for i in range(1,num_views):
            q,k = qs[0], ks[i]
            center_q_x, center_q_y, center_k_x, center_k_y = compute_centers(q, k, coords[0], coords[i])
            if self.sampler is not None:
                total_loss = 0
                sampled, sampler_labels = self.sampler(q, k, center_q_x, center_q_y, center_k_x, center_k_y)
                num_blocks = len(sampled[0])
                for sample in zip(*sampled):
                    total_loss += self._forward(*sample)
                dense_loss += total_loss/num_blocks
            else:
                dense_loss += self._forward(q, k, center_q_x, center_q_y, center_k_x, center_k_y)

            q,k = qs[i], ks[0]
            center_q_x, center_q_y, center_k_x, center_k_y = compute_centers(q, k, coords[0], coords[i])
            if self.sampler is not None:
                total_loss = 0
                sampled, sampler_labels = self.sampler(q, k, center_q_x, center_q_y, center_k_x, center_k_y)
                num_blocks = len(sampled[0])
                for sample in zip(*sampled):
                    total_loss += self._forward(*sample)
                dense_loss += total_loss/num_blocks
            else:
                dense_loss += self._forward(q, k, center_q_x, center_q_y, center_k_x, center_k_y)

        return dense_loss / (num_views - 1)

    def _forward(self, q, k, center_q_x, center_q_y, center_k_x, center_k_y):
        N,C,H,W = q.shape
        q = q.reshape(N,C,H*W)
        k = k.reshape(N,C,H*W)

        q = F.normalize(q, p=2, dim=1, eps=1e-8)
        k = F.normalize(k, p=2, dim=1, eps=1e-8)

        dist_center = compute_pairwise_dist(center_q_x.reshape(N,H*W), center_q_y.reshape(N,H*W), 
                                            center_k_x.reshape(N,H*W), center_k_y.reshape(N,H*W))

        pos_mask = (dist_center < self.pos_pixel_dist).float().detach()

        sample_idx = 0
        pos_mask_sample = pos_mask[sample_idx]

        debugs = []
        for pixel_idx in range(H * W):
            mask_1d = pos_mask_sample[pixel_idx]
            mask_2d = mask_1d.reshape(H, W)
            debugs.append((mask_2d, f'pix_{pixel_idx}.png'))
        debug_image(debugs)

        logit = self.similarity(q,k)

        loss = (logit * pos_mask).sum(-1).sum(-1) / (pos_mask.sum(-1).sum(-1) + 1e-6)
        loss = -2*loss.mean()

        return loss
    
