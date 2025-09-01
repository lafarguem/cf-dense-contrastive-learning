import torch
import torch.nn as nn
import torch.nn.functional as F
from pretraining.utils.samplers import get_sampler
from pretraining.utils.common import reduce_segmentation_mask, masked_log_softmax
from pretraining.utils.distributed import all_gather_tensor, get_world_size

class MatchingRuleSeperateViews:
    def __init__(
            self,
            inter_view_same_pixel = 1,
            inter_view_same_class = 1,
            inter_view_different_class = 0,
            inter_image_same_class = 1,
            inter_image_different_class = 0,
        ):
        """
        -1: do not consider
        0: consider as negative pair
        1: consider as positive pair

        The union of these conditions covers all pairs of pixels. 
        inter_view_same_pixel is the only conditon to overlap with other conditions.
        inter_view_same_pixel is a fallback if labels are not accessible and should be set to 1. If no labels are provided, the 'class' becomes same pixel.
        This will not impact if segmentation masks change between views as inter_view_same_class will be applied instead.
        """
        self.inter_view_same_pixel = inter_view_same_pixel
        self.inter_view_same_class = inter_view_same_class
        self.inter_view_different_class = inter_view_different_class
        self.inter_image_same_class = inter_image_same_class
        self.inter_image_different_class = inter_image_different_class
    
    def get_masks(self, labels_student=None, labels_teacher=None, labels_sampler=None, shape=None, device=None, chunk_divs=1):
        B,HW = shape
        N = B * HW

        image_ids = torch.arange(B).repeat_interleave(HW).to(device)  
        pixel_coords = torch.arange(HW, device=device).repeat(B).view(-1)  


        chunk_size = (N + chunk_divs - 1) // chunk_divs
        numerator_mask = torch.zeros((N, N), dtype=torch.float32, device=device)
        denom_mask = torch.zeros((N, N), dtype=torch.float32, device=device)
        for i in range(0, N, chunk_size):
            row_end = min(i + chunk_size, N)
            for j in range(i, N, chunk_size):
                col_end = min(j + chunk_size, N)
                rows = slice(i, row_end)
                cols = slice(j, col_end)
                
                partial_same_image = (image_ids[rows, None] == image_ids[None, cols])  
                partial_same_pixel = (pixel_coords[rows, None] == pixel_coords[None, cols])  
                if labels_sampler is not None:
                    flat_sampler_labels = labels_sampler.view(-1)
                    pixel_aligned = (flat_sampler_labels == 1)[rows, None] & (flat_sampler_labels == 1)[None, cols]
                    partial_same_pixel &= pixel_aligned  
                if labels_student is not None:
                    partial_same_class = (labels_student[rows, None] == labels_teacher[None, cols])
            
                partial_mask = torch.full((rows.stop - rows.start, cols.stop - cols.start), -1, device=device)

                if labels_student is None:
                    partial_mask[partial_same_image & partial_same_pixel] = self.inter_view_same_pixel
                    partial_mask[partial_same_image & ~partial_same_pixel] = self.inter_view_different_class
                    partial_mask[~partial_same_image] = self.inter_image_different_class
                else:
                    partial_mask[partial_same_image & partial_same_class] = self.inter_view_same_class
                    partial_mask[partial_same_image & ~partial_same_class] = self.inter_view_different_class
                    partial_mask[~partial_same_image & partial_same_class] = self.inter_image_same_class
                    partial_mask[~partial_same_image & ~partial_same_class] = self.inter_image_different_class
                
                if i == j:
                    numerator_mask[rows, cols] = (partial_mask == 1).float()
                    denom_mask[rows, cols] = (partial_mask >= 0).float()
                else:
                    num_part = (partial_mask == 1).float()
                    denom_part = (partial_mask >= 0).float()

                    numerator_mask[rows, cols] = num_part
                    numerator_mask[cols, rows] = num_part.T

                    denom_mask[rows, cols] = denom_part
                    denom_mask[cols, rows] = denom_part.T
                    
        return numerator_mask, denom_mask

class SupervisedCoordContrastiveLoss(nn.Module):
    def __init__(self, temperature, sampler = None, matching_rule = None, bg_anchors = False, use_labels=True, chunk_divs=1):
        super(SupervisedCoordContrastiveLoss, self).__init__()
        self.sampler = get_sampler(sampler)
        self.temperature = temperature
        if matching_rule is None:
            self.matching_rule = MatchingRuleSeperateViews()
        else:
            self.matching_rule = MatchingRuleSeperateViews(**matching_rule)
        self.bg_anchors = bg_anchors
        self.use_labels = use_labels
        self.chunk_divs = chunk_divs
    
    def forward(self, out, dual, labels=None):
        B,V,C,H,W = out.projected_dense_embeddings.shape
        use_labels = labels is not None and self.use_labels
        if dual:
            qs = torch.unbind(out.projected_dense_embeddings, dim=1)
            ks = torch.unbind(out.projected_teacher_dense_embeddings, dim=1)
        else:
            qs = torch.unbind(out.projected_dense_embeddings, dim=1)
            ks = torch.unbind(out.projected_dense_embeddings.clone(), dim=1)
        if use_labels:
            reduced_labels = reduce_segmentation_mask(labels, H,W)
            labelss = reduced_labels.unbind(dim=1)
        
        coords = out.coords
        num_views = out.num_views

        def get_loss(q,k,labels_q=None,labels_k=None, coords_q=None, coords_k=None):
            if use_labels:
                sampled, sampler_labels = self.sampler(q, k, labels_q, labels_k, coords=[coords_q, coords_k], flat=True)
            else:
                sampled, sampler_labels = self.sampler(q, k, coords=[coords_q, coords_k], flat=True)
            total_loss = 0
            num_blocks = len(sampled[0])
            for sample in zip(*sampled):
                total_loss += self._forward(*sample, sampler_labels=sampler_labels)
            return total_loss/num_blocks

        dense_loss = 0
        for i in range(1,num_views):
            if use_labels:
                labels_q, labels_k = labelss[0], labelss[i]
            else:
                labels_q, labels_k = None, None
            if coords is not None:
                coords_q, coords_k = coords[0], coords[i]
            else:
                coords_q, coords_k = None, None
            q, k = qs[0], ks[i]
            dense_loss += get_loss(q, k, labels_q, labels_k, coords_q, coords_k)
            dense_loss += get_loss(k, q, labels_k, labels_q, coords_k, coords_q)

        return dense_loss / (num_views-1)
    
    def _forward(self, student_embeddings, teacher_embeddings, student_labels=None, teacher_labels=None, sampler_labels=None):
        B, C, HW = student_embeddings.shape
        student_flat = student_embeddings.permute(0, 2, 1).reshape(B*HW, C)
        teacher_flat = teacher_embeddings.permute(0, 2, 1).reshape(B*HW, C)

        device = student_flat.device

        student_flat = F.normalize(student_flat, dim=1)
        teacher_flat = F.normalize(teacher_flat, dim=1)

        if student_labels is not None:
            student_labels_flat = student_labels.flatten()
            teacher_labels_flat = teacher_labels.flatten()
        else:
            student_labels_flat = None
            teacher_labels_flat = None

        if get_world_size() > 1:
            student_flat = all_gather_tensor(student_flat)
            teacher_flat = all_gather_tensor(teacher_flat)
            if sampler_labels is not None:
                sampler_labels = all_gather_tensor(sampler_labels)
            if student_labels is not None:
                student_labels_flat = all_gather_tensor(student_labels_flat)
                teacher_labels_flat = all_gather_tensor(teacher_labels_flat)

        logits = torch.matmul(student_flat, teacher_flat.T) / self.temperature  

        numerator_mask, denom_mask = self.matching_rule.get_masks(student_labels_flat, teacher_labels_flat, sampler_labels, 
                                                                  (B*get_world_size(), HW), device)

        log_prob = masked_log_softmax(logits, denom_mask, dim=1)

        numerator_sums = numerator_mask.sum(1)
        mean_log_prob_pos = (numerator_mask * log_prob.masked_fill(torch.isneginf(log_prob), 0.0)).sum(1) / (numerator_sums + 1e-8)

        valid_anchors = numerator_sums > 1e-8
        per_anchor_loss = -mean_log_prob_pos[valid_anchors]

        if student_labels is not None and not self.bg_anchors:
            non_bg_mask = (student_labels_flat[valid_anchors] != 0)
            per_anchor_loss = per_anchor_loss[non_bg_mask]

        loss = per_anchor_loss.mean()
        return loss