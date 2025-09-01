import torch
import torch.nn as nn
import torch.nn.functional as F
from pretraining.utils.samplers import get_sampler
from pretraining.utils.common import reduce_segmentation_mask, masked_log_softmax, compute_pairwise_dist_sqrd
from pretraining.utils.distributed import all_gather_tensor, get_world_size

class MatchingRuleAllViews:
    def __init__(
            self,
            intra_view_same_class = 1,
            intra_view_different_class = 0,
            inter_view_same_pixel = 1,
            inter_view_same_class = 1,
            inter_view_different_class = 0,
            inter_image_same_class = -1,
            inter_image_different_class = -1,
            self_similarity = -1,
        ):
        """
        -1: do not consider
        0: consider as negative pair
        1: consider as positive pair

        The union of these conditions covers all pairs of pixels. 
        self_similarity and inter_view_same_pixel are the only conditons to overlap with other conditions.
        inter_view_same_pixel is a fallback if labels are not accessible and should be set to 1. If no labels are provided, the 'class' becomes same pixel.
        This will not impact if segmentation masks change between views as inter_view_same_class will be applied instead.
        self_similarity has highest probability and should be set to 0 in most cases.
        """
        self.intra_view_same_class = intra_view_same_class
        self.intra_view_different_class = intra_view_different_class
        self.inter_view_same_pixel = inter_view_same_pixel
        self.inter_view_same_class = inter_view_same_class
        self.inter_view_different_class = inter_view_different_class
        self.inter_image_same_class = inter_image_same_class
        self.inter_image_different_class = inter_image_different_class
        self.self_similarity = self_similarity
    
    @torch.no_grad()
    def get_masks(self, pixel_positions=None, distance_threshold = 15, labels=None, shape=None, device=None, chunk_divs=1):
        B, V, HW = shape
        if pixel_positions is not None:
            pixel_positions = pixel_positions.permute(0,1,3,2).reshape(-1,2)
        
        N = B * V * HW

        image_ids = torch.arange(B).repeat_interleave(V * HW).to(device)  
        view_ids = torch.arange(V, device=device).repeat_interleave(HW).repeat(B)  
        pixel_coords = torch.arange(HW, device=device).repeat(B * V).view(-1)  

        numerator_mask = torch.zeros((N, N), dtype=torch.float32, device=device)
        denom_mask = torch.zeros((N, N), dtype=torch.float32, device=device)

        chunk_size = (N + chunk_divs - 1) // chunk_divs

        for i in range(0, N, chunk_size):
            row_end = min(i + chunk_size, N)
            for j in range(i, N, chunk_size):
                col_end = min(j + chunk_size, N)
                rows = slice(i, row_end)
                cols = slice(j, col_end)
                partial_same_image = image_ids[rows, None] == image_ids[None, cols]
                partial_same_view = view_ids[rows, None] == view_ids[None, cols]
                partial_same_pixel = pixel_coords[rows, None] == pixel_coords[None, cols]
                if labels is None:
                    pixel_distances = compute_pairwise_dist_sqrd(pixel_positions[rows,0], pixel_positions[rows,1], pixel_positions[cols,0], pixel_positions[cols,1])
                    partial_same_class = pixel_distances < (distance_threshold ** 2)
                else:
                    partial_same_class = labels[rows, None] == labels[None, cols]

                partial_mask = torch.full((rows.stop - rows.start, cols.stop - cols.start), -1, device=device)

                partial_mask[partial_same_image & partial_same_view & partial_same_class] = self.intra_view_same_class
                partial_mask[partial_same_image & partial_same_view & ~partial_same_class] = self.intra_view_different_class
                partial_mask[partial_same_image & ~partial_same_view & partial_same_class] = self.inter_view_same_class
                partial_mask[partial_same_image & ~partial_same_view & ~partial_same_class] = self.inter_view_different_class

                if labels is None:
                    partial_mask[~partial_same_image] = self.inter_image_different_class
                else:
                    partial_mask[~partial_same_image & partial_same_class] = self.inter_image_same_class
                    partial_mask[~partial_same_image & ~partial_same_class] = self.inter_image_different_class
                
                partial_mask[partial_same_image & partial_same_pixel & partial_same_view] = self.self_similarity

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

def generate_coordinate_map(coords, H=224, W=224):
    device = coords[0].device

    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, H, device=device),
        torch.linspace(0, 1, W, device=device),
        indexing='ij'
    )
    coords_flat = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).float() 
    coords_flat = coords_flat.view(-1, 3).T 

    M_view2canon = torch.stack(coords, dim=1)
    M_canon2view = torch.inverse(M_view2canon)
    B,V = M_view2canon.shape[0], M_view2canon.shape[1]

    proj = M_canon2view @ coords_flat.unsqueeze(0).unsqueeze(0)
    proj_xy = proj[:, :, :2] / proj[:, :, 2:].clamp(min=1e-8)

    return proj_xy.reshape(B,V,2,H,W)

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature, sampler = None, matching_rule = None, bg_anchors = False, use_labels=True, distance_threshold=0.1, chunk_divs=1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.sampler = get_sampler(sampler)
        self.temperature = temperature
        if matching_rule is None:
            self.matching_rule = MatchingRuleAllViews()
        else:
            self.matching_rule = MatchingRuleAllViews(**matching_rule)
        self.bg_anchors = bg_anchors
        self.use_labels = use_labels
        self.distance_threshold = distance_threshold
        self.chunk_divs = chunk_divs

    def forward(self, out, dual, labels=None):
        q = out.projected_dense_embeddings
        B,V,C,H,W = q.shape
        coords = out.coords
        if dual:
            k = out.projected_teacher_dense_embeddings
        else:
            k = q
        if labels is None or not self.use_labels:
            if coords is None:
                pixel_positions  = torch.stack((
                                        torch.linspace(0, 1, W, device=q.device).unsqueeze(0).repeat(H, 1), 
                                        torch.linspace(0, 1, H, device=q.device).unsqueeze(1).repeat(1, W)  
                                    ), dim=0).expand(B, V, 2, H, W)
            else:
                pixel_positions = generate_coordinate_map(coords, H,W)
            (sampled_q,sampled_k,sampled_pixel_positions), sampler_labels = self.sampler(q,k,pixel_positions,flat=True)
            sampled_labels = [None] * len(sampled_q)
        else:
            reduced_labels = reduce_segmentation_mask(labels, H, W)
            (sampled_q,sampled_k,sampled_labels), sampler_labels = self.sampler(q,k,reduced_labels,flat=True)
            sampled_pixel_positions = [None] * len(sampled_q)
        num_blocks = len(sampled_q)
        total_loss = 0
        for i in range(num_blocks):
            total_loss += self._forward(sampled_q[i], sampled_k[i], labels=sampled_labels[i], pixel_positions=sampled_pixel_positions[i])
        return total_loss / num_blocks
    
    def _forward(self, student_embeddings, teacher_embeddings, labels=None, pixel_positions=None):
        B, V, C, HW = student_embeddings.shape
        device = student_embeddings.device

        student_flat = student_embeddings.permute(0, 1, 3, 2).reshape(B*V*HW, C)
        teacher_flat = teacher_embeddings.permute(0, 1, 3, 2).reshape(B*V*HW, C)
        student_flat = F.normalize(student_flat, dim=1)
        teacher_flat = F.normalize(teacher_flat, dim=1)

        if labels is not None:
            labels_flat = labels.flatten()
        else:
            labels_flat = None

        if get_world_size() > 1:
            student_flat = all_gather_tensor(student_flat)
            teacher_flat = all_gather_tensor(teacher_flat)
            if labels is not None:
                labels_flat = all_gather_tensor(labels_flat)
            if pixel_positions is not None:
                pixel_positions = all_gather_tensor(pixel_positions)
        
        N = student_flat.size(0)

        logits = torch.matmul(student_flat, teacher_flat.T) / self.temperature 

        numerator_mask, denom_mask = self.matching_rule.get_masks(pixel_positions=pixel_positions,labels=labels_flat, chunk_divs=self.chunk_divs,
                                                                  shape=(B*get_world_size(), V, HW), device=device, distance_threshold=self.distance_threshold)

        log_prob = masked_log_softmax(logits, denom_mask, dim=1)

        numerator_sums = numerator_mask.sum(1)
        mean_log_prob_pos = (numerator_mask * log_prob.masked_fill(torch.isneginf(log_prob), 0.0)).sum(1) / (numerator_sums + 1e-8)

        valid_anchors = numerator_sums > 1e-8
        loss = -mean_log_prob_pos[valid_anchors].mean()

        if labels is not None and not self.bg_anchors:
            non_bg_mask = (labels_flat != 0).float()
            loss = (loss * non_bg_mask[valid_anchors]).sum() / (non_bg_mask[valid_anchors].sum() + 1e-8)

        return loss