import torch
import torch.nn.functional as F
from collections import defaultdict
from pretraining.evaluation.base_evaluation import BaseEvaluator
import wandb
from pretraining.utils.distributed import is_main_process

def compute_intra_class_variance(features, labels, num_classes):
    class_feats = defaultdict(list)

    for f, y in zip(features, labels):
        class_feats[int(y.item())].append(f)

    variances = []
    for cls in range(num_classes):
        feats_list = class_feats[cls]
        if len(feats_list) == 0:
            continue
        feats = torch.stack(feats_list)  
        mean = feats.mean(dim=0, keepdim=True)
        var = ((feats - mean) ** 2).sum(dim=1).mean()
        variances.append(var.item())

    avg_intra_class_var = sum(variances) / num_classes
    return avg_intra_class_var, variances

def compute_inter_class_distance(features, labels, num_classes, metric="euclidean"):
    class_means = []
    valid_class_indices = []

    for cls in range(num_classes):
        cls_feats = features[labels == cls]
        if cls_feats.numel() == 0:
            continue  
        cls_mean = cls_feats.mean(dim=0)
        class_means.append(cls_mean)
        valid_class_indices.append(cls)

    if len(class_means) < 2:
        return 0.0, None  

    centers = torch.stack(class_means)  

    if metric == "euclidean":
        dists = torch.cdist(centers, centers, p=2)
    elif metric == "cosine":
        dists = 1 - F.cosine_similarity(centers.unsqueeze(1), centers.unsqueeze(0), dim=-1)
    else:
        raise ValueError("Unknown distance metric")

    dists_no_diag = dists[~torch.eye(dists.size(0), dtype=bool, device=dists.device)]
    avg_inter_class_dist = dists_no_diag.mean().item()

    return avg_inter_class_dist, dists

class LatentSpaceEvaluator(BaseEvaluator):
    def __init__(self, eval_name, eval_freq, num_classes, dataset=None):
        super().__init__(eval_name, eval_freq)
        self.num_classes = num_classes
    
    def evaluate(self, step, encoder, embeddings, labels=None, shape=None, logger=None):
        if not is_main_process():
            return
        if labels is None:
            return None
        intra_var, per_class_var = compute_intra_class_variance(embeddings, labels, self.num_classes)
        inter_dist, _ = compute_inter_class_distance(embeddings, labels, self.num_classes)
        return intra_var, inter_dist
    
    def _log(self, step, input, results, logger, shape):
        if not is_main_process():
            return None
        intra_var, inter_dist = results
        ratio = inter_dist/(intra_var+1e-6)
        wandb.log({
            f"eval/{self.eval_name}/intra_class_variance": intra_var,
            f"eval/{self.eval_name}/inter_class_distance": inter_dist,
            f"eval/{self.eval_name}/distance_to_variance_ratio": inter_dist / (intra_var + 1e-6),
        }, step=step)
        logger.info(f'inter class distance ({inter_dist:.4f}) to intra class variance ({intra_var:.4f}) ratio: {ratio:.4f}')
        return {'inter_class_distance': (inter_dist, -1), 'intra_class_var': (intra_var, 1), 'dist_to_var_ratio': (ratio,-1)}