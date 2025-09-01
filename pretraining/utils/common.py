import torch
import torch.nn.functional as F
import numpy as np

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def recursive_to_device(input, device='cuda', non_blocking=True):
    if isinstance(input, dict):
        return {key: recursive_to_device(item, device) for key, item in input.items()}
    elif isinstance(input, list):
        return [recursive_to_device(item, device) for item in input]
    elif isinstance(input, torch.Tensor):
        return input.to(device=device, non_blocking=non_blocking)
    elif isinstance(input, str):
        return input
    else:
        raise NotImplementedError(f"Input should be of type torch.Tensor, dict or list. Not {type(input)}")

def compute_pairwise_dist(x1,y1,x2,y2):
    if not isinstance(x1, torch.Tensor):
        x1, y1, x2, y2 = map(torch.tensor, (x1, y1, x2, y2))
    
    unbatched = x1.dim() == 1
    if unbatched:
        x1 = x1.unsqueeze(0)
        y1 = y1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
        y2 = y2.unsqueeze(0)

    dx = x1.unsqueeze(2) - x2.unsqueeze(1)
    dy = y1.unsqueeze(2) - y2.unsqueeze(1)
    dist = torch.sqrt(dx ** 2 + dy ** 2)

    return dist[0] if unbatched else dist

def compute_pairwise_dist_sqrd(x1,y1,x2,y2):
    if not isinstance(x1, torch.Tensor):
        x1, y1, x2, y2 = map(torch.tensor, (x1, y1, x2, y2))
    
    unbatched = x1.dim() == 1
    if unbatched:
        x1 = x1.unsqueeze(0)
        y1 = y1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
        y2 = y2.unsqueeze(0)

    dx = x1.unsqueeze(2) - x2.unsqueeze(1)
    dy = y1.unsqueeze(2) - y2.unsqueeze(1)
    dist = dx ** 2 + dy ** 2

    return dist[0] if unbatched else dist

def getattr_nested(obj, attr_path):
    for attr in attr_path.split('.'):
        obj = getattr(obj, attr)
    return obj

def reduce_segmentation_mask(mask, target_h, target_w):
    B, V, _, H, W = mask.shape
    
    if H == target_h and W == target_w:
        return mask.long()
    
    mask = mask.reshape(B * V, 1, H, W).float()

    reduced = F.interpolate(mask, size=(target_h, target_w), mode='nearest')
    return reduced.reshape(B, V, 1, target_h, target_w).long()

def has_nan(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()

def rle_decode(shape, rles):
    width, height = shape
    out = torch.zeros((width*height))
    for i,rle_string in enumerate(rles):
        if not rle_string or rle_string == "-1":
            return None

        s = list(map(int, rle_string.strip().split()))
        starts, lengths = s[0::2], s[1::2]
        starts = np.array(starts) - 1
        ends = starts + lengths

        for start, end in zip(starts, ends):
            out[start:end] = i + 1

    out = out.reshape(1, height, width)
    return out.long()

def compute_centers(q,k,coord_q=None,coord_k=None):
    """Adapted from PixPro https://github.com/zdaxie/PixPro"""
    N, C, H, W = q.shape
    q = q.view(N, C, -1)
    k = k.view(N, C, -1)

    device = q.device
    dtype = q.dtype
    if coord_q is None:
        coord_q = torch.tensor([0, 0, 1, 1], dtype=dtype, device=device).repeat(N, 1)
    if coord_k is None:
        coord_k = coord_q

    x_array = torch.arange(0., float(W), dtype=coord_q.dtype, device=coord_q.device).view(1, 1, -1).repeat(1, H, 1)
    y_array = torch.arange(0., float(H), dtype=coord_q.dtype, device=coord_q.device).view(1, -1, 1).repeat(1, 1, W)

    q_bin_width = ((coord_q[:, 2] - coord_q[:, 0]) / W).view(-1, 1, 1)
    q_bin_height = ((coord_q[:, 3] - coord_q[:, 1]) / H).view(-1, 1, 1)
    k_bin_width = ((coord_k[:, 2] - coord_k[:, 0]) / W).view(-1, 1, 1)
    k_bin_height = ((coord_k[:, 3] - coord_k[:, 1]) / H).view(-1, 1, 1)

    q_start_x = coord_q[:, 0].view(-1, 1, 1)
    q_start_y = coord_q[:, 1].view(-1, 1, 1)
    k_start_x = coord_k[:, 0].view(-1, 1, 1)
    k_start_y = coord_k[:, 1].view(-1, 1, 1)


    center_q_x = (x_array + 0.5) * q_bin_width + q_start_x
    center_q_y = (y_array + 0.5) * q_bin_height + q_start_y
    center_k_x = (x_array + 0.5) * k_bin_width + k_start_x
    center_k_y = (y_array + 0.5) * k_bin_height + k_start_y

    return center_q_x, center_q_y, center_k_x, center_k_y

def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor, dim = 1, eps = 1e-8):
    mask = mask.to(dtype=logits.dtype, device=logits.device)

    logits = logits.masked_fill(mask == 0, float('-inf'))

    max_logits, _ = torch.max(logits, dim=dim, keepdim=True)
    max_logits = max_logits.masked_fill(max_logits == float('-inf'), 0.0)
    stable_logits = logits - max_logits

    exp_logits = torch.exp(stable_logits) * mask
    exp_sum = exp_logits.sum(dim=dim, keepdim=True) + eps

    log_prob = stable_logits - torch.log(exp_sum)

    mask_sum = mask.sum(dim=dim, keepdim=True)
    log_prob = log_prob.masked_fill(mask_sum == 0, float('-inf'))

    return log_prob

def dice_score(y_pred, y_true, epsilon=1e-6):
    num_classes = y_pred.shape[1]

    y_pred_classes = torch.argmax(y_pred, dim=1)

    y_pred_one_hot = F.one_hot(y_pred_classes, num_classes).permute(0, 3, 1, 2).float()
    y_true_one_hot = F.one_hot(y_true, num_classes).permute(0, 3, 1, 2).float()

    y_pred_flat = y_pred_one_hot.contiguous().view(y_pred_one_hot.shape[0], num_classes, -1)
    y_true_flat = y_true_one_hot.contiguous().view(y_true_one_hot.shape[0], num_classes, -1)

    intersection = (y_pred_flat * y_true_flat).sum(dim=2)
    union = y_pred_flat.sum(dim=2) + y_true_flat.sum(dim=2)

    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice.mean()

def add_prefix_suffix(d, prefix="", suffix="", v_func=lambda x: x):
    return {f"{prefix}{k}{suffix}": v_func(v) for k, v in d.items()}

def shorten_exp_name(name, limit=80):
    if "_abl" not in name:
        return name
    
    before, after = name.split("_abl", 1)
    parts = [p for p in after.split("_") if p]
    initials = [p[0] for p in parts]
    return f"{before}_abl_{''.join(initials)}"[:limit]