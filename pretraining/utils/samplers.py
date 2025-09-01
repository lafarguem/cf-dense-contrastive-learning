from abc import ABC, abstractmethod
import torch

class BaseSampler(ABC):
    @abstractmethod
    def __call__(self, *args, coords=None, flat=True):
        pass

class IdentitySampler(BaseSampler):
    def __init__(self):
        pass
    
    def __call__(self, *args, coords=None, flat=True):
        if flat:
            return tuple([x.reshape(*x.shape[:-2], -1)] for x in args), None
        else:
            return tuple([x] for x in args), None

class RandomOverlapSampler(BaseSampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.fallback = RandomSampler(self.num_samples)

    def __call__(self, *args, coords=None, return_idx=False, flat=True):
        if not flat:
            raise ValueError('flat=False is incompatible with RandomOverlapSampler')
        if coords is None or args[0].ndim == 5:
            return self.fallback(*args)

        B, _, H, W = args[0].shape
        V = len(coords)
        device = args[0].device

        ys, xs = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing='ij'
        )
        coords_flat = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).float()
        coords_flat = coords_flat.view(-1, 3).T

        M_view2canon = torch.stack(coords, dim=0)
        
        proj = M_view2canon @ coords_flat.unsqueeze(0).unsqueeze(0)
        proj_xy = proj[:, :, :2] / proj[:, :, 2:].clamp(min=1e-8)


        inside = (
            (proj_xy[..., 0, :] >= 0) & (proj_xy[..., 0, :] < 1) &
            (proj_xy[..., 1, :] >= 0) & (proj_xy[..., 1, :] < 1)
        )

        overlap_mask = inside.all(dim=0)

        h_indices = torch.zeros((B, self.num_samples), dtype=torch.long, device=device)
        w_indices = torch.zeros((B, self.num_samples), dtype=torch.long, device=device)
        labels = torch.zeros((B, self.num_samples), dtype=torch.long, device=device)

        for b in range(B):
            overlap_idx = torch.where(overlap_mask[b])[0]
            nonoverlap_idx = torch.where(~overlap_mask[b])[0]

            num_overlap = overlap_idx.numel()
            if num_overlap >= self.num_samples:
                chosen = overlap_idx[torch.randperm(num_overlap, device=device)[:self.num_samples]]
                labels[b] = 1
            else:
                chosen_overlap = overlap_idx
                num_needed = self.num_samples - num_overlap
                chosen_nonoverlap = nonoverlap_idx[torch.randint(0, nonoverlap_idx.numel(), (num_needed,), device=device)]
                chosen = torch.cat([chosen_overlap, chosen_nonoverlap])
                labels[b, :num_overlap] = 1

            h_indices[b] = chosen // W
            w_indices[b] = chosen % W

        samples_list, h_args, w_args = [], [], []
        pts = torch.stack([w_indices/W, h_indices/H, torch.ones_like(h_indices)], dim=-1).float()

        for v, arg in enumerate(args):
            M_view = M_view2canon[v % V]
            proj = torch.bmm(pts, M_view.transpose(1, 2))
            proj_xy = proj[..., :2] / proj[..., 2:].clamp(min=1e-8)

            w_proj = (proj_xy[..., 0] * W).round().clamp(0, W - 1).long()
            h_proj = (proj_xy[..., 1] * H).round().clamp(0, H - 1).long()

            C = arg.shape[-3]
            arg_flat = arg.view(B, C, H * W)
            linear_idx = (h_proj * W + w_proj).unsqueeze(1).expand(-1, C, -1)
            pixels = torch.gather(arg_flat, dim=2, index=linear_idx)

            samples_list.append([pixels])
            h_args.append(h_proj)
            w_args.append(w_proj)

        if return_idx:
            return tuple(samples_list), labels, h_args, w_args
        else:
            return tuple(samples_list), labels

class RandomSampler(BaseSampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __call__(self, *args, coords=None, flat=True):
        if not flat:
            raise ValueError('flat=False is incompatinle with RandomSampler')
        x = args[0]
        device=x.device
        if args[0].ndim == 5:
            B,V,C,H,W = x.shape
        elif args[0].ndim == 4:
            B,C,H,W = x.shape
            V = None
        HW = H * W
        
        idx = torch.randint(0, HW, (B, self.num_samples), device=device)

        sampled_blocks = []
        if V is None:
            for arg in args:
                arg_flat = arg.view(*arg.shape[:-2], -1)
                sampled = arg_flat.gather(
                    2, idx.view(B, 1, self.num_samples).expand(*arg.shape[:-2], self.num_samples)
                )
                sampled_blocks.append([sampled])
        else:
            for arg in args:
                arg_flat = arg.view(*arg.shape[:-2], -1)
                sampled = arg_flat.gather(
                    3, idx.view(B, 1, 1, self.num_samples).expand(*arg.shape[:-2], self.num_samples)
                )
                sampled_blocks.append([sampled])

        return tuple(sampled_blocks), None

class StrideSampler(BaseSampler):
    def __init__(self, stride):
        self.stride = stride

    def __call__(self, *args, coords=None, flat=True):
        if flat:
            return tuple([x[..., ::self.stride, ::self.stride].reshape(*x.shape[:-2], -1)] for x in args), None
        else:
            return tuple([x[..., ::self.stride, ::self.stride]] for x in args), None

class BlockSampler(BaseSampler):
    def __init__(self, block_size):
        self.block_size = block_size

    def __call__(self, *args, coords=None, flat=True):
        N, H, W = args[0].shape[0], args[0].shape[-2], args[0].shape[-1]
        block_samples = [[] for _ in args]
        for i in range(0, H, self.block_size):
            for j in range(0, W, self.block_size):
                h_end = min(i + self.block_size, H)
                w_end = min(j + self.block_size, W)
                for idx, x in enumerate(args):
                    block = x[..., i:h_end, j:w_end]
                    if flat:
                        block_samples[idx].append(block.reshape(*x.shape[:-2],-1))
                    else:
                        block_samples[idx].append(block)

        return tuple(block_samples), None
    
SAMPLER_REGISTRY = {
    "stride": StrideSampler,
    "block": BlockSampler,
    "random": RandomSampler,
    "random_overlap": RandomOverlapSampler,
}

def get_sampler(sampler) -> BaseSampler:
    if sampler is None:
        return IdentitySampler()
    name = sampler["name"]
    kwargs = {k: v for k, v in sampler.items() if k != "name"}
    if name not in SAMPLER_REGISTRY:
        raise ValueError(f"Unknown sampler: {name}")
    return SAMPLER_REGISTRY[name](**kwargs)