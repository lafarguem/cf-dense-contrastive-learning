from torch.nn.parallel import DistributedDataParallel
from hydra.utils import instantiate
from pretraining.utils.distributed import get_local_rank

def build_model(cfg):
    local_rank = get_local_rank()
    model = instantiate(cfg.model)
    print(f"Rank {local_rank} model parameters:", sum(p.numel() for p in model.parameters()))
    print("moving model to local rank: ", local_rank)
    model = model.to(f'cuda:{local_rank}')
    if cfg.runtime.distributed:
        print("distributing model")
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    return model