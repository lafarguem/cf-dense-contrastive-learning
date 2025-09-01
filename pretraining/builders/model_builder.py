from hydra.utils import instantiate
from pretraining.utils.distributed import get_local_rank, DDPWrapper

def build_model(cfg):
    local_rank = get_local_rank()
    model = instantiate(cfg.model, runtime_args=cfg.runtime)
    print(f"Rank {local_rank} model parameters:", sum(p.numel() for p in model.parameters()))
    print("moving model to local rank: ", local_rank)
    model = model.to(cfg.runtime.device)
    if cfg.runtime.distributed:
        print("distributing model")
        model = DDPWrapper(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
        model.set_toggle = model.module.set_toggle
        model.set_global_return = model.module.set_global_return
        model.set_dense_return = model.module.set_dense_return
        
    return model