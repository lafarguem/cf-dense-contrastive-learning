import torch
import torch.distributed as dist
import os
import sys
from torch.distributed.nn.functional import all_gather
from torch.nn.parallel import DistributedDataParallel

local_rank = 0

class DDPWrapper(DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    else:
        return dist.get_rank()

def set_local_rank(i):
    global local_rank
    local_rank = i

def get_local_rank():
    return local_rank

def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    else:
        return dist.get_world_size()

def collect(x):
    if not dist.is_available() or not dist.is_initialized():
        return x
    x = x.contiguous()
    out_list = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0) if get_rank() == 0 else None

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def cleanup_distributed():
    if get_world_size() > 1:
        print(f"[Rank {get_rank()}] Cleaning up distributed...")
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"Error during distributed cleanup: {e}")

def handle_signal(signum, frame):
    print(f"[Rank {os.environ.get('RANK', '?')}] Received signal {signum}. Initiating cleanup.")
    cleanup_distributed()
    sys.exit(0)

class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [
            torch.zeros_like(tensor) for _ in range(dist.get_world_size())
        ]

        dist.all_gather(gathered_tensor, tensor)
        return torch.cat(gathered_tensor, 0)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone().contiguous()
        dist.all_reduce(
            grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False
        )

        idx_from = dist.get_rank() * ctx.batch_size
        idx_to = (dist.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]

def all_gather_tensor(tensor: torch.Tensor):
    gathered = all_gather(tensor)

    gathered_tensor = torch.cat(gathered, dim=0)
    return gathered_tensor