import os
import time
from shutil import copyfile

import torch
import torch.autograd
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel

from pretraining.builders import (
    build_loader, setup_logger, build_scheduler, build_model, 
    build_optimizer, build_evaluator, build_criterion, build_global_loss_weight_scheduler
)
from pretraining.utils.common import AverageMeter, recursive_to_device, shorten_exp_name
from pretraining.utils.distributed import (is_main_process, get_rank, set_local_rank, handle_signal, cleanup_distributed,
                                           get_local_rank, reduce_tensor)
from pretraining.models.contrastive.base_model import DualBranchContrastiveModel, SingleBranchContrastiveModel
from pretraining.utils.debug import log_memory
from pretraining.utils import run_manager

from torch import autocast, GradScaler
import hydra
import tracemalloc

import os
import signal
import sys
import torch
import atexit
import logging
import wandb
import json

def prepare_output(output_dir):
    os.mkdir(os.path.join(output_dir, 'checkpoints'))
    os.mkdir(os.path.join(output_dir, 'debug'))
    os.mkdir(os.path.join(output_dir, 'weights'))

def load_pretrained(model, pretrained_model, logger):
    ckpt = torch.load(pretrained_model, map_location='cpu')
    state_dict = ckpt['model']
    model_dict = model.state_dict()

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    logger.info(f"==> loaded checkpoint '{pretrained_model}' (epoch {ckpt['epoch']})")


def load_checkpoint(cfg, model, optimizer, scheduler, logger, scaler = None, sampler=None):
    logger.info(f"=> loading checkpoint '{cfg.train.resume}'")
    checkpoint = torch.load(cfg.train.resume, map_location='cpu')
    cfg.runtime.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler'])
    if sampler is not None:
        sampler.set_epoch(checkpoint['epoch'])
    logger.info(f"=> loaded successfully '{cfg.train.resume}' (epoch {checkpoint['epoch']})")

    del checkpoint
    torch.cuda.empty_cache()

def save_checkpoint(cfg, epoch, model, optimizer, scheduler, logger, scaler=None, sampler=None, is_best=False):
    logger.info('==> Saving...')
    state = {
        'cfg': cfg,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    if scaler is not None:
        state['scaler'] = scaler.state_dict()

    ckpt_dir = os.path.join(cfg.runtime.output_dir, 'checkpoints')
    file_name = os.path.join(ckpt_dir, f'ckpt_epoch_{epoch}.pt')
    torch.save(state, file_name)

    copyfile(file_name, "/" + os.path.join(*cfg.runtime.output_dir.split('/')[:-1], 'current.pt'))

    weights_dir = os.path.join(cfg.runtime.output_dir, 'weights')
    pretrained_model = model.get_pretrained_model()
    weights_file = os.path.join(weights_dir, f'weights_{epoch}.pt')
    torch.save(pretrained_model, weights_file)

    copyfile(file_name, os.path.join(ckpt_dir, 'last.pt'))
    copyfile(weights_file, os.path.join(weights_dir, 'last.pt'))

    if is_best or not os.path.exists(os.path.join(ckpt_dir, 'best.pt')):
        copyfile(file_name, os.path.join(ckpt_dir, 'best.pt'))
        copyfile(weights_file, os.path.join(weights_dir, 'best.pt'))
        logger.info(f"==> Updated best.pt to epoch {epoch}")

@hydra.main(config_path="configs", config_name="config.yaml", version_base="1.2")
def main(cfg):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.runtime.local_rank = local_rank
    set_local_rank(local_rank)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    cfg.runtime.output_dir = output_dir
    job_name = hydra.core.hydra_config.HydraConfig.get().job.name
    cfg.runtime.job_name = job_name

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    cfg.runtime.distributed = torch.distributed.is_available() and (world_size > 1)

    if cfg.runtime.distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        torch.cuda.set_device(0)
    cudnn.benchmark = True
    if cfg.train.device == 'cuda':
        cfg.runtime.device = f'cuda:{get_local_rank()}'
    else:
        cfg.runtime.device = cfg.train.device

    if is_main_process():
        prepare_output(output_dir)
    
    logger = setup_logger(output=os.path.join(output_dir,'txt-logs',f'{job_name}.log'), 
                          distributed_rank=get_rank(), name=cfg.train.exp_name)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGUSR1, handle_signal)

    atexit.register(cleanup_distributed)

    logger.info(os.getcwd())
    logger.info(f"Output directory: {output_dir}")

    timestamp = os.path.basename(os.path.normpath(output_dir))

    if is_main_process():
        wandb_name = shorten_exp_name(cfg.train.exp_name)
        wandb.login(key='b79184b449074164894b39984f2a5a4fa5bea362')
        run = wandb.init(
            project=cfg.train.project_name,
            name=f'pretraining/{wandb_name}/{timestamp}',
            dir=output_dir,
            mode="online" if not cfg.train.debug.wandb_offline else "offline",
            group=wandb_name,
        )
        if not cfg.train.debug.wandb_offline:
            with open(os.path.join(output_dir,"wandb_url.txt"), "w") as f:
                f.write(run.url)
            if cfg.runtime.pipeline_metadata is not None:
                with open(cfg.runtime.pipeline_metadata, "r") as f:
                    data = json.load(f)
                data["pretraining"]["wandb_url"] = run.url
                with open(cfg.runtime.pipeline_metadata, "w") as f:
                    json.dump(data, f, indent=2)
    else:
        run = None

    train_loader, eval_loader = build_loader(cfg, logger)

    cfg.runtime.num_instances = len(train_loader.dataset)
    cfg.runtime.iter_per_epoch = len(train_loader)

    logger.info('building model')
    model = build_model(cfg)
    if isinstance(model, DistributedDataParallel):
        real_model = model.module
    else:
        real_model = model

    if isinstance(real_model, SingleBranchContrastiveModel):
        dual = False
        logger.info('Model is single branch')
    elif isinstance(real_model, DualBranchContrastiveModel):
        dual = True
        logger.info('Model is dual branch')
    else:
        raise TypeError("model should be of type SingleBranchContrastiveModel or DualBranchContrastiveModel")
    
    logger.info('building criterion')
    criterion = build_criterion(cfg, dual)

    logger.info('building optimizer')
    optimizer = build_optimizer(cfg, model)

    logger.info('building scheduler')
    scheduler = build_scheduler(cfg, optimizer)

    logger.info('building global loss scheduler')
    global_loss_weight_scheduler = build_global_loss_weight_scheduler(cfg, model, criterion, len(train_loader))

    logger.info(f'amp: {cfg.train.amp}')
    scaler = GradScaler(enabled=(cfg.train.amp == 'fp16' or cfg.train.amp is True))

    if cfg.train.pretrained_model:
        logger.info('loading pretrained model')
        assert os.path.isfile(cfg.train.pretrained_model)
        if cfg.runtime.distributed:
            load_pretrained(model.module, cfg.train.pretrained_model)
        else:
            load_pretrained(model, cfg.train.pretrained_model)
    if cfg.train.auto_resume:
        resume_file = "/" + os.path.join(*cfg.runtime.output_dir.split('/')[:-1], 'current.pt')
        if os.path.exists(resume_file):
            logger.info(f'auto resume from {resume_file}')
            cfg.train.resume = resume_file
        else:
            logger.info(f'no checkpoint found in {output_dir.split('/')[0]}, ignoring auto resume')
    if cfg.train.resume:
        assert os.path.isfile(cfg.train.resume)
        logger.info('loading checkpoint')
        if cfg.train.distributed:
            load_checkpoint(cfg, model.module, optimizer, scheduler, scaler=scaler, sampler=train_loader.sampler)
        else:
            load_checkpoint(cfg, model, optimizer, scheduler, scaler=scaler, sampler=train_loader.sampler)

    if cfg.train.debug.ram:
        tracemalloc.start(25)

    logger.info('setting up evaluators')
    eval_manager = build_evaluator(cfg, eval_loader, logger)
    torch.autograd.set_detect_anomaly(cfg.train.debug.nan)
    logger.info(f'nan debug mode enabled: {cfg.train.debug.nan}')
    run_manager.setup_run(cfg, run)
    if cfg.train.debug.gpu_memory:
        torch.cuda.memory._record_memory_history()
    try:
        train(cfg, eval_manager, model, train_loader, eval_loader, 
            criterion, optimizer, scheduler, global_loss_weight_scheduler,
            logger, scaler, run)
    except RuntimeError as e:
        if cfg.train.debug.gpu_memory:
            torch.cuda.memory._dump_snapshot(os.path.join(output_dir, 'debug', 'snapshot_final.pickle'))
        if "CUDA out of memory" in str(e):
            logger.error("Caught CUDA OOM. Cleaning up and exiting.")
            torch.cuda.empty_cache()
            cleanup_distributed()
        raise  

def train(
        cfg, 
        eval_manager, 
        model, 
        train_loader, 
        eval_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        global_loss_weight_scheduler, 
        logger, 
        scaler, 
        run
    ):
    if cfg.train.evaluate_at_start:
        eval_manager.log(0, 0, encoder=model, force_all=True)

    best_metric = float('inf')
    best_metrics = {}
    last_metrics = {}
    for epoch in range(cfg.runtime.start_epoch, cfg.train.epochs + 1):
        if cfg.runtime.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_loss, eval_loss = train_epoch(epoch, train_loader, eval_loader, model, criterion, optimizer, scheduler, 
            global_loss_weight_scheduler, cfg, logger, scaler)
        force = epoch == cfg.train.epochs
        metrics = eval_manager.log(epoch, epoch*len(train_loader), encoder=model, force_all=force)
        if is_main_process():
            metrics['train_loss'] = (train_loss, 1)
            metrics['eval_loss'] = (eval_loss, 1)

            for key, val in metrics.items():
                if (val[0]*val[1]) < (best_metrics.get(f'best_{key}', val[1]*float('inf'))*val[1]):
                    best_metrics[f'best_{key}'] = val[0]
                last_metrics[f'last_{key}'] = val[0]

            current_metric = metrics.get(cfg.train.metric_key)
            if current_metric is not None:
                is_better = current_metric[0]*current_metric[1] < best_metric
                if is_better:
                    best_metric = current_metric[0]*current_metric[1]
            else:
                is_better = False
            if epoch % cfg.train.save_freq == 0 or force:
                if cfg.runtime.distributed:
                    save_checkpoint(cfg, epoch, model.module, optimizer, scheduler, logger, sampler=train_loader.sampler, scaler=scaler, is_best=is_better)
                else:
                    save_checkpoint(cfg, epoch, model, optimizer, scheduler, logger, sampler=train_loader.sampler, scaler=scaler, is_best=is_better)
        if cfg.train.debug.ram:
            if epoch == 2:
                snapshot1 = tracemalloc.take_snapshot()
                torch.cuda.memory._dump_snapshot(os.path.join(cfg.runtime.output_dir, 'debug', f'snapshot_{epoch}.pickle'))
            elif epoch in [5,10,15,20]:
                snapshot2 = tracemalloc.take_snapshot()
                log_memory(logger, snapshot1, snapshot2, epoch, num_lines=10)
                snapshot1 = snapshot2
                torch.cuda.memory._dump_snapshot(os.path.join(cfg.runtime.output_dir, 'debug', f'snapshot_{epoch}.pickle'))
    if cfg.train.debug.ram:
        tracemalloc.stop()
    if is_main_process():
        run.finish()
    run_manager.update_run(best_metrics)
    run_manager.update_run(last_metrics)
    run_manager.consume_run()
    if cfg.train.debug.gpu_memory:
        torch.cuda.memory._dump_snapshot(os.path.join(cfg.runtime.output_dir, 'debug', 'snapshot_final.pickle'))

def train_epoch(
        epoch, 
        train_loader,
        eval_loader,
        model, 
        criterion, 
        optimizer, 
        scheduler, 
        global_loss_weight_scheduler,
        cfg, 
        logger, 
        scaler = None, 
        prev_snapshot=None
    ):
    if cfg.train.amp is not False:
        if cfg.train.amp == 'bf16':
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
        cfg.runtime.amp = True
    
    model.train()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    if cfg.train.debug.gpu_time:
        gpu_time = AverageMeter()
        gpu_proportion = AverageMeter()

    batch_end = time.time()
    train_len = len(train_loader)
    for idx, data in enumerate(train_loader):
        step = (epoch - 1) * train_len + idx
        optimizer.zero_grad()
        data = recursive_to_device(data, device=cfg.runtime.device)
        if cfg.train.debug.gpu_time:
            gpu_start = time.time()
        with autocast(device_type=cfg.train.device, enabled=cfg.runtime.amp, dtype=amp_dtype):
            out = model(data)
            loss, dense_loss, global_loss = criterion(out, data, return_sub_losses=True)

        scaler.scale(loss).backward()
        if cfg.train.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        global_loss_weight_scheduler.step()

        if not torch.isfinite(loss).all():
            torch.autograd.set_detect_anomaly(True)

        if cfg.runtime.distributed:
            loss_avg = reduce_tensor(loss.detach())
            if dense_loss is not None:
                dense_loss_avg = reduce_tensor(dense_loss.detach())
            if global_loss is not None:
                global_loss_avg = reduce_tensor(global_loss.detach())
        else:
            loss_avg = loss.detach()
            if dense_loss is not None:
                dense_loss_avg = dense_loss.detach()
            if global_loss is not None:
                global_loss_avg = global_loss.detach()
        
        loss_meter.update(loss_avg.item(), data['x'][0].size(0))
        _batch_time = time.time() - batch_end
        batch_time.update(_batch_time)
        if cfg.train.debug.gpu_time:
            _gpu_time = time.time() - gpu_start
            gpu_time.update(_gpu_time)
            gpu_proportion.update(_gpu_time/_batch_time)

        if idx % cfg.train.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Train: [{epoch}/{cfg.train.epochs}][{idx}/{train_len}]  '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                f'lr {lr:.4f}  '
                f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})'
            )
            if cfg.train.debug.gpu_time:
                logger.info(
                    f'GPU time {gpu_time.val:.3f}/{batch_time.val:.3f}s ({gpu_time.val:.3f}/{batch_time.val:.3f}s)     '
                    f'{100*gpu_proportion.val:.3f}% ({100*gpu_proportion.avg:.3f}%)'
                )
            if is_main_process():
                log_dict = {
                    'lr': lr,
                    'global_loss_weight': global_loss_weight_scheduler.global_loss_weight,
                    'loss': loss_meter.val,
                    'epoch': epoch,
                    'step': step
                }
                if dense_loss is not None:
                    log_dict['dense_loss'] = dense_loss_avg
                if global_loss is not None:
                    log_dict['global_loss'] = global_loss_avg
                wandb.log(log_dict, step=step)
                    
        batch_end = time.time()
    
    if epoch % cfg.train.eval_freq == 0:
        with torch.no_grad():
            logger.info('Evaluating model')
            eval_loss_meter = AverageMeter()
            for idx, data in enumerate(eval_loader):
                data = recursive_to_device(data, device=cfg.runtime.device)
                with autocast(device_type=cfg.train.device, enabled=cfg.runtime.amp, dtype=amp_dtype):
                    out = model(data)
                    eval_loss, eval_dense_loss, eval_global_loss = criterion(out, data, return_sub_losses=True)
                if cfg.runtime.distributed:
                    eval_loss_avg = reduce_tensor(eval_loss.detach())
                    if eval_dense_loss is not None:
                        eval_dense_loss_avg = reduce_tensor(eval_dense_loss.detach())
                    if eval_global_loss is not None:
                        eval_global_loss_avg = reduce_tensor(eval_global_loss.detach())
                else:
                    eval_loss_avg = loss.detach()
                    if eval_dense_loss is not None:
                        eval_dense_loss_avg = eval_dense_loss.detach()
                    if eval_global_loss is not None:
                        eval_global_loss_avg = eval_global_loss.detach()
                    
                eval_loss_meter.update(eval_loss_avg.item(), data['x'][0].size(0))

            logger.info(f'Val loss {eval_loss_meter.avg:.3f}')
            final_eval_loss = eval_loss_meter.avg
            if is_main_process():
                eval_log_dict = {'eval_loss': final_eval_loss}
                if eval_dense_loss is not None:
                    eval_log_dict['eval_dense_loss'] = eval_dense_loss_avg
                if eval_global_loss is not None:
                    eval_log_dict['eval_global_loss'] = eval_global_loss_avg
                wandb.log(eval_log_dict, step=step)
    else:
        final_eval_loss = None
                    
    return loss_meter.val, final_eval_loss

if __name__ == '__main__':
    import os
    output_dir = os.environ.get("OUTPUT_DIR")
    if output_dir is not None:
        import sys
        sys.argv.append(f"hydra.run.dir={output_dir}")
    main()
