import time
from shutil import copyfile
import torch
from torch.backends import cudnn
from transfer.builders import (
    build_loader, setup_logger, build_scheduler, build_model, 
    build_optimizer, build_evaluator, build_criterion
)
from pretraining.utils.common import AverageMeter, recursive_to_device, add_prefix_suffix, shorten_exp_name
from pretraining.utils.distributed import (is_main_process, get_rank, set_local_rank, handle_signal, cleanup_distributed,
                                           get_local_rank, reduce_tensor)
from pretraining.utils.debug import log_memory
from torch.amp import autocast, GradScaler
import hydra
import tracemalloc
import os
import signal
import sys
import torch
import atexit
import wandb
from transfer.evaluation.segmentation import SegmentationMetricsEvaluator
from pretraining.utils import run_manager
import json
from torch.utils.data import DataLoader
import numpy as np

def prepare_output(output_dir):
    os.mkdir(os.path.join(output_dir, 'checkpoints'))
    os.mkdir(os.path.join(output_dir, 'debug'))
    os.mkdir(os.path.join(output_dir, 'weights'))

def load_pretrained(model, pretrained_model, logger):
    if not os.path.exists(pretrained_model):
        head = os.path.dirname(pretrained_model)
        pretrained_model = os.path.join(head, 'last.pt')
    ckpt = torch.load(pretrained_model, map_location='cpu')
    encoder_dict = ckpt['encoder']
    decoder_dict = ckpt.get('decoder', None)
    model_encoder = model.encoder.state_dict()
    model_encoder.update(encoder_dict)
    model.encoder.load_state_dict(model_encoder)
    if decoder_dict is not None:
        model_decoder = model.decoder.state_dict()
        model_decoder.update(decoder_dict)
        model.decoder.load_state_dict(model_decoder)
    logger.info(f"==> loaded weights '{pretrained_model}'")

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
    copyfile(file_name, os.path.join(ckpt_dir, 'last.pt'))

    weights_dir = os.path.join(cfg.runtime.output_dir, 'weights')
    pretrained_model = {
        'encoder': model.encoder.state_dict(),
        'decoder': model.decoder.state_dict(),
    }
    weights_file = os.path.join(weights_dir, f'weights_{epoch}.pt')
    torch.save(pretrained_model, weights_file)

    copyfile(weights_file, os.path.join(weights_dir, 'last.pt'))

    if is_best or not os.path.exists(os.path.join(ckpt_dir, 'best.pt')):
        copyfile(file_name, os.path.join(ckpt_dir, 'best.pt'))
        copyfile(weights_file, os.path.join(weights_dir, 'best.pt'))
        logger.info(f"==> Updated best.pt to epoch {epoch}")

@hydra.main(config_path="configs", config_name="config.yaml", version_base="1.2")
def main(cfg):
    run_manager.set_file('/vol/biomedic3/bglocker/mscproj/mal224/DCCL/outputs/transfer_runs.csv')

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
    
    train_loader, eval_loader = build_loader(cfg, logger)

    kfold = not isinstance(train_loader, DataLoader)
    
    wandb.login(key='b79184b449074164894b39984f2a5a4fa5bea362')
    wandb_name = shorten_exp_name(cfg.train.exp_name)

    if not kfold:
        if is_main_process():
            run = wandb.init(
                project=cfg.train.project_name,
                name=f'transfer/{wandb_name}/{timestamp}',
                dir=os.path.join(output_dir),
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
                    with open(cfg.runtime.pipeline_metada, "w") as f:
                        json.dump(data, f, indent=2)
        else:
            run = None
        best_metrics, last_metrics = training_setup_and_train(cfg, train_loader, eval_loader, logger, run)
        run_manager.update_run(best_metrics)
        run_manager.update_run(last_metrics)
        run_manager.consume_run()
    
    else:
        total_best_metrics = {}
        total_last_metrics = {}
        for fold_num in range(len(train_loader)):
            logger.info(f"Training fold {fold_num + 1}")
            t_loader = train_loader[fold_num]
            e_loader = eval_loader[fold_num]

            if is_main_process():
                run = wandb.init(
                    project=cfg.train.project_name,
                    name=f'transfer/{wandb_name}/{timestamp}/fold_{fold_num+1}',
                    dir=os.path.join(output_dir),
                    mode="online" if not cfg.train.debug.wandb_offline else "offline",
                    group=f"{wandb_name}/kfold",
                    config={"fold": fold_num+1},
                    job_type='kfold',
                )

                if not cfg.train.debug.wandb_offline:
                    with open(os.path.join(output_dir, "wandb_url.txt"), "a") as f:
                        f.write(f"Fold {fold_num+1}: {run.url}\n")

                    if cfg.runtime.pipeline_metadata is not None:
                        with open(cfg.runtime.pipeline_metadata, "r") as f:
                            data = json.load(f)

                        if "pretraining" not in data:
                            data["pretraining"] = {}
                        if "wandb_urls" not in data["pretraining"]:
                            data["pretraining"]["wandb_urls"] = []

                        data["pretraining"]["wandb_urls"].append({
                            "fold": fold_num,
                            "url": run.url
                        })

                        with open(cfg.runtime.pipeline_metadata, "w") as f:
                            json.dump(data, f, indent=2)
            else:
                run = None

            best_metrics, last_metrics = training_setup_and_train(cfg, t_loader, e_loader, logger, run)
            for key, val in best_metrics.items():
                if key in total_best_metrics:
                    total_best_metrics[key].append(val)
                else:
                    total_best_metrics[key] = [val]
            for key, val in last_metrics.items():
                if key in total_last_metrics:
                    total_last_metrics[key].append(val)
                else:
                    total_last_metrics[key] = [val]
        
        best_metrics_mean = {}
        best_metrics_std = {}
        
        for key,val in total_best_metrics.items():
            best_metrics_mean[key] = np.mean(val)
            best_metrics_std[f"{key}_std"] = np.std(val)
        
        last_metrics_mean = {}
        last_metrics_std = {}   
        for key,val in total_last_metrics.items():
            last_metrics_mean[key] = np.mean(val)
            last_metrics_std[f"{key}_std"] = np.std(val)
        
        run_manager.update_run(best_metrics_mean)
        run_manager.update_run(best_metrics_std)
        run_manager.update_run(last_metrics_mean)
        run_manager.update_run(last_metrics_std)
        run_manager.consume_run()

def training_setup_and_train(cfg, train_loader, eval_loader, logger, run):
    cfg.runtime.num_instances = len(train_loader.dataset)
    cfg.runtime.iter_per_epoch = len(train_loader)

    logger.info('building model')
    model = build_model(cfg)
    
    logger.info('building criterion')
    criterion = build_criterion(cfg)

    logger.info('building optimizer')
    optimizer = build_optimizer(cfg, model)

    logger.info('building scheduler')
    scheduler = build_scheduler(cfg, optimizer)

    logger.info(f'amp: {cfg.train.amp}')
    scaler = GradScaler(enabled=cfg.train.amp)

    if cfg.train.pretrained_model:
        logger.info('loading pretrained model')
        if cfg.runtime.distributed:
            load_pretrained(model.module, cfg.train.pretrained_model, logger)
        else:
            load_pretrained(model, cfg.train.pretrained_model, logger)
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
            load_checkpoint(cfg, model.module, optimizer, scheduler, logger, scaler=scaler, sampler=train_loader.sampler)
        else:
            load_checkpoint(cfg, model, optimizer, scheduler, logger, scaler=scaler, sampler=train_loader.sampler)

    if cfg.train.debug.ram:
        tracemalloc.start(25)

    logger.info('setting up evaluators')
    eval_manager = build_evaluator(cfg, eval_loader, logger)

    logger.info(f'nan debug mode enabled: {cfg.train.debug.nan}')

    run_manager.setup_run(cfg, run)

    if cfg.train.debug.gpu_memory:
        torch.cuda.memory._record_memory_history()
    try:
        return train(cfg, eval_manager, model, train_loader, eval_loader, 
            criterion, optimizer, scheduler,
            logger, scaler, run
        )
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
        logger, 
        scaler, 
        run
    ):
    if cfg.train.evaluate_at_start:
        eval_manager.log(0, 0, model=model, force_all=True)
    
    best_metric = float('inf')
    best_metrics = {}
    last_metrics = {}
    for epoch in range(cfg.runtime.start_epoch, cfg.train.epochs + 1):
        if cfg.runtime.distributed:
            train_loader.sampler.set_epoch(epoch)
        (
            train_loss, eval_loss, eval_metrics_all, eval_loss_nf, eval_metrics_nf, eval_loss_pe, eval_metrics_pe
        ) = train_epoch(epoch, train_loader, eval_loader, model, criterion, optimizer, scheduler, 
            cfg, logger, scaler)
        if is_main_process():
            force = epoch == cfg.train.epochs
            metrics = eval_manager.log(epoch, epoch*len(train_loader), model=model, force_all=force)
            metrics['train_loss'] = (train_loss, 1)
            if eval_loss is not None:
                metrics['eval_loss'] = (eval_loss, 1)
                metrics.update(eval_metrics_all)
                metrics['eval_loss_nf'] = (eval_loss_nf, 1)
                metrics.update(eval_metrics_nf)
                metrics['eval_loss_pe'] = (eval_loss_pe, 1)
                metrics.update(eval_metrics_pe)

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
    run.finish()
    if cfg.train.debug.gpu_memory:
        torch.cuda.memory._dump_snapshot(os.path.join(cfg.runtime.output_dir, 'debug', 'snapshot_final.pickle'))
    return best_metrics, last_metrics

def train_epoch(
        epoch, 
        train_loader,
        eval_loader,
        model, 
        criterion, 
        optimizer, 
        scheduler, 
        cfg, 
        logger, 
        scaler = None, 
        prev_snapshot=None
    ):
    model.train()

    seg_metrics_fn = SegmentationMetricsEvaluator(**cfg.train.metrics)
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
        labels = data['mask'].squeeze(1)
        if cfg.train.debug.gpu_time:
            gpu_start = time.time()
        with autocast(device_type=cfg.train.device, enabled=cfg.train.amp):
            out = model(data['x'])
            loss = criterion(out, labels)

        scaler.scale(loss).backward()
        if cfg.train.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        if not torch.isfinite(loss).all():
            torch.autograd.set_detect_anomaly(True)

        if cfg.runtime.distributed:
            loss_avg = reduce_tensor(loss.detach())
        else:
            loss_avg = loss.detach()
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
                f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})')
            if cfg.train.debug.gpu_time:
                logger.info(
                    f'GPU time {gpu_time.val:.3f}/{batch_time.val:.3f}s ({gpu_time.val:.3f}/{batch_time.val:.3f}s)     '
                    f'{100*gpu_proportion.val:.3f}% ({100*gpu_proportion.avg:.3f}%)'
                )

            if is_main_process():
                wandb.log({
                    'lr': lr,
                    'loss': loss_meter.val,
                    'epoch': epoch,
                    'step': step
                }, step=step)
                    
        batch_end = time.time()
        
    if epoch % cfg.train.eval_freq == 0:
        with torch.no_grad():
            logger.info('Evaluating model')

            eval_loss_meter_all = AverageMeter()
            seg_metrics_meters_all = {key[0]: AverageMeter() for key in seg_metrics_fn.metrics}

            eval_loss_meter_nf = AverageMeter()
            seg_metrics_meters_nf = {key[0]: AverageMeter() for key in seg_metrics_fn.metrics}

            eval_loss_meter_pe = AverageMeter()
            seg_metrics_meters_pe = {key[0]: AverageMeter() for key in seg_metrics_fn.metrics}

            for idx, data in enumerate(eval_loader):
                data = recursive_to_device(data, device=cfg.runtime.device)
                labels = data['mask'].squeeze(1)
                disease_flags = data['disease']

                with autocast(device_type=cfg.train.device, enabled=False):
                    out = model(data['x'])
                    eval_loss = criterion(out, labels)

                seg_metrics = seg_metrics_fn(out, labels)

                if cfg.runtime.distributed:
                    for key, val in seg_metrics.items():
                        seg_metrics[key] = (reduce_tensor(val[0].detach()), val[1])
                    eval_loss_avg = reduce_tensor(eval_loss.detach())
                else:
                    for key, val in seg_metrics.items():
                        seg_metrics[key] = (val[0].detach(), val[1])
                    eval_loss_avg = eval_loss.detach()

                batch_size = data['x'].size(0)

                eval_loss_meter_all.update(eval_loss_avg.item(), batch_size)
                for key, val in seg_metrics.items():
                    seg_metrics_meters_all[key].update(val[0].item(), batch_size)

                for disease_value, loss_meter, seg_metrics_meters in [
                    (0, eval_loss_meter_nf, seg_metrics_meters_nf),
                    (1, eval_loss_meter_pe, seg_metrics_meters_pe),
                ]:
                    mask_idx = (disease_flags == disease_value)
                    if mask_idx.any():
                        out_masked, labels_masked = out[mask_idx].contiguous(), labels[mask_idx].contiguous()
                        loss_masked = criterion(out_masked, labels_masked)
                        seg_metrics_masked = seg_metrics_fn(out_masked, labels_masked)

                        if cfg.runtime.distributed:
                            for key, val in seg_metrics_masked.items():
                                seg_metrics_masked[key] = (reduce_tensor(val[0].detach()), val[1])
                            loss_masked = reduce_tensor(loss_masked.detach())
                        else:
                            for key, val in seg_metrics_masked.items():
                                seg_metrics_masked[key] = (val[0].detach(), val[1])
                            loss_masked = loss_masked.detach()

                        loss_meter.update(loss_masked.item(), mask_idx.sum().item())
                        for key, val in seg_metrics_masked.items():
                            seg_metrics_meters[key].update(val[0].item(), mask_idx.sum().item())

            all_log_list = [f'Eval loss (all): {eval_loss_meter_all.avg:.3f}']
            for key, val in seg_metrics_meters_all.items():
                all_log_list.append(f'Eval {key} (all): {val.avg:.3f}')
            all_log_str = ' | '.join(all_log_list)

            nf_log_list = [f'Eval loss (nf): {eval_loss_meter_nf.avg:.3f}']
            for key, val in seg_metrics_meters_nf.items():
                nf_log_list.append(f'Eval {key} (nf): {val.avg:.3f}')
            nf_log_str = ' | '.join(nf_log_list)

            pe_log_list = [f'Eval loss (pe): {eval_loss_meter_pe.avg:.3f}']
            for key, val in seg_metrics_meters_pe.items():
                pe_log_list.append(f'Eval {key} (pe): {val.avg:.3f}')
            pe_log_str = ' | '.join(pe_log_list)

            logger.info(all_log_str)
            logger.info(nf_log_str)
            logger.info(pe_log_str)

            final_eval_loss = eval_loss_meter_all.avg
            final_metrics = {f"eval_{key.lower().replace(' ', '_')}": (val.avg, seg_metrics[key][1]) for (key,val) in seg_metrics_meters_all.items()}
            final_eval_loss_nf = eval_loss_meter_nf.avg
            final_metrics_nf = {f"eval_{key.lower().replace(' ', '_')}_nf": (val.avg, seg_metrics[key][1]) for (key,val) in seg_metrics_meters_nf.items()}
            final_eval_loss_pe = eval_loss_meter_pe.avg
            final_metrics_pe = {f"eval_{key.lower().replace(' ', '_')}_pe": (val.avg, seg_metrics[key][1]) for (key,val) in seg_metrics_meters_pe.items()}

            log_dict = {'eval_loss': final_eval_loss, 'eval_loss_nf': final_eval_loss_nf, 'eval_loss_pe': final_eval_loss_pe}
            log_dict.update(add_prefix_suffix(final_metrics, v_func=lambda x: x[0]))
            log_dict.update(add_prefix_suffix(final_metrics_nf, v_func=lambda x: x[0]))
            log_dict.update(add_prefix_suffix(final_metrics_pe, v_func=lambda x: x[0]))
            if is_main_process():
                wandb.log(log_dict, step=step)
    else:
        final_eval_loss = None
        final_metrics = None
        final_eval_loss_nf = None
        final_metrics_nf = None
        final_eval_loss_pe = None
        final_metrics_pe = None
                    
    return (
        loss_meter.val, final_eval_loss, final_metrics, 
        final_eval_loss_nf, final_metrics_nf, 
        final_eval_loss_pe, final_metrics_pe
    )

if __name__ == '__main__':
    import os

    output_dir = os.environ.get("OUTPUT_DIR")
    if output_dir is not None:
        import sys
        sys.argv.append(f"hydra.run.dir={output_dir}")
    main()