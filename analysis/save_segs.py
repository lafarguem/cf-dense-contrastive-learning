import torch
from torch.utils.data import DataLoader
from hydra import compose, initialize
from transfer.builders import build_loader
from torch.backends import cudnn
from transfer.builders import (
    build_loader, build_scheduler, build_model, 
    build_optimizer, build_criterion
)
from pretraining.utils.common import recursive_to_device
from pretraining.utils.distributed import (set_local_rank, handle_signal, cleanup_distributed,
                                           get_local_rank)
from torch.amp import autocast, GradScaler
import os
import signal
import atexit
from pathlib import Path
from analysis import plot_segs

weights_list = [
    ('SimCLR','/vol/biomedic3/bglocker/mscproj/mal224/DCCL/outputs/full/simclr/2025-08-22_11-57-36/pretraining/weights/best.pt'),
    ('No pretraining',''),
    ('CheXmask','/vol/biomedic3/bglocker/mscproj/mal224/DCCL/outputs/transfer/chexmask/2025-08-15_10-30-04/weights/best.pt'),
    ('SSDCL','/vol/biomedic3/bglocker/mscproj/mal224/DCCL/outputs/full/supervised_nocf_dense_all_noflip_random_abl_data.train.augmentation.rotate_data.train.augmentation.crop/2025-08-14_17-59-37/pretraining/weights/best.pt'),
    ('VADeR','/vol/biomedic3/bglocker/mscproj/mal224/DCCL/outputs/full/unsupervised_nocf_dense_two_noflip_random_abl_data.train.augmentation.rotate/2025-08-17_14-57-33/pretraining/weights/best.pt'),
    ('DVD-CL','/vol/biomedic3/bglocker/mscproj/mal224/DCCL/outputs/full/unsupervised_cf_dense_two_noflip_random/2025-08-14_17-03-42/pretraining/weights/best.pt'),
    ('S-DVD-CL','/vol/biomedic3/bglocker/mscproj/mal224/DCCL/outputs/full/supervised_cf_dense_two_noflip_random/2025-08-21_12-38-44/pretraining/weights/best.pt'),
    ('MVD-CL','/vol/biomedic3/bglocker/mscproj/mal224/DCCL/outputs/full/unsupervised_cf_dense_all_noflip_random/2025-08-15_23-28-38/pretraining/weights/best.pt'),
    ('S-MVD-CL','/vol/biomedic3/bglocker/mscproj/mal224/DCCL/outputs/full/supervised_cf_dense_all_noflip_random/2025-08-14_16-59-09/pretraining/weights/best.pt')
]
output_directory = Path('/vol/biomedic3/bglocker/mscproj/mal224/DCCL/analysis/segmentations/')

def load_pretrained(model, pretrained_model):
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

def main(cfg):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.runtime.local_rank = local_rank
    set_local_rank(local_rank)

    cfg.runtime.output_dir = '/vol/biomedic3/bglocker/mscproj/mal224/DCCL/analysis/segmentations/'
    job_name = 'segmentation'
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
    

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGUSR1, handle_signal)

    
    atexit.register(cleanup_distributed)
    
    train_loader, eval_loader = build_loader(cfg)

    kfold = not isinstance(train_loader, DataLoader)
    if kfold:
        train_loader, eval_loader = train_loader[1], eval_loader[1]

    models = []

    for i,(name,weights) in enumerate(weights_list):
        print(f"Training {i+1}/{len(weights_list)} : {name}")
        model = training_setup_and_train(cfg, train_loader, weights)
        models.append((name,model))
    
    test_loader = DataLoader(eval_loader.dataset, batch_size=len(eval_loader.dataset))
    samples = next(iter(test_loader))
    print("Saving batch output")

    out = {}
    out['models'] = {}

    out['inputs'] = samples['x']
    out['labels'] = samples['mask']
    out['disease'] = samples['disease']
    inputs = samples['x'].to(cfg.runtime.device)
    for (name,model) in models:
        out['models'][name] = model(inputs)
    out['mean'] = cfg.data.eval.augmentation.normalize.mean
    out['std'] = cfg.data.eval.augmentation.normalize.std
    torch.save(out, output_directory / 'out.pt')
    plot_segs.main(out)

def training_setup_and_train(cfg, train_loader, weights):
    cfg.runtime.num_instances = len(train_loader.dataset)
    cfg.runtime.iter_per_epoch = len(train_loader)

    model = build_model(cfg)
    
    criterion = build_criterion(cfg)

    optimizer = build_optimizer(cfg, model)

    scheduler = build_scheduler(cfg, optimizer)

    scaler = GradScaler(enabled=cfg.train.amp)
    
    if weights != '':
        if cfg.runtime.distributed:
            load_pretrained(model.module, weights)
        else:
            load_pretrained(model, weights)

    try:
        return train(cfg, model, train_loader, criterion, optimizer, scheduler, scaler)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            cleanup_distributed()
        raise  

def train(
        cfg, 
        model, 
        train_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        scaler, 
    ):
    for epoch in range(cfg.runtime.start_epoch, cfg.train.epochs + 1):
        print(f'{epoch}/{cfg.train.epochs}, [0/{len(train_loader)}]')
        if cfg.runtime.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_epoch(epoch, train_loader, model, criterion, optimizer, scheduler, cfg, scaler)
    return model

def train_epoch(
        epoch, 
        train_loader,
        model, 
        criterion, 
        optimizer, 
        scheduler, 
        cfg, 
        scaler = None, 
    ):
    model.train()
    train_len = len(train_loader)
    for idx, data in enumerate(train_loader):
        step = (epoch - 1) * train_len + idx
        optimizer.zero_grad()
        data = recursive_to_device(data, device=cfg.runtime.device)
        labels = data['mask'].squeeze(1)

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

if __name__ == '__main__':
    with initialize(config_path="../transfer/configs", version_base="1.2"):  
        cfg = compose(config_name="config",  
                      overrides=[
                          '+experiment=default',
                      ])
    main(cfg)