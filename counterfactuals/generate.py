from pretraining.builders.loader_builder import build_loader
from torch.utils.data import DataLoader
import yaml
import torch
import copy
import argparse
from counterfactuals.hvae import HVAE2
import os
from torchvision.utils import save_image
from counterfactuals.train_setup import setup_dataloaders
from counterfactuals.trainer import preprocess_batch

SUFFIX_MAPPING = {'pe': 'disease', 'sc': 'scanner'}

@torch.no_grad
def counterfactual_inference(input_res, parents_x, cf_suffixes, model, batch, device, u_t = 1.0):
    zs = model.abduct(x=batch["x"], parents=batch["pa"])
    if model.cond_prior:
        zs = [zs[j]["z"] for j in range(len(zs))]
    pa = {k: batch[k] for k in parents_x}
    _pa = torch.cat([batch[k] for k in parents_x], dim=1)
    _pa = (
        _pa[..., None, None]
        .repeat(1, 1, *(input_res,) * 2)
        .to(device)
        .float()
    )
    rec_loc, rec_scale = model.forward_latents(zs, parents=_pa)
    cfs = {}
    for suffix in cf_suffixes:
        operations = [SUFFIX_MAPPING[s] for s in suffix.split('_')]

        cf_pa = copy.deepcopy(pa)
        cf_pa = {k: batch[k] for k in parents_x}
        do = {}
        for p in operations:
            cf_pa[p] = 1 - cf_pa[p]
            do[p] = cf_pa[p]
        _cf_pa = torch.cat([cf_pa[k] for k in parents_x], dim=1)
        _cf_pa = (
            _cf_pa[..., None, None]
            .repeat(1, 1, *(input_res,) * 2)
            .to(device)
            .float()
        )
        cf_loc, cf_scale = model.forward_latents(zs, parents=_cf_pa)

        u = (batch["x"] - rec_loc) / rec_scale.clamp(min=1e-12)

        cf_scale = cf_scale * u_t
        cf_x = torch.clamp(cf_loc + cf_scale * u, min=-1, max=1)
        cf_x = (cf_x + 1) / 2.0
        cfs[suffix] = cf_x
    return cfs

def main(args):
    loader = setup_dataloaders(args, all=True)['train']
    dataset = loader.dataset
    dataloader = DataLoader(dataset, args.batch_size)
    print("dataset", len(dataset))
    model = HVAE2(args)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)
    model.to(args.device)
    model.eval()
    
    if args.resume:
        if os.path.isfile(args.resume):
            ckpt = torch.load(args.resume)
            model.load_state_dict(ckpt["model_state_dict"])
            for param in model.parameters():
                param.requires_grad = False
        else:
            if os.path.isfile(args.resume):
                ckpt = torch.load(args.resume)
                model.load_state_dict(ckpt["model_state_dict"])
                for param in model.parameters():
                    param.requires_grad = False
    for i,batch in enumerate(dataloader):
        print(f"{i}/{len(dataloader)}")
        batch = preprocess_batch(args, batch, args.expand_pa)
        if not args.dry_run:
            cfs = counterfactual_inference(args.input_res, args.parents_x, args.cf_suffixes, model, batch, args.device)
        for j in range(len(batch['shortpath'])):
            if not args.dry_run:
                for key, im in cfs.items():
                    path = os.path.join(args.cf_dir, f"{batch['shortpath'][j][:-4]}_{key}.png")
                    save_image(im[j], path)
            else:
                for key in args.cf_suffixes:
                    path = os.path.join(args.cf_dir, f"{batch['shortpath'][j][:-4]}_{key}.png")
                    if not os.path.exists(path):
                        print(path)


if __name__ == "__main__":
    from counterfactuals.hps import add_arguments, setup_hparams
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    parser.add_argument('--cf_dir', help='Path to save counterfactuals', type=str, default="/vol/biomedic3/bglocker/mscproj/mal224/data/counterfactuals")
    parser.add_argument('--batch_size', help='Batch size', type=int, default=16)
    parser.add_argument('--dry_run', help='Check existence of counterfactuals', action='store_true')

    args = setup_hparams(parser)
    args.expand_pa = True
    args.cf_suffixes = ['sc', 'pe', 'sc_pe']
    
    if not os.path.exists(args.cf_dir):
        os.makedirs(args.cf_dir, exist_ok=True)

    main(args)