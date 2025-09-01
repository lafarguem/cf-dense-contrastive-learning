import argparse
import os
import traceback

import send2trash
import torch

import sys
import os

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(current_dir)

from counterfactuals.train_setup import (
    setup_dataloaders,
    setup_directories,
    setup_logging,
    setup_optimizer,
    setup_tensorboard,
)
from counterfactuals.trainer import trainer
from counterfactuals.utils import EMA, seed_all
from counterfactuals.hvae import HVAE2
from counterfactuals.classification.train_classifier import ResNet50MultiLabel

def load_classifier(args):
    if args.classifier_ckpt is None:
        return None
    if os.path.isfile(args.classifier_ckpt):
        print(f"\nLoading checkpoint: {args.classifier_ckpt}")
        model = ResNet50MultiLabel(num_labels=3, in_channels=1, dropout_prob=0.)
        ckpt = torch.load(args.classifier_ckpt, map_location='cpu')  
        model.load_state_dict(ckpt["model_state_dict"])
        for param in model.parameters():
            param.requires_grad = False
        return model.to(args.device)
    else:
        raise FileNotFoundError(f"Classifier checkpoint not found at: {args.classifier_ckpt}")

def main(args):
    seed_all(args.seed, args.deterministic)
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\nLoading checkpoint: {args.resume}")
            ckpt = torch.load(args.resume)
            ckpt_args = {k: v for k, v in ckpt["hparams"].items() if k != "resume"}
            ckpt_args["eval_freq"] = args.eval_freq
            if args.data_dir is not None:
                ckpt_args["data_dir"] = args.data_dir
            if args.lr < ckpt_args["lr"]:
                ckpt_args["lr"] = args.lr
            if args.epochs > ckpt_args["epochs"]:
                ckpt_args["epochs"] = args.epochs
            vars(args).update(ckpt_args)
        else:
            print(f"Checkpoint not found at: {args.resume}")

    dataloaders = setup_dataloaders(args)

    model = HVAE2(args)

    def init_bias(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.zeros_(m.bias)

    model.apply(init_bias)
    ema = EMA(model, beta=args.ema_rate)
    
    assert args.exp_name != "", "No experiment name given."
    args.save_dir = setup_directories(args)
    writer = setup_tensorboard(args, model)
    logger = setup_logging(args)

    optimizer, scheduler = setup_optimizer(args, model)

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)
    model.to(args.device)
    ema.to(args.device)

    
    if args.resume:
        if os.path.isfile(args.resume):
            args.start_epoch = ckpt["epoch"]
            args.iter = ckpt["step"]
            args.best_loss = ckpt["best_loss"]
            model.load_state_dict(ckpt["model_state_dict"])
            ema.ema_model.load_state_dict(ckpt["ema_model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for g in optimizer.param_groups:
                g["lr"] = args.lr
                g["initial_lr"] = args.lr  
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda x: x * 0 + 1
            )
        else:
            print("Checkpoint not found at: {}".format(args.resume))
    else:
        args.start_epoch, args.iter = 0, 0
        args.best_loss = float("inf")

    eval_classifier = load_classifier(args)
    
    try:
        trainer(args, model, ema, dataloaders, optimizer, scheduler, writer, logger, eval_classifier)
    except:  
        print(traceback.format_exc())
        if input("Training interrupted, keep logs? [Y/n]: ") == "n":
            if input(f"Send '{args.save_dir}' to Trash? [y/N]: ") == "y":
                send2trash.send2trash(args.save_dir)
                print("Done.")


if __name__ == "__main__":
    from counterfactuals.hps import add_arguments, setup_hparams

    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = setup_hparams(parser)
    main(args)