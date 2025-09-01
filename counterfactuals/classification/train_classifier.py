import torch
import torch.nn as nn
from torchvision import models
from tqdm.auto import tqdm

import argparse
import os
import traceback

import send2trash
import torch

import os

from counterfactuals.train_setup import (
    setup_dataloaders,
    setup_directories,
    setup_logging,
    setup_optimizer,
    setup_tensorboard,
)
from counterfactuals.utils import seed_all
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

def save_checkpoint(state, output_dir, filename="checkpoint.pt"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    torch.save(state, path)

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

    model = ResNet50MultiLabel(num_labels=3, in_channels=1)

    args.save_dir = setup_directories(args)

    optimizer = optim.AdamW(model.parameters())

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(args.device)
    torch.cuda.set_device(args.device)
    model.to(args.device)

    if args.resume:
        if os.path.isfile(args.resume):
            args.start_epoch = ckpt["epoch"]
            args.iter = ckpt["step"]
            args.best_loss = ckpt["best_loss"]
            model.load_state_dict(ckpt["model_state_dict"])
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

    try:
        train(args, model, dataloaders['train'], dataloaders['valid'], optimizer)
    except:
        print(traceback.format_exc())
        if input("Training interrupted, keep logs? [Y/n]: ") == "n":
            if input(f"Send '{args.save_dir}' to Trash? [y/N]: ") == "y":
                send2trash.send2trash(args.save_dir)
                print("Done.")


class ResNet50MultiLabel(nn.Module):
    def __init__(self, num_labels=3, in_channels=3, dropout_prob=0.5):
        super(ResNet50MultiLabel, self).__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        if in_channels != 3:
            old_conv = self.base_model.conv1
            self.base_model.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            
            with torch.no_grad():
                if in_channels < 3:
                    self.base_model.conv1.weight[:, :in_channels, :, :] = old_conv.weight[:, :in_channels, :, :]
                else:
                    self.base_model.conv1.weight[:, :3, :, :] = old_conv.weight

        in_features = self.base_model.fc.in_features
        if dropout_prob > 0.:
            self.dropout = nn.Dropout(p=dropout_prob)
            self.base_model.fc = nn.Sequential(
                self.dropout,
                nn.Linear(in_features, num_labels)
            )
        else:
            self.base_model.fc = nn.Linear(in_features, num_labels)
    def forward(self, x):
        return self.base_model(x)

def train(args, model, train_loader, val_loader, optimizer):
    model = model.to(args.device)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Training", mininterval=120)
        for sample in progress_bar:
            images = sample['x'].to(args.device)
            targets = torch.stack([sample[pa].float() for pa in args.parents_x], dim=1).to(args.device).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            args.iter += 1
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Avg Train Loss = {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for sample in tqdm(val_loader, desc="Validation", leave=False):
                images = sample['x'].to(args.device)
                targets = torch.stack([sample[pa].float() for pa in args.parents_x], dim=1).to(args.device).float()

                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).int()

                all_preds.append(preds.cpu())
                all_targets.append(targets.int().cpu())

        avg_val_loss = val_loss / len(val_loader)

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        f1 = f1_score(all_targets, all_preds, average='macro')
        acc = accuracy_score(all_targets, all_preds)

        print(f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}, F1 = {f1:.4f}, Accuracy = {acc:.4f}")

        is_best = avg_val_loss < args.best_loss
        if is_best:
            args.best_loss = avg_val_loss

        checkpoint = {
            'epoch': epoch + 1,
            'step': args.iter,
            'best_loss': args.best_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'hparams': vars(args),
        }

        save_checkpoint(checkpoint, args.save_dir, filename="last.pt")
        if is_best:
            save_checkpoint(checkpoint, args.save_dir, filename="best.pt")

    print("Training complete!")


if __name__ == "__main__":
    from counterfactuals.hps import add_arguments, setup_hparams

    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = setup_hparams(parser)
    main(args)