import torch
from torch.utils.data import DataLoader
from pretraining.evaluation.base_evaluation import BaseEvaluator
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pretraining.utils.common import recursive_to_device, reduce_segmentation_mask, add_prefix_suffix
from torch.utils.data.distributed import DistributedSampler
from pretraining.utils.distributed import get_world_size, get_local_rank, is_main_process
from pretraining.datasets.base_dataset import TransferDataset, CustomAugmentationDataset
from transfer.evaluation.visualizations import VisualizationEvaluator
import wandb
import os
import gc
from torch.nn.parallel import DistributedDataParallel
from transfer.evaluation.segmentation import SegmentationMetricsEvaluator

class LinearProbeEvaluator(BaseEvaluator):
    def __init__(
            self, 
            eval_name,
            eval_freq,
            num_classes, 
            dataloader,
            optimizer, 
            probe, 
            criterion, 
            in_channels, 
            epochs,
            mean,
            std,
            dataset=None,
            train_augmentation=None,
            eval_augmentation=None,
            train_prop=0.9,
            device='cuda'
        ):
        super().__init__(eval_name, eval_freq)
        self.num_classes = num_classes
        self.train_dataset, self.val_dataset = TransferDataset(dataset).split(train_prop=train_prop)
        if train_augmentation is not None:
            train_augmentation = instantiate(train_augmentation)
            self.train_dataset = CustomAugmentationDataset(self.train_dataset, train_augmentation)
        if eval_augmentation is not None:
            eval_augmentation = instantiate(eval_augmentation)
            self.val_dataset = CustomAugmentationDataset(self.val_dataset, eval_augmentation)
        self.loader_kwargs = OmegaConf.to_container(dataloader, resolve=True)
        if device == 'cuda':
            self.device = f'cuda:{get_local_rank()}'
        else:
            self.device = device
        self.probe_cfg = probe
        self.optimizer_cfg = optimizer
        self.criterion_cfg = criterion
        self.epochs = epochs
        self.in_channels = in_channels
        self.visualizer = VisualizationEvaluator('linear_probe_visualizer', 1, mean, std, directory='linear_probe')
        if get_world_size() > 1:
            sampler = DistributedSampler(self.train_dataset)
            val_sampler = DistributedSampler(self.val_dataset)
        else:
            sampler = None
            val_sampler = None
        self.train_loader = DataLoader(self.train_dataset, sampler=sampler, **self.loader_kwargs)
        self.val_loader = DataLoader(self.val_dataset, sampler=val_sampler, **self.loader_kwargs)
        self.metrics_fn = SegmentationMetricsEvaluator()

    def evaluate(self, step=None, encoder=None, embeddings=None, labels=None, shape=None, logger=None):
        if get_world_size() > 1:
            torch.distributed.barrier()
        probe = instantiate(self.probe_cfg, in_channels=self.in_channels, out_channels=self.num_classes).to(self.device)
        optimizer = instantiate(self.optimizer_cfg, params=probe.parameters())
        criterion = instantiate(self.criterion_cfg)

        if get_world_size() > 1:
            probe = DistributedDataParallel(probe, [get_local_rank()], broadcast_buffers=False)

        last_loss, avg_metrics = self.train_probe(step, encoder, probe, optimizer, criterion, logger)
        del probe, optimizer, criterion
        torch.cuda.empty_cache()
        gc.collect()

        return last_loss, avg_metrics

    def train_probe(self, step, encoder, probe, optimizer, criterion, logger):
        encoder.eval()
        self.visualizer.directory = f'linear_probe/step-{step}'
        os.makedirs(os.path.join(self.output_dir, self.visualizer.directory), exist_ok=True)
        self.visualizer.output_dir = self.output_dir
        avg_loss = None
        best_loss = torch.inf

        step = 0

        probe.train()
        for epoch in range(self.epochs):
            probe.train()

            total_loss = 0

            if get_world_size() > 1:
                self.train_loader.sampler.set_epoch(epoch)
                self.val_loader.sampler.set_epoch(epoch)

            for sample in self.train_loader:
                sample = recursive_to_device(sample, self.device)
                
                with torch.no_grad():
                    features = encoder.evaluate(sample)
                step += 1
                logits = probe(features)  

                B, C, H, W = logits.shape
                mask = reduce_segmentation_mask(sample["mask"].reshape(B, 1, 1, H, W), H, W).reshape(B, H, W)
                loss = criterion(logits, mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if get_world_size() > 1:
                total_loss_tensor = torch.tensor(total_loss, device=get_local_rank())
                torch.distributed.all_reduce(total_loss_tensor)
                avg_loss = total_loss_tensor.item() / get_world_size() / len(self.train_loader)
            else:
                avg_loss = total_loss / len(self.train_loader)

            if avg_loss < best_loss:
                best_loss = avg_loss
            logger.info(f'Linear Probe: Epoch {epoch+1}: loss={avg_loss:.4f} | best loss={best_loss:.4f}')

        probe.eval()
        total_val_loss = 0
        total_metrics = None
        with torch.no_grad():
            for sample in self.val_loader:
                sample = recursive_to_device(sample, self.device)

                features = encoder.evaluate(sample)
                logits = probe(features).detach()  
                B, C, H, W = logits.shape
                mask = reduce_segmentation_mask(sample["mask"].reshape(B, 1, 1, H, W), H, W).reshape(B, H, W)

                loss = criterion(logits, mask)

                total_val_loss += loss.item()
                if total_metrics is None:
                    total_metrics = {key: (val[0].item(), val[1]) for (key,val) in self.metrics_fn(logits, mask).items()}
                else:
                    total_metrics = {key: (total_metrics[key][0] + val[0].item(), val[1]) for (key,val) in self.metrics_fn(logits, mask).items()}

        if get_world_size() > 1:
            total_val_loss_tensor = torch.tensor(total_val_loss, device=get_local_rank())
            torch.distributed.all_reduce(total_val_loss_tensor)
            avg_val_loss = total_val_loss_tensor.item() / get_world_size() / len(self.val_loader)

            total_metrics = {key: (torch.distributed.all_reduce(torch.tensor(val[0], device=get_local_rank())), val[1]) for key, val in total_metrics.items()}
            avg_metrics = {key: (val[0] / get_world_size() / len(self.val_loader), val[1]) for key, val in total_metrics.items()}
        else:
            avg_val_loss = total_val_loss/len(self.val_loader)
            avg_metrics = {key: (val[0] / len(self.val_loader), val[1]) for key, val in total_metrics.items()}

        log_list = [f"Linear Probe: End {epoch+1}: loss = {avg_loss:.4f} | val loss = {avg_val_loss:.4f}"]
        for key,val in avg_metrics.items():
            log_list.append(f"{key} = {val[0]:.4f}")
        log_str = ' | '.join(log_list)
        logger.info(log_str)
        self.visualizer._log(epoch, sample, (logits,sample['mask']), logger)
        return avg_val_loss, avg_metrics

    def _log(self, step, input, results, logger, shape):
        if not is_main_process():
            return
        last_loss, metrics = results
        log_dict = {
            f"eval/{self.eval_name}/last_epoch_loss": last_loss,
        }
        for key, val in metrics.items():
            log_dict[f"eval/{self.eval_name}/last_{key.lower().replace(' ', '_')}"] = val[0]
        wandb.log(log_dict, step=step)
        output = {'probe_last_loss': (last_loss, 1)}
        metrics = {f"probe_{key.lower().replace(' ', '_')}": val for key, val in metrics.items()}
        output.update(metrics)
        return output