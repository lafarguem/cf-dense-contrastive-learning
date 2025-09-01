from abc import ABC, abstractmethod
import torch
from pretraining.utils.common import recursive_to_device, reduce_segmentation_mask
import traceback
import torch.nn.functional as F
from pretraining.utils.debug import warn_once
from pretraining.utils.distributed import is_main_process

class BaseEvaluator(ABC):
    def __init__(self, eval_name, eval_freq, output_dir=None, dataset=None):
        self.eval_name = eval_name
        self.eval_freq = eval_freq
        self.output_dir = output_dir

    @abstractmethod
    def evaluate(self, step, encoder, embeddings, labels, shape, logger):
        pass

    @abstractmethod
    def _log(self, step, input, results, logger, shape):
        pass

    def log(self, step, input, results=None, encoder=None, embeddings=None, labels=None, logger=None, shape=None):
        if results is None:
            results = self.evaluate(step, encoder, embeddings, labels, shape, logger)
        if results is None:
            warn_once(logger, f'{self.eval_name}_dnf', f'{self.eval_name} evaluator did not run.')
        else:
            return self._log(step, input, results, logger, shape)


class EvaluatorManager:
    def __init__(self, evaluators, dataloader, logger, output_dir, device):
        self.evaluators = evaluators
        for evaluator in self.evaluators:
            evaluator.output_dir = output_dir
        self.dataloader = dataloader
        self.iter_loader = iter(dataloader)
        self.logger = logger
        self.output_dir = output_dir
        self.device = device

    def data_step(self):
        try:
            sample = next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.dataloader)
            sample = next(self.iter_loader)
        for key,el in sample.items():
            sample[key] = el[:min(len(el),8)]
        return recursive_to_device(sample, device=self.device)
    
    def get_embeddings(self, encoder):
        dense_return = encoder.return_dense
        encoder.set_dense_return(True)
        sample = self.data_step()
        with torch.no_grad():
            out = encoder(sample)

        embeddings = out.dense_embeddings
        N,V,C,H,W = embeddings.shape
        embeddings = embeddings.permute(0, 1, 3, 4, 2).contiguous()
        embeddings = F.normalize(embeddings, dim=-1, eps=1e-8)
        embeddings = embeddings.reshape(N*V*H*W, C)

        mask = sample.get('mask', None)
        if mask is not None:
            labels = reduce_segmentation_mask(mask, H, W).flatten()
        else:
            labels = None
        encoder.set_dense_return(dense_return)
        return sample, embeddings, labels, (N,V,C,H,W)

    def evaluate(self, epoch, encoder, force_all=False):
        if self.evaluators == []:
            return 
        encoder.eval()
        to_use = [e for e in self.evaluators if (epoch % e.eval_freq == 0 or force_all)]
        if not to_use:
            return

        input, embeddings, labels, shape = self.get_embeddings(encoder)

        results = {}
        for evaluator in to_use:
            self.logger.info(f'Evaluating {evaluator.eval_name}')
            results[evaluator.eval_name] = evaluator.evaluate(encoder, embeddings, labels, shape)

        if is_main_process():
            return results
        else:
            return None

    def log(self, epoch, step, input=None, results=None, encoder=None, force_all=False):
        if self.evaluators == []:
            return {}
        encoder.eval()
        to_use = [e for e in self.evaluators if (epoch % e.eval_freq == 0 or force_all)]
        metrics = {}
        if not to_use:
            return metrics

        if results is not None:
            for evaluator in to_use:
                metric = evaluator.log(step, input=input, results=results[evaluator.eval_name])
                if metric is not None and is_main_process():
                    metrics.update(metric)
            return metrics

        input, embeddings, labels, shape = self.get_embeddings(encoder)
        for evaluator in to_use:
            self.logger.info(f'Step {step}: Evaluating {evaluator.eval_name}')
            try:
                metric = evaluator.log(
                    step,
                    input,
                    encoder=encoder,
                    embeddings=embeddings,
                    labels=labels,
                    logger=self.logger,
                    shape=shape
                )
                if metric is not None and is_main_process():
                    metrics.update(metric)
            except Exception as e:
                self.logger.error("=" * 80)
                self.logger.error(f"[Evaluator: {evaluator.__class__.__name__}] Exception occurred:")
                self.logger.error("".join(traceback.format_exception(type(e), e, e.__traceback__)).strip())
                self.logger.error("=" * 80)
        return metrics