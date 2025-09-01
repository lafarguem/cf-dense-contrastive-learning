from abc import ABC, abstractmethod
import torch
from pretraining.utils.common import recursive_to_device
import traceback
from pretraining.utils.debug import warn_once
from pretraining.utils.distributed import get_local_rank, is_main_process

class BaseEvaluator(ABC):
    def __init__(self, eval_name, eval_freq, output_dir=None, dataset=None):
        self.eval_name = eval_name
        self.eval_freq = eval_freq
        self.output_dir = output_dir

    @abstractmethod
    def evaluate(self, model, out, mask):
        pass

    @abstractmethod
    def _log(self, step, input, results, logger):
        pass

    def log(self, step, input, results=None, model=None, out=None, mask=None, logger=None):
        if results is None:
            results = self.evaluate(model, out, mask)
        if results is None:
            warn_once(logger, f'{self.eval_name}_dnf', f'{self.eval_name} evaluator did not run.')
        else:
            self._log(step, input, results, logger)

class EvaluatorManager:
    def __init__(self, evaluators, dataloader, logger, output_dir):
        self.evaluators = evaluators
        for evaluator in self.evaluators:
            evaluator.output_dir = output_dir
        self.dataloader = dataloader
        self.iter_loader = iter(dataloader)
        self.logger = logger
        self.output_dir = output_dir

    def data_step(self):
        try:
            sample = next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.dataloader)
            sample = next(self.iter_loader)

        return recursive_to_device(sample, device=f'cuda:{get_local_rank()}')
    
    def get_output(self, model):
        sample = self.data_step()
        with torch.no_grad():
            out = model(sample['x'])

        mask = sample.get('mask', None)

        return sample, out, mask

    def evaluate(self, epoch, model, force_all=False):
        model.eval()
        to_use = [e for e in self.evaluators if (epoch % e.eval_freq == 0 or force_all)]
        if not to_use:
            return

        input, out, mask = self.get_output(model)

        if is_main_process():
            results = {}
            for evaluator in to_use:
                self.logger.info(f'Evaluating {evaluator.eval_name}')
                results[evaluator.eval_name] = evaluator.evaluate(model, out, mask)
            return results
        else:
            return None

    def log(self, epoch, step, input=None, results=None, model=None, force_all=False):
        model.eval()
        to_use = [e for e in self.evaluators if (epoch % e.eval_freq == 0 or force_all)]
        metrics = {}
        if not to_use:
            return metrics

        if results is not None:
            if is_main_process():
                for evaluator in to_use:
                    metric = evaluator.log(step, input=input, results=results[evaluator.eval_name])
                if metric is not None:
                    metrics.update(metric)
            return metrics

        input, out, mask = self.get_output(model)

        if is_main_process():
            for evaluator in to_use:
                self.logger.info(f'Step {step}: Evaluating {evaluator.eval_name}')
                try:
                    metric = evaluator.log(
                        step,
                        input,
                        model=model,
                        out=out,
                        mask=mask,
                        logger=self.logger,
                    )
                    if metric is not None:
                        metrics.update(metric)
                except Exception as e:
                    self.logger.error("=" * 80)
                    self.logger.error(f"[Evaluator: {evaluator.__class__.__name__}] Exception occurred:")
                    self.logger.error("".join(traceback.format_exception(type(e), e, e.__traceback__)).strip())
                    self.logger.error("=" * 80)
        return metrics