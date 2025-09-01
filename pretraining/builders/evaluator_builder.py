from hydra.utils import instantiate
from pretraining.evaluation.base_evaluation import EvaluatorManager

def build_evaluator(cfg, eval_loader, logger):
    evaluators = []
    for evaluator_cfg in cfg.get('evaluator', []):
        if cfg.evaluator[evaluator_cfg] is not None:
            evaluators.append(instantiate(cfg.evaluator[evaluator_cfg], dataset=eval_loader.dataset))
    
    manager = EvaluatorManager(evaluators, eval_loader, logger, cfg.runtime.output_dir, cfg.runtime.device)

    return manager