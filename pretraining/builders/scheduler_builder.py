from hydra.utils import instantiate
from torch.optim.lr_scheduler import _LRScheduler

# Class from "https://github.com/ildoonet/pytorch-gradual-warmup-lr"
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
      Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
      Args:
          optimizer (Optimizer): Wrapped optimizer.
          multiplier: init learning rate = base lr / multiplier
          warmup_epoch: target learning rate is reached at warmup_epoch, gradually
          after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
      """

    def __init__(self, optimizer, multiplier, warmup_epoch, after_scheduler, last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch:
            return self.after_scheduler.get_lr()
        else:
            return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.)
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch > self.warmup_epoch:
            self.after_scheduler.step()
        else:
            super(GradualWarmupScheduler, self).step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """

        state = {key: value for key, value in self.__dict__.items() if key != 'optimizer' and key != 'after_scheduler'}
        state['after_scheduler'] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        after_scheduler_state = state_dict.pop('after_scheduler')
        self.__dict__.update(state_dict)
        self.after_scheduler.load_state_dict(after_scheduler_state)

def build_scheduler(cfg, optimizer):
    if "CosineAnnealingLR" in cfg.scheduler._target_:
        scheduler = instantiate(
            cfg.scheduler, 
            optimizer=optimizer,
            T_max=(cfg.train.epochs - cfg.train.warmup_epoch)*cfg.runtime.iter_per_epoch,
        )
    elif "MultiStepLR" in cfg.scheduler._target_:
        scheduler = instantiate(
            cfg.scheduler, 
            optimizer=optimizer,
            milestones=[(m - cfg.train.warmup_epoch)*cfg.runtime.iter_per_epoch 
                        for m in cfg.scheduler.milestones]
        )
    else:
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    if cfg.train.warmup_epoch > 0:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=cfg.train.warmup_multiplier,
            after_scheduler=scheduler,
            warmup_epoch=cfg.train.warmup_epoch * cfg.runtime.iter_per_epoch)
    return scheduler

class GlobalLossWeightScheduler:
    def __init__(self, update_func, model, criterion):
        self.i = 0
        self.global_loss_weight = 0
        self.get_weight = update_func
        self.model = model
        self.criterion = criterion
        self.model.set_toggle('both')
        self.step()

    def step(self):
        self.global_loss_weight = self.get_weight(self.i, self.global_loss_weight)
        self.criterion.global_loss_weight = self.global_loss_weight
        self.i += 1

class ConstantUpdater:
    def __init__(self, init_value, steps_in_epoch=None):
        self.init_value = init_value

    def __call__(self, i, prev):
        return self.init_value
    
class LinearUpdater:
    def __init__(self, keypoints, steps_in_epoch):
        self.init_value = keypoints[0][1]
        self.keypoint_idx = 0
        self.keypoints = [(x*steps_in_epoch, y) for x,y in keypoints]

    def __call__(self, i, prev):
        if self.keypoint_idx == len(self.keypoints) - 1:
            return self.keypoints[-1][1]
        x1, y1 = self.keypoints[self.keypoint_idx]
        x2, y2 = self.keypoints[self.keypoint_idx + 1]
        t = (i - x1) / (x2-x1)
        y = (1-t)*y1 + t*y2
        if i >= self.keypoints[self.keypoint_idx+1][0]:
            self.keypoint_idx += 1
        
        return y

def build_global_loss_weight_scheduler(cfg, model, criterion, steps_in_epoch):
    return GlobalLossWeightScheduler(instantiate(cfg.global_loss_weight_scheduler, steps_in_epoch=steps_in_epoch), model, criterion)