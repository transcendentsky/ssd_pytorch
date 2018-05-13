"""
def scheduler(self, epoch):
        lr_max = self._lr_max
        lr_min = self._lr_min
        if epoch >= self.ti + self.base:
            self.base += self.ti
            self.ti = self.ti * 2
        epoch_n = epoch - self.base
        lr = lr_min + (lr_max - lr_min)*(math.cos(epoch_n * math.pi / self.ti / 2))
        # print("lr: ", lr, end=' ')
        # lr = lr_min + (lr_max - lr_min)*(math.cos(epoch_n * math.pi / ti /2))
        return lr
"""
import math
from bisect import bisect_right
from functools import partial
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class WarmRestart(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, ti, eta_min=0, last_epoch=-1):
        self.ti = ti
        self.eta_min = eta_min
        self.base = 0
        super(WarmRestart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.ti + self.base:
            self.base += self.ti
            self.ti = self.ti * 2
        epoch_n = self.last_epoch - self.base
        return [self.eta_min + (base_lr - self.eta_min) *
                math.cos(math.pi * epoch_n / self.ti / 2)
                for base_lr in self.base_lrs]