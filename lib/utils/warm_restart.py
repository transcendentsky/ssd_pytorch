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

    def __init__(self, optimizer, ti=70, eta_min=1e-10, last_epoch=-1, gamma=1.0, beta=0.5, fixed=False):
        self.ti = ti
        self.eta_min = eta_min
        self.base = 0
        self.gamma = gamma
        self.beta = beta
        self.ratio = 1.0
        self.fixed = fixed
        super(WarmRestart, self).__init__(optimizer, last_epoch)

    def set_base(self):
        if self.last_epoch >= self.ti + self.base:
            self.base += self.ti
            self.ti = self.ti * self.gamma
            self.ratio = self.ratio * self.beta
            return self.set_base()
        else:
            return

    def get_lr(self):
        if self.fixed:
            return self.base_lrs
        self.set_base()
        epoch_n = self.last_epoch - self.base
        diff = self.base_lrs[0] * self.ratio - self.eta_min
        if diff > 0:
            return [self.eta_min + (diff) *
                math.cos(math.pi * epoch_n / self.ti / 2)
                for base_lr in self.base_lrs]
        else:
            return self.base_lrs
