import warnings
import torch
from torch.nn.parameter import Parameter

from torch.nn.modules.module import Module
from torch.nn import functional as F

from torch.nn.modules.activation import Threshold

class Swish(Threshold):
    r"""Applies the rectified linear unit function element-wise
        :math:`\text{ReLU}(x)= \max(0, x)`

        .. image:: scripts/activation_images/ReLU.png

        Args:
            inplace: can optionally do the operation in-place. Default: ``False``

        Shape:
            - Input: :math:`(N, *)` where `*` means, any number of additional
              dimensions
            - Output: :math:`(N, *)`, same shape as the input

        Examples::

            >>> m = nn.ReLU()
            >>> input = torch.randn(2)
            >>> output = m(input)
        """

    def __init__(self, inplace=False):
        super(Swish, self).__init__(0, 0, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

    def forward(self, input):
        return swish(input)

def swish(input):
    return input * F.sigmoid(input)