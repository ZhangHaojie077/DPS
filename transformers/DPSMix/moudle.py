import math
from this import s
import numpy as np
import torch
from torch.futures import S
import torch.nn as nn
from torch.nn import parameter
import torch.nn.init as init
import torch.nn.functional as F
from torch import Tensor
from .grads_ops import grads_ops

from torch.nn import Parameter

class DPSMix(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    def __init__(self, in_features: int, out_features: int, bias: bool = True, p=0.0, max_steps=0,update_ratio=0.1,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(DPSMix, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.steps = 0
        self.p = p
        self.max_steps = max_steps
        self.update_ratio = update_ratio
        self.target = None
        self.pretrain_weight = None
        self.update_steps = int(self.max_steps*self.update_ratio)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            self.steps += 1
            if self.steps == 1:
                self.target = self.weight.data.clone()
                self.pretrain_weight = self.weight.data.clone()
            else:
                if math.ceil(self.steps/self.update_steps)%2 == 0 and (self.steps-1)%self.update_steps == 0:
                    self.target = self.weight.data.clone()
            if math.ceil(self.steps/self.update_steps)%2 == 1:
                cur_target = self.pretrain_weight
            else:
                cur_target = self.target
        else:
            cur_target = self.target
        

        return F.linear(input, grads_ops(self.weight, cur_target, self.p, self.steps, self.update_steps, self.training), self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, p={}, update_ratio={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.p, self.update_ratio
        )