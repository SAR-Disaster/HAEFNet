                              
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

                                                              
_num_parallel = 3


def set_num_parallel(n):
                                      
    global _num_parallel
    _num_parallel = n
    print(f"Setting num_parallel to {n}")
    return _num_parallel


def get_num_parallel():
    return _num_parallel


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        if not isinstance(x_parallel, list):
            raise TypeError(f"Expected list input, got {type(x_parallel)}")
        return [self.module(x) for x in x_parallel]


class Additional_One_ModuleParallel(nn.Module):
    def __init__(self, module):
        super(Additional_One_ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel, x_arg):
        if x_arg == None:
            return [self.module(x, None) for x in x_parallel]
        elif isinstance(x_arg, list):
            return [self.module(x_parallel[i], x_arg[i]) for i in range(len(x_parallel))]
        else:
            return [self.module(x_parallel[i], x_arg) for i in range(len(x_parallel))]


class Additional_Two_ModuleParallel(nn.Module):
    def __init__(self, module):
        super(Additional_Two_ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel, x_arg1, x_arg2):
        return [self.module(x_parallel[i], x_arg1, x_arg2) for i in range(len(x_parallel))]


class LayerNormParallel(nn.Module):
    def __init__(self, num_features):
        super(LayerNormParallel, self).__init__()

                          
        num_parallel = get_num_parallel()

                          
        self.layers = nn.ModuleList([nn.LayerNorm(num_features, eps=1e-6) for _ in range(num_parallel)])

    def forward(self, x):
        num_parallel = get_num_parallel()                 

        if isinstance(x, torch.Tensor):
            return self.layers[0](x)

        assert isinstance(x, list), "Input should be a list of tensors"
        assert len(x) == num_parallel, f"Expected {num_parallel} inputs, got {len(x)}"

        out = []
        for i in range(num_parallel):
            out.append(self.layers[i](x[i]))

        return out
