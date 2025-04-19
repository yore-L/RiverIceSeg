from typing import Any, Callable, Dict, List, Optional, Union
import collections
from collections import defaultdict
import copy
import os
import re
import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import Optimizer


def merge_dicts(*dicts: dict) -> dict:
    """Recursive dict merge.
        Instead of updating only top-level keys,
        ``merge_dicts`` recurses down into dicts nested
        to an arbitrary depth, updating keys.

        Args:
            *dicts: several dictionaries to merge

        Returns:
            dict: deep-merged dictionary
        """
    assert len(dicts) > 1
    dict_ = copy.deepcopy(dicts[0])

    for merge_dict in dicts[1:]:
        merge_dict = merge_dict or {}
        for k in merge_dict:
            if (
                k in dict_
                and isinstance(dict_[k], dict)
                and isinstance(merge_dict[k], collections.Mapping)
            ):
                dict_[k] = merge_dicts(dict_[k], merge_dict[k])
            else:
                dict_[k] = merge_dict[k]

    return dict_

def process_model_params(
        model: nn.Module, layerwise_params: Dict[str, dict] = None,
        no_bias_weight_decay: bool = True, lr_scaling = 1.0
    ) -> List[Union[torch.nn.Parameter, dict]]:
    """
    获取"torch.optim.Optimizer"的模型参数。
    Args：
        模型（torch.nn.Module）：模型到处理
        layerwise_params（Dict）：顺序敏感的字典，其中每个键都是正则表达式模式，值是逐层选项用于与图案匹配的图层
        no_bias_weight_decay（bool）：如果为真，则删除weight_decay对于模型中的所有"偏差"参数
        lr_scaling（float）：分层学习率缩放，如果为1.0，则学习率将不会按比例缩放
    Returns：
        iterable：优化器的参数
    """
    params = list(model.named_parameters())
    layerwise_params = layerwise_params or collections.OrderedDict()

    model_params = []
    for name, parameters in params:
        options = {}
        for pattern, pattern_options in layerwise_params.items():
            if re.match(pattern, name) is not None:
                options = merge_dicts(options, pattern_options)

        # 无偏差衰减https://arxiv.org/abs/1812.01187
        if no_bias_weight_decay and name.endswith('bias'):
            options['weight_decay'] = 0.0
        # lr 线性缩放 https://arxiv.org/pdf/1706.02677.pdf
        if "lr" in options:
            options["lr"] *= lr_scaling

        model_params.append({"params": parameters, **options})

    return model_params

class Lookahead(Optimizer):
    """Implements Lookahead algorithm.

        It has been proposed in `Lookahead Optimizer: k steps forward,
        1 step back`_.

        Adapted from:
        https://github.com/alphadl/lookahead.pytorch (MIT License)

        .. _`Lookahead Optimizer\: k steps forward, 1 step back`:
            https://arxiv.org/abs/1907.08610
        """
    def __init__(self, optimizer: Optimizer, k: int = 5, alpha: float = 0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure: Optional[Callable] = None):
       """
       执行优化器步骤
        Args：
            closure（可调用，可选）：重新评估的模型并返回损失。
        Returns：
            计算损失
        """
       loss = self.optimizer.step(closure)
       for group in self.param_groups:
           if group["counter"] == 0:
               self.update(group)
           group["counter"] += 1
           if group["counter"] >= self.k:
               group["counter"] = 0
       return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["slow_state"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["fast_state"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)

    @classmethod
    def get_from_params(cls, params: Dict, base_optimizer_params: Dict = None, **kwargs) -> "Lookahead":
        from catalyst.registry import OPTIMIZERS
        base_optimizer = OPTIMIZERS.get_from_params(params=params, **base_optimizer_params)
        optimizer = cls(optimizer=base_optimizer, **kwargs)
        return optimizer