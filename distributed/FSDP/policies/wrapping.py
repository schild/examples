# holds various wrapping policies for fsdp


import torch.distributed as dist
import torch.nn as nn
import torch

from transformers.models.t5.modeling_t5 import T5Block

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    BackwardPrefetch,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

import functools
from typing import Type


def get_size_policy(min_params=1e8):
    return functools.partial(
        size_based_auto_wrap_policy, min_num_params=min_params
    )


def get_t5_wrapper():
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    """
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            T5Block,
        },
    )
