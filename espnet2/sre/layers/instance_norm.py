from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.layers.abs_normalize import AbsNormalize


class ESPnetInstanceNorm1d(AbsNormalize):
    """InstanceNorm1d supporting zero-padded tensor

    The difference from InstanceNorm1d

    1. Normalize along the 2nd axis. i.e. Deal the input tensor as (B, D, C)
       where, B is batch size, D is feature size, C is channel number.
    2. Support variable length dimension using "input_lengths" tesnor
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        self.instance_norm = torch.nn.InstanceNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(
        self, x: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function

        Args:
            x: (B, L, D)
            ilens: (B,)
        """
        with torch.no_grad():
            x = self.instance_norm(x)
            if ilens is not None:
                # Fill 0
                mask = make_pad_mask(ilens, x, 1)
                x = x.masked_fill(mask, 0.0)
            else:
                mask = None

            if ilens is not None:
                x = x.masked_fill(mask, 0.0)
        return x, ilens

    '''
    def inverse(
        self, x: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return x, ilens
    '''
