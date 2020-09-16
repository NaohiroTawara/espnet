
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
import torch.nn as nn
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.sre.net.abs_net import AbsNet


class TDNN(AbsNet):
    """TDNNEncoder class.

    Args:
        input_size: The number of expected features in the input
        output_size: The number of output features
        hidden_size: The number of hidden features
        num_layers: Number of recurrent pooling
        dropout: dropout probability

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 400,
        output_size: int = 400,
        batch_norm: bool = True,
    ):
        assert check_argument_types()
        super(AbsNet, self).__init__()
        self._output_size = output_size

        if batch_norm:
            self.enc = torch.nn.Sequential(
                    torch.nn.Conv1d(input_size, hidden_size, 5, 1),
                    torch.nn.BatchNorm1d(hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Conv1d(hidden_size, hidden_size, 3, 2),
                    torch.nn.BatchNorm1d(hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Conv1d(hidden_size, hidden_size, 3, 3),
                    torch.nn.BatchNorm1d(hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Conv1d(hidden_size, output_size, 1, 1),
                    torch.nn.BatchNorm1d(output_size),
            )
        else:
            self.enc = torch.nn.Sequential(
                torch.nn.Conv1d(input_size, hidden_size, 5, 1),
                torch.nn.ReLU(),
                torch.nn.Conv1d(hidden_size, hidden_size, 3, 2),
                torch.nn.ReLU(),
                torch.nn.Conv1d(hidden_size, hidden_size, 3, 3),
                torch.nn.ReLU(),
                torch.nn.Conv1d(hidden_size, output_size, 1, 1),
            )


    def output_size(self) -> int:
        return self._output_size

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, L, D)
            x_lengths: (B,)
        Returns:
            (B, L, O)
        """

        # x: (B, L, D) -> (B, D, L)
        x = x.transpose(1, 2)
        # x: (B, D, L) -> (B, O, L)
        x = self.enc(x)
        # x: (B, O, L) -> (B, L, O)
        x = x.transpose(1, 2)

        return x, x_lengths