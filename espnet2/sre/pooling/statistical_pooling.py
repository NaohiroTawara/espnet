import torch
from espnet2.sre.pooling.abs_pooling import AbsPooling
from typeguard import check_argument_types


class StatisticalPooling(AbsPooling):
    def __init__(self, pooling_type: str="mean_std"):
        assert check_argument_types()
        super().__init__()
        self.pooling_type = pooling_type

    def extr_repr(self):
        return f"pooling_type={self.pooling_type}"

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor = None):
        if "mean" in self.pooling_type:
            z1 = torch.mean(x, 2)
            if "std" in self.pooling_type:
                z2 = torch.mean(torch.mul(x, x), 2) - torch.mul(z1, z1)
                x = torch.cat([z1, z2], 1)
            else:
                x = z1
        elif self.pooling_type == "last":
            x = x[:, :, -1]
        else:
            assert f"unknown pooling type:{self.pooling_type}"

        return x
