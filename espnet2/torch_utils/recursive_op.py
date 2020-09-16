from distutils.version import LooseVersion

import torch

if torch.distributed.is_available():
    if LooseVersion(torch.__version__) > LooseVersion("1.0.1"):
        from torch.distributed import ReduceOp
    else:
        from torch.distributed import reduce_op as ReduceOp
else:
    ReduceOp = None


def recursive_sum(obj, weight: torch.Tensor, distributed: bool = False):
    assert weight.dim() == 1, weight.size()
    if isinstance(obj, (tuple, list)):
        return type(obj)(recursive_sum(v, weight, distributed) for v in obj)
    elif isinstance(obj, dict):
        return {k: recursive_sum(v, weight, distributed) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        assert obj.size() == weight.size(), (obj.size(), weight.size())
        obj = (obj * weight.type(obj.dtype)).sum()
        if distributed:
            torch.distributed.all_reduce(obj, op=ReduceOp.SUM)
        return obj
    elif obj is None:
        return None
    else:
        raise ValueError(type(obj))


def recursive_divide(a, b: torch.Tensor):
    if isinstance(a, (tuple, list)):
        return type(a)(recursive_divide(v, b) for v in a)
    elif isinstance(a, dict):
        return {k: recursive_divide(v, b) for k, v in a.items()}
    elif isinstance(a, torch.Tensor):
        assert a.size() == b.size(), (a.size(), b.size())
        return a / b.type(a.dtype)
    elif a is None:
        return None
    else:
        raise ValueError(type(a))


def recursive_average(obj, weight: torch.Tensor, distributed: bool = False):
    obj = recursive_sum(obj, weight, distributed)
    weight = weight.sum()
    if distributed:
        torch.distributed.all_reduce(weight, op=ReduceOp.SUM)
    # Normalize weight to be sum-to-1
    obj = recursive_divide(obj, weight)
    return obj, weight


def recursive_gather(output_obj, obj, distributed: bool = False):
    if isinstance(obj, (tuple, list)):
        return type(obj)(recursive_gather(o, v, distributed) for o, v in zip(output_obj, obj))
    elif isinstance(obj, dict):
        return {k: recursive_gather(v_o, v, distributed)
                for (k_o, v_o), (k, v) in zip(output_obj.items(), obj.items())}
    elif isinstance(obj, torch.Tensor):
        if distributed:
            # Note(naohiro): gather has not been implemented in NCCL, so all_gather is used.
            torch.distributed.all_gather(output_obj, obj)
        return output_obj
    elif obj is None:
        return None
    else:
        raise ValueError(type(obj))
