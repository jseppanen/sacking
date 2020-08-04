
from collections import defaultdict
from numbers import Number
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ImportError:
    wandb = None


_MEASUREMENTS: Optional[Dict[str, List[Tensor]]] = None


class Measurements:
    """Log statistics of values over several batches."""

    def __init__(self):
        global _MEASUREMENTS
        if _MEASUREMENTS is not None:
            raise TypeError("Measurements is singleton")
        _MEASUREMENTS = defaultdict(list)

    def close(self):
        global _MEASUREMENTS
        _MEASUREMENTS = None

    @staticmethod
    @torch.no_grad()
    def update(values: Dict[str, Union[Number, List, Tensor]]) -> None:
        """Add new measurements."""

        if _MEASUREMENTS is None:
            return

        for key, value in values.items():
            if isinstance(value, Number):
                value = [value]
            value = torch.as_tensor(value)
            if value.dim() != 1:
                raise TypeError("only vector shaped measurements are supported")
            _MEASUREMENTS[key].append(value.detach())

    def report(self, writer: SummaryWriter, step: int) -> None:
        """Write statistics to Tensorboard/W&B."""

        assert _MEASUREMENTS is not None

        measurements = {
            key: torch.cat(values).cpu().numpy()
            for key, values in _MEASUREMENTS.items()
        }
        for key in measurements:
            _MEASUREMENTS.pop(key)
        for key, values in measurements.items():
            writer.add_histogram(key, values, step)
            writer.add_scalar(f"{key}_mean", values.mean(), step)
            writer.add_scalar(f"{key}_std", values.std(), step)
        if wandb and wandb.run:
            wandb.log({key: wandb.Histogram(values)
                       for key, values in measurements.items()},
                      step=step)
            wandb.log({f"{key}_mean": values.mean()
                       for key, values in measurements.items()},
                      step=step)
            wandb.log({f"{key}_std": values.std()
                       for key, values in measurements.items()},
                      step=step)
