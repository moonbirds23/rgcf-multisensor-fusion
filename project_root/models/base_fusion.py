from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class FusionForwardOutput:
    pred: torch.Tensor
    weights: Optional[torch.Tensor] = None
    aux: Dict[str, torch.Tensor] = field(default_factory=dict)


class FusionModelBase(nn.Module, ABC):
    pass
