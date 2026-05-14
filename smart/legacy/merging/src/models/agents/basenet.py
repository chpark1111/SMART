from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def normalize_feature(
        self, obs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Todo: Add feature normalization
        # Note that tetrahedrals are zero padded to support different length tetrahedrals
        return obs
