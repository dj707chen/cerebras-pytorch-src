# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from .BCELoss import BCELoss
from .BCEWithLogitsLoss import BCEWithLogitsLoss
from .CosineEmbeddingLoss import CosineEmbeddingLoss
from .CrossEntropyLoss import CrossEntropyLoss
from .CTCLoss import CTCLoss
from .DiceCELoss import DiceCELoss
from .DiceLoss import DiceLoss
from .GaussianNLLLoss import GaussianNLLLoss
from .HingeEmbeddingLoss import HingeEmbeddingLoss
from .HuberLoss import HuberLoss
from .KLDivLoss import KLDivLoss
from .L1Loss import L1Loss
from .MarginRankingLoss import MarginRankingLoss
from .MSELoss import MSELoss
from .MultiLabelSoftMarginLoss import MultiLabelSoftMarginLoss
from .MultiMarginLoss import MultiMarginLoss
from .NLLLoss import NLLLoss
from .PoissonNLLLoss import PoissonNLLLoss
from .SmoothL1Loss import SmoothL1Loss
from .TripletMarginLoss import TripletMarginLoss
from .TripletMarginWithDistanceLoss import TripletMarginWithDistanceLoss

__all__ = [
    "BCELoss",
    "BCEWithLogitsLoss",
    "CosineEmbeddingLoss",
    "CrossEntropyLoss",
    "CTCLoss",
    "DiceCELoss",
    "DiceLoss",
    "GaussianNLLLoss",
    "HingeEmbeddingLoss",
    "HuberLoss",
    "KLDivLoss",
    "L1Loss",
    "MarginRankingLoss",
    "MSELoss",
    "MultiLabelSoftMarginLoss",
    "MultiMarginLoss",
    "NLLLoss",
    "PoissonNLLLoss",
    "SmoothL1Loss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
]
