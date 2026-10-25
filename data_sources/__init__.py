"""Live data ingestion adapters for Malaysian 4D operators."""
from .base import LotterySource, NormalizedDraw
from .damacai import Damacai
from .magnum import Magnum
from .fouryes3 import FourYes3

__all__ = ['LotterySource', 'NormalizedDraw', 'Damacai', 'Magnum', 'FourYes3']
