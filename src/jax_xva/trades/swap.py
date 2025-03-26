
from dataclasses import dataclass
from typing import List
from jax_xva.trades.base import TradeData

@dataclass
class SwapTradeData(TradeData):
    """Interest rate swap trade data."""
    fixed_rate: float
    payment_frequency: float
    is_payer: bool  # True if paying fixed rate

# TODO: double check correctness
class MultiCurrencySwapData(SwapTradeData):
    """Multi-currency swap trade data."""
    currencies: List[str]
    fixed_rates: List[float]
    payment_frequencies: List[float]
    is_payers: List[bool]
