from dataclasses import dataclass
from jax_xva.trades.base import TradeData

@dataclass
class OptionTradeData(TradeData):
    """Option trade data."""
    strike: float
    underlying: str
    is_call: bool

