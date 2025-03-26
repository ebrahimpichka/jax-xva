
from dataclasses import dataclass

@dataclass
class TradeData:
    """Base class for trade data."""
    trade_id: str
    counterparty: str
    start_date: float
    end_date: float
    notional: float