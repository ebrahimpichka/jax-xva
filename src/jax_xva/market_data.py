
from dataclasses import dataclass
from typing import Dict, List

import jax.numpy as jnp


@dataclass
class MarketData:
    """Container for market data required for XVA calculations."""
    valuation_date: float
    discount_curve: Dict[float, float]
    hazard_rates: Dict[str, Dict[float, float]]  # Entity -> {time: hazard_rate}
    recovery_rates: Dict[str, float]  # Entity -> recovery_rate
    funding_spreads: Dict[float, float]  # Time -> spread
    
    def __post_init__(self):
        """Convert dictionaries to sorted arrays for more efficient JAX operations."""
        # discount curve to arrays
        times = jnp.array(sorted(self.discount_curve.keys()))
        rates = jnp.array([self.discount_curve[t] for t in times])
        self.discount_times = times
        self.discount_rates = rates
        
        # hazard rates to arrays
        self.hazard_times = {}
        self.hazard_values = {}
        for entity, rates in self.hazard_rates.items():
            times = jnp.array(sorted(rates.keys()))
            values = jnp.array([rates[t] for t in times])
            self.hazard_times[entity] = times
            self.hazard_values[entity] = values


def create_sample_market_data(
        valuation_date: float = 0.0,
        tenors: List[float] = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0),
        base_rate: float = 0.03,
        hazard_base: float = 0.02,
        funding_spread_base: float = 0.005) -> MarketData:
    """
    Create sample market data for testing.
    
    Args:
        valuation_date: Valuation date as year fraction from reference
        tenors: List of tenors in years
        base_rate: Base interest rate
        hazard_base: Base hazard rate
        funding_spread_base: Base funding spread
        
    Returns:
        MarketData object with sample data
    """
    # Create discount curve: simple increasing rates
    discount_curve = {t: base_rate + 0.002 * jnp.sqrt(t) for t in tenors}
    
    # Create hazard rates for two counterparties
    hazard_rates = {
        'Counterparty_A': {t: hazard_base * (1.0 + 0.1 * jnp.sqrt(t)) for t in tenors},
        'Bank': {t: hazard_base * 0.7 * (1.0 + 0.05 * jnp.sqrt(t)) for t in tenors}
    }
    
    # Create recovery rates
    recovery_rates = {
        'Counterparty_A': 0.4,
        'Bank': 0.4
    }
    
    # Create funding spreads
    funding_spreads = {t: funding_spread_base * (1.0 + 0.1 * jnp.sqrt(t)) for t in tenors}
    
    return MarketData(
        valuation_date=valuation_date,
        discount_curve=discount_curve,
        hazard_rates=hazard_rates,
        recovery_rates=recovery_rates,
        funding_spreads=funding_spreads
    )