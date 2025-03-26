from functools import partial

import jax.numpy as jnp
from jax import jit

from jax_xva.market_data import MarketData
from jax_xva.exposure import ExposureCalculator

class ColVA:
    """
    Collateral Valuation Adjustment calculator.
    
    ColVA represents the cost/benefit of posting/receiving collateral.
    """
    
    def __init__(self, market_data: MarketData, exposure_calculator: ExposureCalculator):
        self.market_data = market_data
        self.exposure_calculator = exposure_calculator
    
    @partial(jit, static_argnums=(0,))
    def calculate_colva(self, collateral_profile: jnp.ndarray, times: jnp.ndarray,
                       collateral_rate: float, funding_rate: float) -> float:
        """
        Calculate ColVA using the formula:
        ColVA = ∫ C(t) * (r_C - r_F) * DF(t) dt
        
        Args:
            collateral_profile: Collateral balance profile
            times: Corresponding time points
            collateral_rate: Rate paid/received on collateral
            funding_rate: Funding rate
            
        Returns:
            ColVA value
        """
        # Calculate discount factors
        discount_factors = self.exposure_calculator.discount_factors(times)
        
        # Rate difference
        rate_diff = collateral_rate - funding_rate
        
        # Integrate using trapezoidal rule
        dt = jnp.diff(times, prepend=times[0])
        integrand = collateral_profile * rate_diff * discount_factors
        
        # Return ColVA
        return jnp.sum(integrand * dt)