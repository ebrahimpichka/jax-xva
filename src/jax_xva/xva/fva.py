from functools import partial

import jax.numpy as jnp
from jax import jit, vmap

from jax_xva.market_data import MarketData
from jax_xva.exposure import ExposureCalculator




class FVA:
    """
    Funding Valuation Adjustment calculator.
    
    FVA represents the cost of funding uncollateralized derivatives positions.
    It accounts for the spread between the risk-free rate and the institution's
    funding rate.
    """
    
    def __init__(self, market_data: MarketData, exposure_calculator: ExposureCalculator):
        self.market_data = market_data
        self.exposure_calculator = exposure_calculator
    
    @partial(jit, static_argnums=(0,))
    def calculate_fva(self, counterparty: str, own_entity: str,
                     exposure_profile: jnp.ndarray, times: jnp.ndarray) -> float:
        """
        Calculate FVA using the formula:
        FVA = ∫ (EE(t) - NEE(t)) * s_F(t) * SP_cpty(t) * SP_own(t) * DF(t) dt
        
        Args:
            counterparty: Counterparty name
            own_entity: Own entity name
            exposure_profile: Expected exposure profile (positive values)
            times: Corresponding time points
            
        Returns:
            FVA value
        """
        # Calculate discount factors
        discount_factors = self.exposure_calculator.discount_factors(times)
        
        # Calculate survival probabilities
        cpty_survival = vmap(lambda t: self.exposure_calculator.survival_probability(counterparty, t))(times)
        own_survival = vmap(lambda t: self.exposure_calculator.survival_probability(own_entity, t))(times)
        
        # Interpolate funding spreads
        funding_times = jnp.array(sorted(self.market_data.funding_spreads.keys()))
        funding_spreads = jnp.array([self.market_data.funding_spreads[t] for t in funding_times])
        
        # Vectorized interpolation of funding spreads
        def interpolate_spread(t):
            idx = jnp.searchsorted(funding_times, t)
            if idx == 0:
                return funding_spreads[0]
            elif idx == len(funding_times):
                return funding_spreads[-1]
            t0, t1 = funding_times[idx-1], funding_times[idx]
            s0, s1 = funding_spreads[idx-1], funding_spreads[idx]
            return s0 + (s1 - s0) * (t - t0) / (t1 - t0)
        
        spreads = vmap(interpolate_spread)(times)
        
        # Integrate using trapezoidal rule
        dt = jnp.diff(times, prepend=times[0])
        integrand = exposure_profile * spreads * cpty_survival * own_survival * discount_factors
        
        # Return FVA
        return jnp.sum(integrand * dt)