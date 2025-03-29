from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap

from jax_xva.market_data import MarketData
from jax_xva.utils.interpolation import (
    interpolate_discount_factors,
    interpolate_hazard_rates,
    interpolate_survival_probabilities,
    interpolate_default_probability_density
)

class ExposureCalculator:
    """
    Calculate exposures for XVA.
    
    This class provides utilities for exposure calculation, including
    interpolation of market curves and calculation of risk metrics.
    """
    
    def __init__(self, market_data: MarketData):
        self.market_data = market_data
    
    @partial(jit, static_argnums=(0,))
    def interpolate_discount_factor(self, t: float) -> float:
        """Interpolate discount factor at time t."""
        times = self.market_data.discount_times
        rates = self.market_data.discount_rates
        
        # Linear interpolation
        idx = jnp.searchsorted(times, t)
        
        # Extrapolate if beyond curve
        if idx == 0:
            return jnp.exp(-rates[0] * t)
        elif idx == len(times):
            return jnp.exp(-rates[-1] * t)
        
        # Interpolate
        t0, t1 = times[idx-1], times[idx]
        r0, r1 = rates[idx-1], rates[idx]
        r_t = r0 + (r1 - r0) * (t - t0) / (t1 - t0)
        
        return jnp.exp(-r_t * t)
    
    @partial(jit, static_argnums=(0,))
    def discount_factors(self, times: jnp.ndarray) -> jnp.ndarray:
        """Calculate discount factors for an array of times."""
        return vmap(self.interpolate_discount_factor)(times) # changed to interpax
        # return interpolate_discount_factors(
        #     self.market_data.discount_times,
        #     self.market_data.discount_rates,
        #     times
        # )
    
    @partial(jit, static_argnums=(0,))
    def interpolate_hazard_rate(self, entity: str, t: float) -> float:
        """Interpolate hazard rate for entity at time t."""
        times = self.market_data.hazard_times[entity]
        rates = self.market_data.hazard_values[entity]
        
        # Linear interpolation
        idx = jnp.searchsorted(times, t)
        
        # Extrapolate if beyond curve
        if idx == 0:
            return rates[0]
        elif idx == len(times):
            return rates[-1]
        
        # Interpolate
        t0, t1 = times[idx-1], times[idx]
        r0, r1 = rates[idx-1], rates[idx]
        
        return r0 + (r1 - r0) * (t - t0) / (t1 - t0)
    
    @partial(jit, static_argnums=(0,))
    def survival_probability(self, entity: str, t: float) -> float:
        """Calculate survival probability for entity at time t."""
        hazard_rate = self.interpolate_hazard_rate(entity, t)
        return jnp.exp(-hazard_rate * t)
        # return interpolate_survival_probabilities(
        #     self.market_data.hazard_times[entity],
        #     self.market_data.hazard_values[entity],
        #     jnp.array([t])
        # )[0]
    
    @partial(jit, static_argnums=(0,))
    def default_probability_density(self, entity: str, t: float) -> float:
        """Calculate default probability density for entity at time t."""
        hazard_rate = self.interpolate_hazard_rate(entity, t)
        return hazard_rate * self.survival_probability(entity, t)
        # return interpolate_default_probability_density(
        #     self.market_data.hazard_times[entity],
        #     self.market_data.hazard_values[entity],
        #     jnp.array([t])
        # )[0]