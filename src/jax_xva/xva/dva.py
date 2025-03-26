
from functools import partial

import jax.numpy as jnp
from jax import jit, vmap

from jax_xva.market_data import MarketData
from jax_xva.exposure import ExposureCalculator

class DVA:
    """
    Debt Valuation Adjustment calculator.
    """
    
    def __init__(self, market_data: MarketData, exposure_calculator: ExposureCalculator):
        self.market_data = market_data
        self.exposure_calculator = exposure_calculator
        
    @partial(jit, static_argnums=(0,))
    def calculate_dva(self, own_entity: str, negative_exposure_profile: jnp.ndarray, 
                     times: jnp.ndarray) -> float:
        """
        Calculate DVA using the formula:
        DVA = (1-R) * ∫ NEE(t) * PD_own(t) * DF(t) dt
        
        Args:
            own_entity: Own entity name
            negative_exposure_profile: Negative expected exposure profile array
            times: Corresponding time points
            
        Returns:
            DVA value
        """
        # Get recovery rate
        recovery_rate = self.market_data.recovery_rates[own_entity]
        loss_given_default = 1.0 - recovery_rate
        
        # Calculate discount factors
        discount_factors = self.exposure_calculator.discount_factors(times)
        
        # Calculate default probability density
        def_prob_density = vmap(lambda t: self.exposure_calculator.default_probability_density(own_entity, t))(times)
        
        # Integrate using trapezoidal rule
        dt = jnp.diff(times, prepend=times[0])
        integrand = negative_exposure_profile * def_prob_density * discount_factors
        
        # Return DVA
        return loss_given_default * jnp.sum(integrand * dt)