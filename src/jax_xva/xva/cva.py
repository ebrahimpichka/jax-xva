from functools import partial

import jax.numpy as jnp
from jax import jit, vmap

from jax_xva.market_data import MarketData
from jax_xva.exposure import ExposureCalculator


class CVA:
    """
    Credit Valuation Adjustment calculator.
    """
    
    def __init__(self, market_data: MarketData, exposure_calculator: ExposureCalculator):
        self.market_data = market_data
        self.exposure_calculator = exposure_calculator
    
    @partial(jit, static_argnums=(0,))
    def calculate_cva(self, counterparty: str, exposure_profile: jnp.ndarray, 
                     times: jnp.ndarray) -> float:
        """
        Calculate CVA using the formula:
        CVA = (1-R) * ∫ EE(t) * PD(t) * DF(t) dt
        
        Args:
            counterparty: Name of counterparty
            exposure_profile: Expected exposure profile array
            times: Corresponding time points
            
        Returns:
            CVA value
        """
        # Get recovery rate
        recovery_rate = self.market_data.recovery_rates[counterparty]
        loss_given_default = 1.0 - recovery_rate
        
        # Calculate discount factors
        discount_factors = self.exposure_calculator.discount_factors(times)
        
        # Calculate default probability density
        def_prob_density = vmap(lambda t: self.exposure_calculator.default_probability_density(counterparty, t))(times)
        
        # Integrate using trapezoidal rule
        dt = jnp.diff(times, prepend=times[0])
        integrand = exposure_profile * def_prob_density * discount_factors
        
        # Return CVA 
        return loss_given_default * jnp.sum(integrand * dt)