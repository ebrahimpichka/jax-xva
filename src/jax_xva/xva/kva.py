from functools import partial

import jax.numpy as jnp
from jax import jit

from jax_xva.market_data import MarketData
from jax_xva.exposure import ExposureCalculator

class KVA:
    """
    Capital Valuation Adjustment calculator.
    
    KVA represents the cost of holding capital against CCR (Counterparty Credit Risk)
    and other risks associated with derivatives transactions.
    """
    
    def __init__(self, market_data: MarketData, exposure_calculator: ExposureCalculator):
        self.market_data = market_data
        self.exposure_calculator = exposure_calculator
    
    @partial(jit, static_argnums=(0,))
    def calculate_kva(self, capital_profile: jnp.ndarray, times: jnp.ndarray, 
                     hurdle_rate: float) -> float:
        """
        Calculate KVA using the formula:
        KVA = ∫ K(t) * h * DF(t) dt
        
        Args:
            capital_profile: Regulatory capital profile
            times: Corresponding time points
            hurdle_rate: Hurdle rate for capital (cost of equity)
            
        Returns:
            KVA value
        """
        # Calculate discount factors
        discount_factors = self.exposure_calculator.discount_factors(times)
        
        # Integrate using trapezoidal rule
        dt = jnp.diff(times, prepend=times[0])
        integrand = capital_profile * hurdle_rate * discount_factors
        
        # Return KVA
        return jnp.sum(integrand * dt)
    
    @partial(jit, static_argnums=(0,1,))
    def calculate_capital_profile(self, exposure_profile: jnp.ndarray, counterparty: str,
                                 lgd: float = 0.45, alpha: float = 1.4) -> jnp.ndarray:
        """
        Calculate regulatory capital profile using standardized approach.
        
        This is a simplified version of SA-CCR or IMM capital calculation.
        
        Args:
            exposure_profile: Expected exposure profile
            counterparty: Counterparty name for PD lookup
            lgd: Loss given default (standard 0.45 in Basel rules)
            alpha: Alpha factor (1.4 in Basel rules)
            
        Returns:
            Capital profile
        """
        # Get PD for counterparty (simplified as average hazard rate)
        hazard_rates = self.market_data.hazard_values[counterparty]
        pd = jnp.mean(hazard_rates)
        
        # Calculate capital using simplified formula
        # K = alpha * EAD * (PD * LGD) / (1 - PD * LGD)
        # where EAD is the exposure at default (we use EE profile)
        
        rwa_factor = (pd * lgd) / (1.0 - pd * lgd)
        capital_profile = alpha * exposure_profile * rwa_factor
        
        # Apply 8% capital requirement
        capital_profile = capital_profile * 0.08
        
        return capital_profile

