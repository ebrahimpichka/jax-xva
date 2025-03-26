from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
import jax.scipy.stats
from jax_xva.market_data import MarketData
from jax_xva.exposure import ExposureCalculator

class MVA:
    """
    Margin Valuation Adjustment calculator.
    
    MVA represents the cost of funding the initial margin (IM) requirements
    for cleared or bilateral trades under regulatory rules.
    """
    
    def __init__(self, market_data: MarketData, exposure_calculator: ExposureCalculator):
        self.market_data = market_data
        self.exposure_calculator = exposure_calculator
    
    @partial(jit, static_argnums=(0,))
    def calculate_mva(self, im_profile: jnp.ndarray, times: jnp.ndarray, funding_spread: float) -> float:
        """
        Calculate MVA using the formula:
        MVA = ∫ IM(t) * s_F(t) * DF(t) dt
        
        Args:
            im_profile: Initial margin profile array
            times: Corresponding time points
            funding_spread: Funding spread for initial margin
            
        Returns:
            MVA value
        """
        # Calculate discount factors
        discount_factors = self.exposure_calculator.discount_factors(times)
        
        # Integrate using trapezoidal rule
        dt = jnp.diff(times, prepend=times[0])
        integrand = im_profile * funding_spread * discount_factors
        
        # Return MVA
        return jnp.sum(integrand * dt)
    
    @partial(jit, static_argnums=(0,))
    def simulate_im_profile(self, key, pfe_profile: jnp.ndarray, holding_period: float = 10.0/365.0,
                           confidence_level: float = 0.99, scaling_factor: float = 1.0) -> jnp.ndarray:
        """
        Simulate initial margin profile based on potential future exposure.
        
        The ISDA SIMM or regulatory CEM/SA-CCR approaches would be used in practice,
        but we use a simplified approach here.
        
        Args:
            key: JAX random key
            pfe_profile: PFE profile at some confidence level
            holding_period: MPOR (margin period of risk) in years
            confidence_level: Confidence level for IM calculation
            scaling_factor: Additional multiplier for conservatism
            
        Returns:
            Initial margin profile
        """
        # Scale PFE by sqrt(holding period ratio) and confidence level adjustment
        z_score_ratio = jax.scipy.stats.norm.ppf(confidence_level) / jax.scipy.stats.norm.ppf(0.95)
        holding_period_ratio = jnp.sqrt(holding_period / (1.0/365.0))  # Relative to 1 day
        
        # Apply scalings
        im_profile = pfe_profile * z_score_ratio * holding_period_ratio * scaling_factor
        
        return im_profile