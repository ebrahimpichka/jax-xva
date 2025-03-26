from functools import partial
from typing import Dict, Optional, Union

import jax.numpy as jnp
from jax import jit, vmap
import interpax

def interpolate_1d(x: jnp.ndarray, y: jnp.ndarray, 
                  x_new: Union[float, jnp.ndarray],
                  method: str = 'linear',
                  extrapolate: bool = True) -> Union[float, jnp.ndarray]:
    """
    One-dimensional interpolation using interpax.
    
    Args:
        x: Original x-coordinates
        y: Original y-values
        x_new: New x-coordinates to interpolate at
        method: Interpolation method ('linear', 'cubic', 'quintic')
        extrapolate: Whether to extrapolate beyond the bounds
        
    Returns:
        Interpolated values
    """
    return interpax.interp1d(x, y, method=method, extrapolate=extrapolate)(x_new)

@partial(jit, static_argnums=(0,))
def interpolate_discount_factors(times: jnp.ndarray,
                               rates: jnp.ndarray,
                               new_times: jnp.ndarray) -> jnp.ndarray:
    """
    Interpolate discount factors from rates.
    
    Args:
        times: Original time points
        rates: Original rates
        new_times: New time points to interpolate at
        
    Returns:
        Interpolated discount factors
    """
    # First interpolate rates
    interpolated_rates = interpolate_1d(times, rates, new_times)
    # Then calculate discount factors
    return jnp.exp(-interpolated_rates * new_times)

@partial(jit, static_argnums=(0,))
def interpolate_hazard_rates(times: jnp.ndarray,
                           rates: jnp.ndarray,
                           new_times: jnp.ndarray) -> jnp.ndarray:
    """
    Interpolate hazard rates.
    
    Args:
        times: Original time points
        rates: Original hazard rates
        new_times: New time points to interpolate at
        
    Returns:
        Interpolated hazard rates
    """
    return interpolate_1d(times, rates, new_times)

@partial(jit, static_argnums=(0,))
def interpolate_survival_probabilities(times: jnp.ndarray,
                                    hazard_rates: jnp.ndarray,
                                    new_times: jnp.ndarray) -> jnp.ndarray:
    """
    Interpolate survival probabilities from hazard rates.
    
    Args:
        times: Original time points
        hazard_rates: Original hazard rates
        new_times: New time points to interpolate at
        
    Returns:
        Interpolated survival probabilities
    """
    # First interpolate hazard rates
    interpolated_hazards = interpolate_1d(times, hazard_rates, new_times)
    # Then calculate survival probabilities
    return jnp.exp(-interpolated_hazards * new_times)

@partial(jit, static_argnums=(0,))
def interpolate_default_probability_density(times: jnp.ndarray,
                                         hazard_rates: jnp.ndarray,
                                         new_times: jnp.ndarray) -> jnp.ndarray:
    """
    Interpolate default probability density from hazard rates.
    
    Args:
        times: Original time points
        hazard_rates: Original hazard rates
        new_times: New time points to interpolate at
        
    Returns:
        Interpolated default probability density
    """
    # First interpolate hazard rates
    interpolated_hazards = interpolate_1d(times, hazard_rates, new_times)
    # Then calculate survival probabilities
    survival_probs = jnp.exp(-interpolated_hazards * new_times)
    # Finally calculate default probability density
    return interpolated_hazards * survival_probs

@partial(jit, static_argnums=(0,))
def interpolate_funding_spreads(times: jnp.ndarray,
                              spreads: jnp.ndarray,
                              new_times: jnp.ndarray) -> jnp.ndarray:
    """
    Interpolate funding spreads.
    
    Args:
        times: Original time points
        spreads: Original funding spreads
        new_times: New time points to interpolate at
        
    Returns:
        Interpolated funding spreads
    """
    return interpolate_1d(times, spreads, new_times)

@partial(jit, static_argnums=(0,))
def interpolate_exposure_profile(times: jnp.ndarray,
                               exposures: jnp.ndarray,
                               new_times: jnp.ndarray) -> jnp.ndarray:
    """
    Interpolate exposure profile.
    
    Args:
        times: Original time points
        exposures: Original exposure values
        new_times: New time points to interpolate at
        
    Returns:
        Interpolated exposure profile
    """
    return interpolate_1d(times, exposures, new_times)

@partial(jit, static_argnums=(0,))
def interpolate_initial_margin(times: jnp.ndarray,
                             im_values: jnp.ndarray,
                             new_times: jnp.ndarray) -> jnp.ndarray:
    """
    Interpolate initial margin profile.
    
    Args:
        times: Original time points
        im_values: Original initial margin values
        new_times: New time points to interpolate at
        
    Returns:
        Interpolated initial margin profile
    """
    return interpolate_1d(times, im_values, new_times)

@partial(jit, static_argnums=(0,))
def interpolate_capital_profile(times: jnp.ndarray,
                              capital_values: jnp.ndarray,
                              new_times: jnp.ndarray) -> jnp.ndarray:
    """
    Interpolate regulatory capital profile.
    
    Args:
        times: Original time points
        capital_values: Original capital values
        new_times: New time points to interpolate at
        
    Returns:
        Interpolated capital profile
    """
    return interpolate_1d(times, capital_values, new_times)
