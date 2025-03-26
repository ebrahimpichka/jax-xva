
from typing import Optional

import jax
import jax.numpy as jnp
from jax import jit, random
from functools import partial


class BlackScholesModel:
    """Black-Scholes model for equity or FX."""
    
    def __init__(self, volatility: float, drift: Optional[float] = None):
        self.volatility = volatility
        self.drift = drift
    
    @partial(jit, static_argnums=(0,))
    def simulate_paths(self, key, initial_price: float, times: jnp.ndarray, n_paths: int,
                       interest_rate: Optional[float] = None) -> jnp.ndarray:
        """
        Simulate price paths using exact solution.
        
        Args:
            key: JAX random key
            initial_price: Initial asset price
            times: Time grid for simulation
            n_paths: Number of paths to simulate
            interest_rate: Risk-free rate (if not provided, use self.drift)
            
        Returns:
            Array of shape (n_paths, len(times)) containing simulated prices
        """
        dt = jnp.diff(times, prepend=0.0)
        n_steps = len(times)
        
        # Use provided interest rate or drift
        mu = interest_rate if interest_rate is not None else self.drift
        if mu is None:
            mu = 0.0
        
        # Generate correlated normal random variables
        key, subkey = random.split(key)
        Z = random.normal(subkey, (n_paths, n_steps))
        
        # Calculate paths using vectorized operations
        # S_t = S_0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t)
        t = times.reshape(1, -1)  # Shape (1, n_steps)
        
        # Cumulative Brownian motion (taking into account time increments)
        W = jnp.zeros((n_paths, n_steps))
        for i in range(1, n_steps):
            W = W.at[:, i].set(W[:, i-1] + Z[:, i] * jnp.sqrt(dt[i]))
        
        drift_term = (mu - 0.5 * self.volatility**2) * t
        diffusion_term = self.volatility * W
        
        return initial_price * jnp.exp(drift_term + diffusion_term)