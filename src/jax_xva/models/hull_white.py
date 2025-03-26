from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, random

class HullWhiteModel:
    """One-factor Hull-White interest rate model."""
    
    def __init__(self, mean_reversion: float, volatility: float):
        self.mean_reversion = mean_reversion
        self.volatility = volatility
    
    @partial(jit, static_argnums=(0,))
    def simulate_paths(self, key, initial_rate: float, times: jnp.ndarray, n_paths: int) -> jnp.ndarray:
        """
        Simulate interest rate paths using Euler discretization.
        
        Args:
            key: JAX random key
            initial_rate: Initial short rate
            times: Time grid for simulation
            n_paths: Number of paths to simulate
            
        Returns:
            Array of shape (n_paths, len(times)) containing simulated rates
        """
        dt = jnp.diff(times, prepend=0.0)[1:]
        sqrt_dt = jnp.sqrt(dt)
        n_steps = len(times) - 1
        
        # Generate standard normal random variables
        subkeys = random.split(key, n_steps)
        dW = jnp.zeros((n_paths, n_steps))
        
        for i in range(n_steps):
            dW = dW.at[:, i].set(random.normal(subkeys[i], (n_paths,)) * sqrt_dt[i])
        
        # Initialize paths
        paths = jnp.zeros((n_paths, len(times)))
        paths = paths.at[:, 0].set(initial_rate)
        
        # Simulate paths using vectorized operations
        def update_step(i, paths):
            r_t = paths[:, i]
            drift = self.mean_reversion * (0.0 - r_t) * dt[i]
            diffusion = self.volatility * dW[:, i]
            paths = paths.at[:, i+1].set(r_t + drift + diffusion)
            return paths
        
        paths = jax.lax.fori_loop(0, n_steps, update_step, paths)
        return paths