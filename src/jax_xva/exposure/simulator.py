
from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import jit, vmap

from jax_xva.market_data import MarketData
from jax_xva.engine import SimulationConfig

class ExposureSimulator:
    """
    Class for simulating exposures for various trade types.
    """
    
    def __init__(self, market_data: MarketData, simulation_config: SimulationConfig):
        self.market_data = market_data
        self.config = simulation_config
        
    @partial(jit, static_argnums=(0,))
    def simulate_interest_rate_swap(self, key, trade_data, hull_white_model) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulate exposure paths for an interest rate swap.
        
        Args:
            key: JAX random key
            trade_data: Swap trade data
            hull_white_model: Hull-White model for rate simulation
            
        Returns:
            Tuple of (exposure paths, simulation times)
        """
        # Create time grid
        dt = self.config.time_horizon / self.config.n_time_steps
        times = jnp.linspace(0.0, self.config.time_horizon, self.config.n_time_steps + 1)
        
        # Filter times to trade lifetime
        valid_times = (times >= trade_data.start_date) & (times <= trade_data.end_date)
        valid_indices = jnp.where(valid_times, size=jnp.sum(valid_times))[0]
        sim_times = times[valid_indices]
        
        # Get initial rate from discount curve
        initial_rate = -jnp.log(self.market_data.discount_curve[1.0]) / 1.0
        
        # Simulate rate paths
        rate_paths = hull_white_model.simulate_paths(key, initial_rate, times, self.config.n_paths)
        
        # Define swap pricing function
        def price_swap(rates, valuation_time, fixed_rate=0.02, payment_frequency=0.5):
            """Calculate swap price given simulated rates."""
            remaining_time = trade_data.end_date - valuation_time
            if remaining_time <= 0:
                return jnp.zeros(rates.shape[0])
            
            # Number of remaining payments
            n_payments = jnp.ceil(remaining_time / payment_frequency).astype(int)
            
            # Payment times
            payment_times = jnp.arange(1, n_payments + 1) * payment_frequency + valuation_time
            payment_times = payment_times[payment_times <= trade_data.end_date]
            
            # Calculate discount factors using simulated rates
            # Simplified: use average rate for each path as constant discount rate
            avg_rates = jnp.mean(rates, axis=1)
            
            # Calculate swap value for each path
            swap_values = jnp.zeros(rates.shape[0])
            
            def calc_payment_value(payment_time, avg_rate):
                # Simplified pricing: fixed leg - floating leg
                df = jnp.exp(-avg_rate * (payment_time - valuation_time))
                fixed_payment = fixed_rate * payment_frequency * trade_data.notional
                floating_payment = avg_rate * payment_frequency * trade_data.notional
                return (fixed_payment - floating_payment) * df
            
            # Sum all payment values for each path
            for payment_time in payment_times:
                swap_values += vmap(lambda r: calc_payment_value(payment_time, r))(avg_rates)
            
            return swap_values
        
        # Calculate exposure paths
        exposure_paths = jnp.zeros((self.config.n_paths, len(sim_times)))
        
        for i, time_idx in enumerate(valid_indices):
            val_time = times[time_idx]
            # Use rates up to current time for pricing
            current_rates = rate_paths[:, :time_idx+1]
            swap_values = price_swap(current_rates, val_time)
            # Positive exposure only (max with zero) for CVA calculation
            exposure_paths = exposure_paths.at[:, i].set(jnp.maximum(swap_values, 0.0))
            
        return exposure_paths, sim_times
    
    @partial(jit, static_argnums=(0,))
    def expected_exposure(self, exposure_paths: jnp.ndarray) -> jnp.ndarray:
        """Calculate expected exposure from exposure paths."""
        return jnp.mean(exposure_paths, axis=0)
    
    @partial(jit, static_argnums=(0,))
    def potential_future_exposure(self, exposure_paths: jnp.ndarray, quantile: float = 0.95) -> jnp.ndarray:
        """Calculate potential future exposure at given quantile."""
        return jnp.quantile(exposure_paths, quantile, axis=0)