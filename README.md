# JAX XVA Library

A high-performance library for XVA (Valuation Adjustments) calculations using JAX.

## Overview

This library provides efficient implementations of various XVA components using JAX's high-performance features:

- **Just-in-time (JIT) compilation**: Accelerates calculations by compiling functions
- **Automatic differentiation**: Enables efficient calculation of sensitivities and Greeks
- **Vectorization**: Fast parallel computation of simulations and valuation steps
- **GPU/TPU acceleration**: Transparent hardware acceleration for compute-intensive tasks

## Key Features (In Progress)

- Comprehensive XVA calculations (CVA, DVA, FVA, MVA, KVA)
- Exposure simulation for various trade types
- Efficient Monte Carlo simulation
- Portfolio aggregation with netting
- Risk sensitivities calculation
- Performance benchmarking capabilities

## Components

### XVA Components (In Progress)

- **CVA (Credit Valuation Adjustment)**: Adjustment for counterparty credit risk
- **DVA (Debt Valuation Adjustment)**: Adjustment for own credit risk
- **FVA (Funding Valuation Adjustment)**: Adjustment for funding costs
- **MVA (Margin Valuation Adjustment)**: Adjustment for initial margin costs
- **KVA (Capital Valuation Adjustment)**: Adjustment for capital costs
- **ColVA (Collateral Valuation Adjustment)**: Adjustment for collateral costs/benefits

### Market Models

- **Hull-White Model**: One-factor interest rate model for simulating rate paths
- **Black-Scholes Model**: Model for equity and FX underlyings

### Exposure Calculation

- **Exposure Simulator**: Simulates future exposure profiles
- **Exposure Calculator**: Utilities for exposure calculations and interpolation

### Risk Management Tools

- **XVA Engine**: Main engine orchestrating XVA calculations
- **Sensitivity Analysis**: Calculation of risk sensitivities using auto-differentiation
- **Stress Testing**: Framework for running stress scenarios

## Installation

```bash
pip install jax-xva
```

## Quick Start

```python
import jax
import jax.numpy as jnp
from jax import random

from jax_xva_library import (
    MarketData, SwapTradeData, SimulationConfig,
    HullWhiteModel, XVAEngine, create_sample_market_data
)

# Create market data
market_data = create_sample_market_data()

# Configure simulation
sim_config = SimulationConfig(
    n_paths=1000,
    n_time_steps=40,
    time_horizon=5.0,
    random_seed=42
)

# Create models
models = {
    'interest_rate': HullWhiteModel(mean_reversion=0.03, volatility=0.01)
}

# Define trades
trades = [
    SwapTradeData(
        trade_id="Swap_1",
        counterparty="Counterparty_A",
        start_date=0.0,
        end_date=5.0,
        notional=1_000_000.0,
        fixed_rate=0.025,
        payment_frequency=0.5,
        is_payer=True
    )
]

# Create XVA engine
xva_engine = XVAEngine(market_data, sim_config)

# Simulate exposures
exposures = xva_engine.simulate_portfolio_exposures(
    random.PRNGKey(42), trades, models)

# Define netting sets
netting_sets = {"NS_1": ["Swap_1"]}

# Aggregate exposures
aggregated = xva_engine.aggregate_exposures(exposures, netting_sets)

# Calculate XVA components
xva_results = xva_engine.calculate_portfolio_xva(
    counterparty="Counterparty_A",
    own_entity="Bank",
    exposures=aggregated
)

# Print results
print(f"CVA: ${xva_results['CVA']:.2f}")
print(f"FVA: ${xva_results['FVA']:.2f}")
print(f"Total XVA: ${xva_results['TotalXVA']:.2f}")
```

## Performance

The JAX XVA library significantly outperforms traditional implementations:

- 10-50x speedup for exposure simulation compared to NumPy
- Efficient GPU/TPU utilization for large portfolios
- Scalable to millions of simulation paths
- Fast risk sensitivity calculations using auto-differentiation

## Advanced Usage

### Custom Trade Types

Extend the library with custom trade types by inheriting from `TradeData`:

```python
@dataclass
class CustomOptionTradeData(TradeData):
    strike: float
    underlying: str
    is_call: bool
    barrier_level: Optional[float] = None
```

### Custom Market Models

Implement custom market models by defining a class with a `simulate_paths` method:

```python
class CustomModel:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    
    @partial(jit, static_argnums=(0,))
    def simulate_paths(self, key, initial_value, times, n_paths):
        # Simulation logic here
        return paths
```

### Using GPU Acceleration

Enable GPU acceleration (if available) with:

```python
jax.config.update('jax_platform_name', 'gpu')
```

## Requirements

- JAX >= 0.5.3
- NumPy >= 2.2.4
- Matplotlib >= 3.10.1 (for visualization)

## License

MIT License

## Citation

If you use this library in your research, please cite:

```
@software{jax_xva_library,
  author = {XVA Library Contributors},
  title = {JAX XVA: High-Performance XVA Calculations with JAX},
  year = {2025},
  url = {https://github.com/ebrahimpichka/jax-xva}
}
```