import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from jax_xva import (
    MarketData, TradeData, SwapTradeData, SimulationConfig,
    HullWhiteModel, BlackScholesModel, ExposureCalculator,
    ExposureSimulator, CVA, DVA, FVA, MVA, KVA, XVAEngine,
    create_sample_market_data
)

def main():
    # Set random seed for reproducibility
    key = random.PRNGKey(42)
    
    # Step 1: Create market data
    print("Creating market data...")
    market_data = create_sample_market_data()
    
    # Step 2: Create simulation configuration
    print("Setting up simulation...")
    sim_config = SimulationConfig(
        n_paths=1000,
        n_time_steps=40,
        time_horizon=5.0,
        random_seed=42
    )
    
    # Step 3: Create models
    print("Creating market models...")
    models = {
        'interest_rate': HullWhiteModel(mean_reversion=0.03, volatility=0.01)
    }
    
    # Step 4: Create trades
    print("Creating sample portfolio...")
    trades = [
        SwapTradeData(
            trade_id="Swap_1",
            counterparty="Counterparty_A",
            start_date=0.0,
            end_date=3.0,
            notional=1_000_000.0,
            fixed_rate=0.025,
            payment_frequency=0.5,
            is_payer=True
        ),
        SwapTradeData(
            trade_id="Swap_2",
            counterparty="Counterparty_A",
            start_date=0.5,
            end_date=5.0,
            notional=2_000_000.0,
            fixed_rate=0.028,
            payment_frequency=0.5,
            is_payer=False
        )
    ]
    
    # Step 5: Create XVA Engine
    print("Initializing XVA Engine...")
    xva_engine = XVAEngine(market_data, sim_config)
    
    # Step 6: Simulate exposures
    print("Simulating portfolio exposures...")
    key, subkey = random.split(key)
    exposures = xva_engine.simulate_portfolio_exposures(subkey, trades, models)
    
    # Step 7: Create netting sets
    netting_sets = {"NS_1": ["Swap_1", "Swap_2"]}
    
    # Step 8: Aggregate exposures within netting sets
    print("Aggregating exposures within netting sets...")
    aggregated_exposures = xva_engine.aggregate_exposures(exposures, netting_sets)
    
    # Step 9: Calculate XVA components
    print("Calculating XVA components...")
    xva_results = xva_engine.calculate_portfolio_xva(
        counterparty="Counterparty_A",
        own_entity="Bank",
        exposures=aggregated_exposures,
        hurdle_rate=0.12,
        funding_spread=0.006
    )
    
    # Print results
    print("\nXVA Results:")
    print(f"CVA:  ${xva_results['CVA']:.2f}")
    print(f"DVA:  ${xva_results['DVA']:.2f}")
    print(f"FVA:  ${xva_results['FVA']:.2f}")
    print(f"MVA:  ${xva_results['MVA']:.2f}")
    print(f"KVA:  ${xva_results['KVA']:.2f}")
    print(f"Total XVA: ${xva_results['TotalXVA']:.2f}")
    
    # Step 10: Calculate XVA sensitivities
    print("\nCalculating XVA sensitivities...")
    params = {
        'hurdle_rate': 0.12
    }
    
    sensitivities = xva_engine.calculate_xva_sensitivities(
        counterparty="Counterparty_A",
        own_entity="Bank",
        exposures=aggregated_exposures,
        params=params
    )
    
    print("\nXVA Sensitivities:")
    for param, sensitivity in sensitivities.items():
        print(f"{param}: {sensitivity:.2f}")
    
    # Step 11: Visualize exposure profiles
    plot_exposure_profiles(aggregated_exposures)


def plot_exposure_profiles(exposures):
    """Plot exposure profiles for visualization."""
    plt.figure(figsize=(12, 8))
    
    for netting_set, data in exposures.items():
        times = np.array(data['times'])
        ee = np.array(data['ee'])
        pfe = np.array(data['pfe'])
        
        plt.plot(times, ee, label=f"{netting_set} - EE")
        plt.plot(times, pfe, '--', label=f"{netting_set} - PFE (95%)")
    
    plt.title("Exposure Profiles")
    plt.xlabel("Time (years)")
    plt.ylabel("Exposure ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_xva_stress_test():
    """Run a simple XVA stress test by shocking market parameters."""
    print("\nRunning XVA stress test...")
    
    # Create base market data
    market_data = create_sample_market_data()
    
    # Define stress scenarios
    scenarios = {
        "Base": {
            "hazard_multiplier": 1.0,
            "funding_spread_multiplier": 1.0
        },
        "Credit Stress": {
            "hazard_multiplier": 2.0,
            "funding_spread_multiplier": 1.5
        },
        "Funding Stress": {
            "hazard_multiplier": 1.2,
            "funding_spread_multiplier": 3.0
        },
        "Combined Stress": {
            "hazard_multiplier": 2.5,
            "funding_spread_multiplier": 2.5
        }
    }
    
    # Sample trades
    trades = [
        SwapTradeData(
            trade_id="Swap_1",
            counterparty="Counterparty_A",
            start_date=0.0,
            end_date=5.0,
            notional=10_000_000.0,
            fixed_rate=0.025,
            payment_frequency=0.5,
            is_payer=True
        )
    ]
    
    # Run scenarios
    results = {}
    
    for name, params in scenarios.items():
        print(f"\nRunning scenario: {name}")
        
        # Apply stress to market data
        stressed_market_data = apply_stress(market_data, params)
        
        # Configure simulation
        sim_config = SimulationConfig(n_paths=500, n_time_steps=20, time_horizon=5.0)
        
        # Create models
        models = {
            'interest_rate': HullWhiteModel(mean_reversion=0.03, volatility=0.01)
        }
        
        # Create XVA engine
        xva_engine = XVAEngine(stressed_market_data, sim_config)
        
        # Simulate exposures
        exposures = xva_engine.simulate_portfolio_exposures(
            random.PRNGKey(42), trades, models)
        
        # Define netting sets
        netting_sets = {"NS_1": ["Swap_1"]}
        
        # Aggregate exposures
        aggregated = xva_engine.aggregate_exposures(exposures, netting_sets)
        
        # Calculate XVA
        xva_results = xva_engine.calculate_portfolio_xva(
            counterparty="Counterparty_A",
            own_entity="Bank",
            exposures=aggregated
        )
        
        # Store results
        results[name] = xva_results
    
    # Print comparison
    print("\nXVA Stress Test Results:")
    print("------------------------")
    print(f"{'Scenario':<15} {'CVA':<10} {'FVA':<10} {'Total XVA':<10}")
    print("-" * 45)
    
    for name, res in results.items():
        print(f"{name:<15} {res['CVA']:<10.2f} {res['FVA']:<10.2f} {res['TotalXVA']:<10.2f}")


def apply_stress(market_data, stress_params):
    """Apply stress to market data."""
    # Create a copy of market data
    stressed_data = MarketData(
        valuation_date=market_data.valuation_date,
        discount_curve=market_data.discount_curve.copy(),
        hazard_rates={k: v.copy() for k, v in market_data.hazard_rates.items()},
        recovery_rates=market_data.recovery_rates.copy(),
        funding_spreads=market_data.funding_spreads.copy()
    )
    
    # Apply hazard rate stress
    for entity in stressed_data.hazard_rates:
        for tenor in stressed_data.hazard_rates[entity]:
            stressed_data.hazard_rates[entity][tenor] *= stress_params["hazard_multiplier"]
    
    # Apply funding spread stress
    for tenor in stressed_data.funding_spreads:
        stressed_data.funding_spreads[tenor] *= stress_params["funding_spread_multiplier"]
    
    return stressed_data


if __name__ == "__main__":
    main()
    # Uncomment to run stress test
    # run_xva_stress_test()