
from functools import partial
from dataclasses import dataclass
from typing import Dict, List, Union

import jax
import jax.numpy as jnp
from jax import jit, random

from jax_xva.market_data import MarketData
from jax_xva.trades.base import TradeData
from jax_xva.trades.swap import SwapTradeData
from jax_xva.trades.option import OptionTradeData
from jax_xva.models.hull_white import HullWhiteModel
from jax_xva.models.black_scholes import BlackScholesModel
from jax_xva.exposure import ExposureCalculator, ExposureSimulator
from jax_xva.xva.cva import CVA
from jax_xva.xva.dva import DVA
from jax_xva.xva.fva import FVA
from jax_xva.xva.mva import MVA
from jax_xva.xva.kva import KVA
from jax_xva.xva.colva import ColVA


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulations."""
    n_paths: int
    n_time_steps: int
    time_horizon: float
    random_seed: int = 42


class XVAEngine:
    """
    Main XVA engine that orchestrates the calculation of all XVA components.
    """

    def __init__(self, market_data: MarketData, simulation_config: SimulationConfig):
        # Set up components
        self.market_data = market_data
        self.config = simulation_config
        self.exposure_calculator = ExposureCalculator(market_data)
        self.exposure_simulator = ExposureSimulator(market_data, simulation_config)

        # Initialize XVA calculators
        self.cva_calculator = CVA(market_data, self.exposure_calculator)
        self.dva_calculator = DVA(market_data, self.exposure_calculator)
        self.fva_calculator = FVA(market_data, self.exposure_calculator)
        self.mva_calculator = MVA(market_data, self.exposure_calculator)
        self.kva_calculator = KVA(market_data, self.exposure_calculator)
        self.colva_calculator = ColVA(market_data, self.exposure_calculator)

    def simulate_portfolio_exposures(self, key, trades: List[TradeData], 
                            models: Dict[str, Union[HullWhiteModel, BlackScholesModel]]) -> Dict:
        """
        Simulate exposures for a portfolio of trades.
        
        Args:
            key: JAX random key
            trades: List of trade data objects
            models: Dictionary of market models for each underlying
            
        Returns:
            Dictionary with exposure profiles
        """
        results = {}
        subkeys = random.split(key, len(trades))
        
        # Simulate each trade
        for i, trade in enumerate(trades):
            if isinstance(trade, SwapTradeData):
                exposure_paths, times = self.exposure_simulator.simulate_interest_rate_swap(
                    subkeys[i], trade, models['interest_rate'])
            elif isinstance(trade, OptionTradeData):
                # Implementation for option exposure simulation would go here
                # TODO: implement option exposure simulation
                raise NotImplementedError("Option exposure simulation not implemented (yet)")
            else:
                raise ValueError(f"Unsupported trade type: {type(trade)}")
            
            # Calculate exposure profiles
            ee = self.exposure_simulator.expected_exposure(exposure_paths)
            pfe = self.exposure_simulator.potential_future_exposure(exposure_paths)
            
            # Store results
            results[trade.trade_id] = {
                'exposure_paths': exposure_paths,
                'times': times,
                'ee': ee,
                'pfe': pfe
            }
        
        return results
    
    @partial(jit, static_argnums=(0,))
    def aggregate_exposures(self, exposures: Dict[str, Dict], 
                           netting_sets: Dict[str, List[str]]) -> Dict:
        """
        Aggregate exposures within netting sets.
        
        Args:
            exposures: Dictionary with exposure results per trade
            netting_sets: Dictionary mapping netting set IDs to trade IDs
            
        Returns:
            Dictionary with aggregated exposures per netting set
        """
        aggregated = {}
        
        for netting_set, trade_ids in netting_sets.items():
            # Get exposures for trades in this netting set
            trade_exposures = [exposures[tid] for tid in trade_ids if tid in exposures]
            
            if not trade_exposures:
                continue
            
            # Ensure all trades use the same time grid
            # In practice, this requires interpolation
            times = trade_exposures[0]['times']
            
            # Sum exposures across trades (simplified - should consider netting)
            total_exposure_paths = sum(te['exposure_paths'] for te in trade_exposures)
            
            # Calculate netted profiles
            ee = self.exposure_simulator.expected_exposure(total_exposure_paths)
            pfe = self.exposure_simulator.potential_future_exposure(total_exposure_paths)
            
            # Store results
            aggregated[netting_set] = {
                'exposure_paths': total_exposure_paths,
                'times': times,
                'ee': ee,
                'pfe': pfe
            }
        
        return aggregated
    
    def calculate_portfolio_xva(self, counterparty: str, own_entity: str,
                               exposures: Dict[str, Dict], 
                               hurdle_rate: float = 0.1,
                               funding_spread: float = 0.005) -> Dict[str, float]:
        """
        Calculate all XVA components for a portfolio.
        
        Args:
            counterparty: Counterparty name
            own_entity: Own entity name
            exposures: Dictionary with exposure profiles
            hurdle_rate: Hurdle rate for KVA calculation
            funding_spread: Funding spread for MVA calculation
            
        Returns:
            Dictionary with XVA components
        """
        results = {}
        
        # Sum EE profiles across netting sets
        total_ee = None
        times = None
        
        for ns, data in exposures.items():
            if total_ee is None:
                total_ee = data['ee']
                times = data['times']
            else:
                # In practice, this requires time interpolation # TODO: implement time interpolation
                total_ee += data['ee']
        
        # Calculate negative exposures (for DVA)
        negative_ee = -jnp.minimum(total_ee, 0.0)
        positive_ee = jnp.maximum(total_ee, 0.0)
        
        # Calculate PFE for IM calculation
        pfe = sum(data['pfe'] for data in exposures.values())
        
        # Calculate initial margin profile
        im_profile = self.mva_calculator.simulate_im_profile(
            random.PRNGKey(42), pfe)
        
        # Calculate capital profile
        capital_profile = self.kva_calculator.calculate_capital_profile(
            positive_ee, counterparty)
        
        # Calculate XVA components
        results['CVA'] = self.cva_calculator.calculate_cva(
            counterparty, positive_ee, times)
        
        results['DVA'] = self.dva_calculator.calculate_dva(
            own_entity, negative_ee, times)
        
        results['FVA'] = self.fva_calculator.calculate_fva(
            counterparty, own_entity, positive_ee, times)
        
        results['MVA'] = self.mva_calculator.calculate_mva(
            im_profile, times, funding_spread)
        
        results['KVA'] = self.kva_calculator.calculate_kva(
            capital_profile, times, hurdle_rate)
        
        # Calculate total XVA
        results['TotalXVA'] = results['CVA'] - results['DVA'] +\
              results['FVA'] + results['MVA'] + results['KVA']
        
        return results
    
    def calculate_xva_sensitivities(self, counterparty: str, own_entity: str,
                                   exposures: Dict[str, Dict],
                                   params: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Calculate XVA sensitivities using JAX auto-differentiation.
        
        Args:
            counterparty: Counterparty name
            own_entity: Own entity name
            exposures: Dictionary of exposure profiles
            params: Dictionary of parameters to calculate sensitivities for
            
        Returns:
            Dictionary of XVA sensitivities
        """
        # Define function for AD
        def xva_calculation_wrapper(hazard_rate_cpty, hazard_rate_own, funding_spread, hurdle_rate):
            # Update market data with perturbed parameters
            updated_market_data = self.market_data
            
            # Update hazard rates
            for entity, times_dict in updated_market_data.hazard_rates.items():
                if entity == counterparty:
                    for t in times_dict:
                        times_dict[t] = hazard_rate_cpty
                elif entity == own_entity:
                    for t in times_dict:
                        times_dict[t] = hazard_rate_own
            
            # Update funding spreads
            for t in updated_market_data.funding_spreads:
                updated_market_data.funding_spreads[t] = funding_spread
            
            # Recalculate XVA
            xva_engine = XVAEngine(updated_market_data, self.config)
            xva_results = xva_engine.calculate_portfolio_xva(
                counterparty, own_entity, exposures, hurdle_rate, funding_spread)
            
            return xva_results['TotalXVA']
        
        # Get initial parameter values
        hazard_rate_cpty = jnp.mean(jnp.array([v for v in self.market_data.hazard_rates[counterparty].values()]))
        hazard_rate_own = jnp.mean(jnp.array([v for v in self.market_data.hazard_rates[own_entity].values()]))
        funding_spread = jnp.mean(jnp.array([v for v in self.market_data.funding_spreads.values()]))
        hurdle_rate = params.get('hurdle_rate', 0.1)
        
        # Create gradient function
        grad_fn = jax.grad(xva_calculation_wrapper, argnums=(0, 1, 2, 3))
        
        # Calculate gradients
        grads = grad_fn(hazard_rate_cpty, hazard_rate_own, funding_spread, hurdle_rate)
        
        # Package results
        sensitivities = {
            'hazard_rate_cpty': grads[0],
            'hazard_rate_own': grads[1],
            'funding_spread': grads[2],
            'hurdle_rate': grads[3]
        }
        
        return sensitivities