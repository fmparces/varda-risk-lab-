"""
Varda: Capital Markets Network Risk Lab

Varda is an AI-powered capital markets risk lab for ECM, DCM, and Leveraged Finance.

It models how credit, liquidity, and systemic risk propagate across issuers, deals,
syndicate banks, and investors, under different market regimes.

Core Capital Markets Capabilities:
- Deal-aware network modeling of issuers, tranches, and syndicate exposures
- Monte Carlo loss distributions for ECM/DCM/LevFin underwriting and pipelines
- Markov chain rating and risk-state transitions for bonds, loans, and issuers
- Scenario-based spread, PD, and refi-wall analysis under macro/market constraints
- Network analytics to identify systemic issuers, sponsors, and investor hubs
- Pipeline-level P&L and fee-at-risk across ECM, DCM, and LevFin franchises

One-liner for decks/emails:
"Varda is a complexity lab for ECM, DCM, and LevFin desks, letting them stress test
deals, issuers, and syndicate networks under realistic market regimes—the way big
banks run internal risk engines."
"""

from __future__ import annotations

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Import base functionality from financial_risk_lab
from financial_risk_lab import (  # type: ignore
    Entity, Relationship, MarketConstraint, MarketState, MarkovChain,
    RiskType, create_credit_rating_chain, create_risk_state_chain,
    create_market_regime_chain
)


class DealType(Enum):
    """Types of capital markets deals."""
    ECM_IPO = "ecm_ipo"
    ECM_FOLLOW_ON = "ecm_follow_on"
    DCM_IG = "dcm_investment_grade"
    DCM_HY = "dcm_high_yield"
    LEVFIN_LBO = "levfin_lbo"
    LEVFIN_REF = "levfin_refinancing"


@dataclass
class Tranche:
    """
    Represents a capital markets tranche (bond, loan, equity slice).

    For DCM/LevFin, PD/LGD/EAD are key for loss; for ECM, PD~0 but
    you care about price support / stabilization and overhang.
    """
    id: str
    deal_id: str
    currency: str
    notional: float       # EAD
    coupon: float         # as % (e.g., 0.05 = 5%)
    spread_bps: float     # vs benchmark
    maturity_years: float
    rating: str
    pd_annual: float      # annual default probability (approx)
    lgd: float            # loss given default (0-1)
    is_secured: bool = False
    seniority: str = "senior"  # senior, mezz, junior, equity
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate tranche data."""
        if self.notional < 0:
            warnings.warn(f"Notional for {self.id} should be >= 0, got {self.notional}")
        if self.spread_bps < 0:
            warnings.warn(f"Spread for {self.id} should be >= 0 bps, got {self.spread_bps}")
        if not 0.0 <= self.pd_annual <= 1.0:
            warnings.warn(f"PD for {self.id} should be in [0, 1], got {self.pd_annual}")
        if not 0.0 <= self.lgd <= 1.0:
            warnings.warn(f"LGD for {self.id} should be in [0, 1], got {self.lgd}")


@dataclass
class CapitalMarketsDeal:
    """
    Represents an ECM/DCM/LevFin transaction.

    This ties your generic Entity/Relationship world to concrete
    capital-markets objects (underwriting, fees, tranches).
    """
    id: str
    issuer_entity_id: str
    deal_type: DealType
    tranches: List[Tranche]
    bookrunners: List[str]       # entity_ids of banks
    co_managers: List[str] = field(default_factory=list)
    issue_date: Optional[pd.Timestamp] = None
    tenor_years: Optional[float] = None
    gross_fees: float = 0.0      # total fee pool in currency
    bank_share: Dict[str, float] = field(default_factory=dict)  # bank_id -> fee share (0-1)
    pipeline_stage: str = "mandated"   # idea, mandated, launched, priced, closed
    sector: Optional[str] = None
    region: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate deal data."""
        if self.bank_share:
            total_share = sum(self.bank_share.values())
            if abs(total_share - 1.0) > 0.01:  # Allow small rounding errors
                warnings.warn(f"Bank share sum should be ~1.0, got {total_share}")


@dataclass
class CapitalMarketsScenario:
    """
    Scenario for capital markets analysis.

    Combines macro/market constraints with spread/PD shocks and valuation inputs.

    Typical usage in a big-bank style workflow:
    - Derive pd_multiplier from a market regime Markov chain using
      Varda.calibrate_pd_multiplier_from_regime(...)
    - Attach MarketConstraint objects representing macro/market stress.
    """
    name: str
    description: str
    market_constraints: List[MarketConstraint] = field(default_factory=list)
    spread_shock_bps: float = 0.0          # parallel credit spread shock
    pd_multiplier: float = 1.0             # multiply PDs by this factor
    equity_vol_multiplier: float = 1.0     # for ECM price paths
    discount_rate_shift_bps: float = 0.0   # shift in discount rate
    horizon_years: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Varda:
    """
    Main Varda platform class for capital markets risk modeling and simulation.
    
    Varda models systemic risk, contagion, and credit scenarios for ECM, DCM, and LevFin using:
    - Network models to represent entity relationships (issuers, banks, investors)
    - Deal-aware modeling of capital markets transactions
    - Fluid dynamics metaphors for risk propagation
    - Markov chain models for state transitions (credit ratings, risk states)
    - Monte Carlo simulations for scenario analysis and loss distributions
    - Pipeline-level fee-at-risk and P&L analytics

    Outputs can be mapped to standard risk measures (EL, VaR, ES) and used to
    support internal stress tests and capital planning (e.g., CCAR / ICAAP / IFRS9 / CECL inputs).
    """
    
    def __init__(self, name: str = "Varda Capital Markets Risk Lab") -> None:
        """
        Initialize the Varda platform.
        
        Args:
            name: Name identifier for this Varda instance
        """
        self.name = name
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.simulation_history: List[Dict[str, Any]] = []
        self.markov_chains: Dict[str, MarkovChain] = {}
        # NOTE: For full production use you may want per-chain states:
        # Dict[chain_name, Dict[entity_id, state]]
        self.entity_states: Dict[str, str] = {}  # Track current state for each entity
        self.market_states: Dict[str, MarketState] = {}  # Market states as entities
        self.market_constraints: List[MarketConstraint] = []  # Global market constraints
        
        # Capital markets extensions
        self.deals: Dict[str, CapitalMarketsDeal] = {}
        self.tranches: Dict[str, Tranche] = {}
        
    # -------------------------------------------------------------------------
    # Core entity / network methods
    # -------------------------------------------------------------------------
    def add_entity(self, entity: Entity, initial_state: Optional[str] = None) -> None:
        """
        Add an entity to the network.
        
        Args:
            entity: Entity to add
            initial_state: Initial state for Markov chain modeling (optional)
        """
        self.entities[entity.id] = entity
        if initial_state is not None:
            self.entity_states[entity.id] = initial_state
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship between entities."""
        if relationship.source_id not in self.entities:
            raise ValueError(f"Source entity {relationship.source_id} not found")
        if relationship.target_id not in self.entities:
            raise ValueError(f"Target entity {relationship.target_id} not found")
        self.relationships.append(relationship)
    
    def add_deal(self, deal: CapitalMarketsDeal) -> None:
        """Register a capital markets deal and its tranches."""
        if deal.issuer_entity_id not in self.entities:
            raise ValueError(f"Issuer entity {deal.issuer_entity_id} not found in Varda.entities")
        
        self.deals[deal.id] = deal
        for tranche in deal.tranches:
            if tranche.id in self.tranches:
                warnings.warn(f"Tranche {tranche.id} already exists. Overwriting.")
            self.tranches[tranche.id] = tranche
    
    def get_deals_by_type(self, deal_type: DealType) -> List[CapitalMarketsDeal]:
        """Get all deals of a specific type."""
        return [d for d in self.deals.values() if d.deal_type == deal_type]
    
    def get_network_adjacency(self) -> pd.DataFrame:
        """
        Build adjacency matrix representing the entity network.
        
        Returns:
            DataFrame with entities as rows/columns and relationship strengths as values
        """
        entity_ids = list(self.entities.keys())
        n = len(entity_ids)
        adj_matrix = np.zeros((n, n))
        
        id_to_idx = {entity_id: idx for idx, entity_id in enumerate(entity_ids)}
        
        for rel in self.relationships:
            source_idx = id_to_idx[rel.source_id]
            target_idx = id_to_idx[rel.target_id]
            adj_matrix[source_idx, target_idx] = rel.strength
            
        return pd.DataFrame(adj_matrix, index=entity_ids, columns=entity_ids)
    
    # -------------------------------------------------------------------------
    # Risk propagation on networks
    # -------------------------------------------------------------------------
    def propagate_risk_fluid(
        self,
        initial_shock: Optional[Dict[str, float]] = None,
        diffusion_rate: float = 0.1,
        iterations: int = 10
    ) -> pd.DataFrame:
        """
        Simulate risk propagation using fluid dynamics-inspired diffusion model.

        Risk flows through the network like a fluid, with diffusion based on
        relationship strengths and connection topology.

        Args:
            initial_shock: Dict mapping entity_id to initial risk shock value
            diffusion_rate: Rate at which risk diffuses through connections (0-1)
            iterations: Number of propagation steps
            
        Returns:
            DataFrame with risk levels for each entity at each iteration
        """
        entity_ids = list(self.entities.keys())
        n = len(entity_ids)
        
        # Initialize risk levels
        risk_levels = np.zeros((iterations + 1, n))
        
        # Set initial conditions
        if initial_shock:
            for idx, entity_id in enumerate(entity_ids):
                risk_levels[0, idx] = initial_shock.get(entity_id, self.entities[entity_id].initial_risk_score)
        else:
            for idx, entity_id in enumerate(entity_ids):
                risk_levels[0, idx] = self.entities[entity_id].initial_risk_score
        
        # Get adjacency matrix
        adj_matrix = self.get_network_adjacency().values
        
        # Normalize adjacency matrix (row normalization for diffusion)
        row_sums = adj_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        normalized_adj = adj_matrix / row_sums
        
        # Propagate risk through iterations
        for t in range(iterations):
            # Diffusion step: risk flows to neighbors
            diffused = normalized_adj @ risk_levels[t]
            # Update: blend current risk with diffused risk
            risk_levels[t + 1] = (1 - diffusion_rate) * risk_levels[t] + diffusion_rate * diffused
            # Ensure risk stays in [0, 1]
            risk_levels[t + 1] = np.clip(risk_levels[t + 1], 0, 1)
        
        # Convert to DataFrame
        columns = [f"iteration_{i}" for i in range(iterations + 1)]
        return pd.DataFrame(risk_levels.T, index=entity_ids, columns=columns)
    
    # -------------------------------------------------------------------------
    # Regime-aware PD calibration
    # -------------------------------------------------------------------------
    def calibrate_pd_multiplier_from_regime(
        self,
        market_chain_name: str,
        scenario_constraints: List[MarketConstraint],
        base_state: str = "Normal",
        stressed_state: str = "Crisis"
    ) -> float:
        """
        Derive a PD multiplier from how much the steady-state probability of a
        stressed regime (e.g., 'Crisis') increases relative to the baseline.

        This gives a big-bank-style mapping from macro/market constraints
        to PD multipliers used in capital markets scenarios.
        """
        if market_chain_name not in self.markov_chains:
            raise ValueError(f"Markov chain '{market_chain_name}' not found")

        chain = self.markov_chains[market_chain_name]
        unconstrained = chain.stationary_distribution()
        constrained, _ = chain.constrained_stationary_distribution(
            scenario_constraints,
            state_names=chain.states
        )

        base_idx = chain.state_to_idx.get(base_state)
        stress_idx = chain.state_to_idx.get(stressed_state)
        if base_idx is None or stress_idx is None:
            raise ValueError(f"States '{base_state}' or '{stressed_state}' not found in market chain")

        base_crisis_prob = unconstrained[stress_idx]
        stressed_crisis_prob = constrained[stress_idx]

        # Simple mapping: PD multiplier grows with crisis prob ratio; floor at 1.0
        ratio = (stressed_crisis_prob + 1e-6) / (base_crisis_prob + 1e-6)
        return float(max(1.0, ratio))
    
    # -------------------------------------------------------------------------
    # Tranche loss distributions (DCM / LevFin)
    # -------------------------------------------------------------------------
    def simulate_tranche_loss_distribution(
        self,
        tranche_ids: List[str],
        scenario: CapitalMarketsScenario,
        n_simulations: int = 10000,
        risk_free_rate: float = 0.03,
        random_seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Monte Carlo loss distribution for a set of DCM/LevFin tranches
        over the scenario horizon.

        Assumes:
        - Annual PD scaled to the horizon using 1 - (1 - PD)^t
        - LGD applied if default occurs
        - Discounting using (risk_free_rate + scenario.discount_rate_shift_bps)

        NOTE:
        - Scenario.market_constraints are applied via regime analysis and used
          to set scenario.pd_multiplier upstream, via calibrate_pd_multiplier_from_regime.

        Args:
            tranche_ids: List of tranche IDs to simulate
            scenario: Capital markets scenario with PD multipliers and constraints
            n_simulations: Number of Monte Carlo simulations
            risk_free_rate: Risk-free rate (e.g., 0.03 = 3%)
            random_seed: Optional random seed for reproducibility

        Returns:
            DataFrame with loss outcomes for each tranche in each simulation
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        records: List[pd.Series] = []
        effective_rf = risk_free_rate + scenario.discount_rate_shift_bps / 10000.0
        horizon = scenario.horizon_years

        for tranche_id in tranche_ids:
            if tranche_id not in self.tranches:
                raise ValueError(f"Tranche {tranche_id} not found")
            tr = self.tranches[tranche_id]

            # Adjust PD for scenario (regime-aware if pd_multiplier was calibrated)
            pd_annual = tr.pd_annual * scenario.pd_multiplier
            pd_annual = min(max(pd_annual, 0.0), 1.0)
            pd_horizon = 1.0 - (1.0 - pd_annual) ** horizon
            lgd = tr.lgd
            notional = tr.notional

            # Simulate defaults
            default_draws = np.random.rand(n_simulations)
            default_flags = (default_draws < pd_horizon).astype(float)

            # Simplified: loss = default * notional * LGD, discounted
            cash_loss = default_flags * notional * lgd
            discounted_loss = cash_loss / ((1.0 + effective_rf) ** horizon)

            records.append(pd.Series(discounted_loss, name=tranche_id))

        loss_df = pd.concat(records, axis=1)
        loss_df.attrs["scenario"] = scenario.name
        loss_df.attrs["n_simulations"] = n_simulations
        self.simulation_history.append({
            "type": "tranche_loss",
            "scenario": scenario.name,
            "loss_df": loss_df
        })
        return loss_df
    
    def summarize_loss_distribution(
        self,
        loss_df: pd.DataFrame,
        var_levels: List[float] = [0.95, 0.99]
    ) -> pd.DataFrame:
        """
        Summarize loss distributions with EL, UL, VaR, and ES, per tranche.

        This gives standard risk metrics that map naturally into internal
        risk reports and capital planning.

        Args:
            loss_df: DataFrame of simulated losses (e.g., from simulate_tranche_loss_distribution)
            var_levels: Confidence levels for VaR/ES (e.g., [0.95, 0.99])

        Returns:
            DataFrame indexed by tranche_id with columns: EL, UL, VaR_xx, ES_xx.
        """
        summary_records: List[Dict[str, Any]] = []
        for col in loss_df.columns:
            losses = loss_df[col]
            el = float(losses.mean())
            ul = float(losses.std())
            rec: Dict[str, Any] = {"tranche_id": col, "EL": el, "UL": ul}
            for q in var_levels:
                var = float(losses.quantile(q))
                es = float(losses[losses >= var].mean())
                rec[f"VaR_{int(q * 100)}"] = var
                rec[f"ES_{int(q * 100)}"] = es
            summary_records.append(rec)
        return pd.DataFrame(summary_records).set_index("tranche_id")
    
    # -------------------------------------------------------------------------
    # Deal / pipeline risk-return views
    # -------------------------------------------------------------------------
    def summarize_deal_risk_and_return(
        self,
        deal_id: str,
        loss_df: pd.DataFrame,
        var_level: float = 0.99
    ) -> Dict[str, Any]:
        """
        Summarize a single deal's underwriting risk vs economics.

        This is the "does the fee pay for the tail risk?" view that
        ECM/DCM/LevFin and risk teams care about.

        Returns a dict with:
        - deal_id, deal_type, stage, sector, region
        - notional, gross_fees, fee_bps
        - EL, VaR, ES at the given confidence
        - EL_pct_notional, VaR_pct_notional
        - EL_over_fees, VaR_over_fees
        """
        if deal_id not in self.deals:
            raise ValueError(f"Deal {deal_id} not found")

        deal = self.deals[deal_id]

        # Collect tranche IDs that appear in loss_df
        deal_tranche_ids = [t.id for t in deal.tranches if t.id in loss_df.columns]
        if not deal_tranche_ids:
            raise ValueError(f"No tranche losses found in loss_df for deal {deal_id}")

        # Aggregate losses across tranches
        deal_losses = loss_df[deal_tranche_ids].sum(axis=1)
        el = float(deal_losses.mean())
        var = float(deal_losses.quantile(var_level))
        es = float(deal_losses[deal_losses >= var].mean())

        total_notional = float(sum(self.tranches[t].notional for t in deal_tranche_ids))
        total_notional = max(total_notional, 1e-6)
        gross_fees = float(deal.gross_fees)

        fee_bps = gross_fees / total_notional * 10_000.0 if total_notional > 0 else 0.0
        el_pct_notional = el / total_notional if total_notional > 0 else 0.0
        var_pct_notional = var / total_notional if total_notional > 0 else 0.0

        el_over_fees = el / gross_fees if gross_fees > 0 else np.nan
        var_over_fees = var / gross_fees if gross_fees > 0 else np.nan

        return {
            "deal_id": deal_id,
            "deal_type": deal.deal_type.value,
            "pipeline_stage": deal.pipeline_stage,
            "sector": deal.sector,
            "region": deal.region,
            "notional": total_notional,
            "gross_fees": gross_fees,
            "fee_bps": fee_bps,
            "EL": el,
            "VaR": var,
            "ES": es,
            "EL_pct_notional": el_pct_notional,
            "VaR_pct_notional": var_pct_notional,
            "EL_over_fees": el_over_fees,
            "VaR_over_fees": var_over_fees,
        }

    def summarize_pipeline_risk_and_return(
        self,
        deal_ids: Optional[List[str]],
        loss_df: pd.DataFrame,
        var_level: float = 0.99
    ) -> pd.DataFrame:
        """
        Summarize risk/return metrics across a set of deals.

        This produces a DataFrame that can be used as:
        - A pipeline risk dashboard for ECM/DCM/LevFin heads
        - Input to capital / buffer discussions (e.g., "top 20 deals by VaR/fees")

        If deal_ids is None, all deals that have at least one tranche in loss_df
        are included.
        """
        if deal_ids is None:
            # Auto-detect deals that appear in loss_df via their tranches
            deal_ids = []
            for deal_id, deal in self.deals.items():
                if any(t.id in loss_df.columns for t in deal.tranches):
                    deal_ids.append(deal_id)

        records: List[Dict[str, Any]] = []
        for d_id in deal_ids:
            try:
                rec = self.summarize_deal_risk_and_return(
                    deal_id=d_id,
                    loss_df=loss_df,
                    var_level=var_level
                )
                records.append(rec)
            except ValueError:
                # Deal has no relevant tranches in loss_df, skip
                continue

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records).set_index("deal_id")
    
    # -------------------------------------------------------------------------
    # Pipeline fee-at-risk (ECM / DCM / LevFin)
    # -------------------------------------------------------------------------
    def compute_pipeline_fee_at_risk(
        self,
        deal_ids: Optional[List[str]] = None,
        loss_df: Optional[pd.DataFrame] = None,
        loss_threshold_ratio: float = 0.02,
        fee_haircut_if_loss: float = 0.5
    ) -> Dict[str, pd.DataFrame]:
        """
        Estimate fee-at-risk for deals in the pipeline.

        Simplified logic:
        - For each deal, aggregate loss on its tranches from loss_df.
        - If loss / notional > loss_threshold_ratio in a simulation,
          assume bank loses `fee_haircut_if_loss` of its fees in that sim.
        - Return distribution of fee outcomes per deal and per bank.

        Args:
            deal_ids: List of deal IDs to analyze (None = all deals)
            loss_df: DataFrame with loss simulation results (from simulate_tranche_loss_distribution)
            loss_threshold_ratio: Loss-to-notional ratio threshold for fee impairment (default 2%)
            fee_haircut_if_loss: Fraction of fees lost if threshold exceeded (default 50%)

        Returns:
            Dictionary mapping deal_id to DataFrame of fee outcomes per bank per simulation.
        """
        if loss_df is None:
            raise ValueError("loss_df (simulation results) must be provided")

        if deal_ids is None:
            deal_ids = list(self.deals.keys())

        sim_index = loss_df.index
        fee_results: Dict[str, pd.DataFrame] = {}

        for deal_id in deal_ids:
            deal = self.deals.get(deal_id)
            if deal is None:
                continue

            # Collect tranche IDs for this deal that we have in loss_df
            deal_tranche_ids = [t.id for t in deal.tranches if t.id in loss_df.columns]
            if not deal_tranche_ids:
                continue

            # Aggregate losses across tranches per simulation
            deal_losses = loss_df[deal_tranche_ids].sum(axis=1)
            total_notional = sum(self.tranches[t].notional for t in deal_tranche_ids)
            total_notional = max(total_notional, 1e-6)
            loss_ratio = deal_losses / total_notional

            # Determine fee haircut per simulation
            base_fee = deal.gross_fees
            haircut_flags = (loss_ratio > loss_threshold_ratio).astype(float)
            bank_fee_outcomes: Dict[str, pd.Series] = {}

            for bank_id, share in deal.bank_share.items():
                bank_base_fee = base_fee * share
                bank_fee_after = bank_base_fee * (1.0 - fee_haircut_if_loss * haircut_flags)
                bank_fee_outcomes[bank_id] = bank_fee_after

            fee_results[deal_id] = pd.DataFrame(bank_fee_outcomes, index=sim_index)

        return fee_results
    
    def aggregate_fee_at_risk(
        self,
        fee_results: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Aggregate fee-at-risk across deals to get per-bank fee distributions.

        This is useful for Head of ECM/DCM/LevFin views: "What is my fee
        distribution under this scenario across the whole pipeline?"
        """
        if not fee_results:
            return pd.DataFrame()

        # Collect per-(deal, bank) series and align on simulation index
        bank_fee_series: Dict[Tuple[str, str], pd.Series] = {}
        for deal_id, df in fee_results.items():
            for bank_id in df.columns:
                key = (deal_id, bank_id)
                bank_fee_series[key] = df[bank_id]

        panel = pd.DataFrame(bank_fee_series)  # index: sim, cols: (deal, bank)

        # Sum across deals per bank
        per_bank: Dict[str, pd.Series] = {}
        for (deal_id, bank_id) in panel.columns:
            series = panel[(deal_id, bank_id)]
            if bank_id not in per_bank:
                per_bank[bank_id] = series.copy()
            else:
                per_bank[bank_id] = per_bank[bank_id] + series

        per_bank_df = pd.DataFrame(per_bank)
        return per_bank_df
    
    # -------------------------------------------------------------------------
    # Generic risk simulation (network-level)
    # -------------------------------------------------------------------------
    def monte_carlo_simulation(
        self,
        n_simulations: int = 1000,
        shock_distribution: str = "normal",
        shock_params: Optional[Dict[str, float]] = None,
        diffusion_rate: float = 0.1,
        iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations to assess risk under various network shocks.
        
        Args:
            n_simulations: Number of Monte Carlo runs
            shock_distribution: Distribution type for shocks ("normal", "uniform", "exponential")
            shock_params: Parameters for shock distribution
            diffusion_rate: Risk diffusion rate
            iterations: Propagation iterations per simulation
            
        Returns:
            Dictionary with simulation results including statistics and distributions
        """
        entity_ids = list(self.entities.keys())
        n_entities = len(entity_ids)
        
        # Default shock parameters
        if shock_params is None:
            shock_params = {"mean": 0.1, "std": 0.05} if shock_distribution == "normal" else {}
        
        # Storage for final risk levels across all simulations
        final_risks = np.zeros((n_simulations, n_entities))
        
        for sim in range(n_simulations):
            # Generate random initial shock
            initial_shock: Dict[str, float] = {}
            for entity_id in entity_ids:
                if shock_distribution == "normal":
                    shock = np.random.normal(shock_params.get("mean", 0.1), 
                                            shock_params.get("std", 0.05))
                elif shock_distribution == "uniform":
                    shock = np.random.uniform(shock_params.get("low", 0.0),
                                             shock_params.get("high", 0.2))
                elif shock_distribution == "exponential":
                    shock = np.random.exponential(shock_params.get("scale", 0.1))
                else:
                    shock = 0.1
                
                initial_shock[entity_id] = float(np.clip(shock, 0, 1))
            
            # Run propagation
            risk_evolution = self.propagate_risk_fluid(
                initial_shock=initial_shock,
                diffusion_rate=diffusion_rate,
                iterations=iterations
            )
            
            # Store final risk levels
            final_risks[sim, :] = risk_evolution.iloc[:, -1].values
        
        # Compute statistics
        results: Dict[str, Any] = {
            "mean_risk": pd.Series(np.mean(final_risks, axis=0), index=entity_ids),
            "std_risk": pd.Series(np.std(final_risks, axis=0), index=entity_ids),
            "p5_risk": pd.Series(np.percentile(final_risks, 5, axis=0), index=entity_ids),
            "p95_risk": pd.Series(np.percentile(final_risks, 95, axis=0), index=entity_ids),
            "max_risk": pd.Series(np.max(final_risks, axis=0), index=entity_ids),
            "all_simulations": pd.DataFrame(final_risks, columns=entity_ids),
            "n_simulations": n_simulations
        }
        
        self.simulation_history.append(results)
        return results
    
    # -------------------------------------------------------------------------
    # Network metrics / contagion
    # -------------------------------------------------------------------------
    def identify_systemic_risk_hubs(self, threshold: float = 0.7) -> List[str]:
        """
        Identify entities that act as systemic risk hubs (highly connected, high risk).
        
        Args:
            threshold: Risk threshold for identifying hubs
            
        Returns:
            List of entity IDs that are systemic risk hubs
        """
        adj_matrix = self.get_network_adjacency()
        
        # Calculate connectivity (sum of incoming and outgoing connections)
        connectivity = adj_matrix.sum(axis=0) + adj_matrix.sum(axis=1)
        
        # Get current risk levels
        risk_levels = pd.Series({
            entity_id: entity.initial_risk_score 
            for entity_id, entity in self.entities.items()
        })
        
        # Identify hubs: high connectivity AND high risk
        hubs = []
        for entity_id in self.entities.keys():
            conn_score = connectivity[entity_id]
            risk_score = risk_levels[entity_id]
            hub_score = (conn_score / connectivity.max()) * risk_score if connectivity.max() > 0 else 0
            
            if hub_score >= threshold:
                hubs.append(entity_id)
        
        return hubs
    
    def get_risk_contagion_paths(
        self,
        source_entity_id: str,
        max_depth: int = 3
    ) -> List[List[str]]:
        """
        Find all paths through which risk can propagate from a source entity.
        
        Args:
            source_entity_id: Starting entity for contagion analysis
            max_depth: Maximum path length to explore
            
        Returns:
            List of paths (each path is a list of entity IDs)
        """
        if source_entity_id not in self.entities:
            raise ValueError(f"Entity {source_entity_id} not found")
        
        paths: List[List[str]] = []
        
        def dfs(current_id: str, path: List[str], depth: int):
            if depth >= max_depth:
                return
            
            for rel in self.relationships:
                if rel.source_id == current_id and rel.target_id not in path:
                    new_path = path + [rel.target_id]
                    paths.append(new_path)
                    dfs(rel.target_id, new_path, depth + 1)
        
        dfs(source_entity_id, [source_entity_id], 0)
        return paths
    
    # -------------------------------------------------------------------------
    # Markov chain / rating / regime methods
    # -------------------------------------------------------------------------
    def add_markov_chain(self, name: str, chain: MarkovChain) -> None:
        """Add a Markov chain model to Varda."""
        self.markov_chains[name] = chain
    
    def simulate_entity_transitions(
        self,
        chain_name: str,
        entity_ids: Optional[List[str]] = None,
        n_steps: int = 12,
        initial_states: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Simulate state transitions for entities using a Markov chain.
        
        Args:
            chain_name: Name of the Markov chain to use
            entity_ids: List of entity IDs to simulate (None = all entities)
            n_steps: Number of time steps to simulate
            initial_states: Dict mapping entity_id to initial state (overrides stored states)
            
        Returns:
            DataFrame with entity states at each time step
        """
        if chain_name not in self.markov_chains:
            raise ValueError(f"Markov chain '{chain_name}' not found")
        
        chain = self.markov_chains[chain_name]
        
        if entity_ids is None:
            entity_ids = list(self.entities.keys())
        
        # Determine initial states
        if initial_states is None:
            initial_states = {}
        
        # Simulate for each entity
        results: Dict[str, List[str]] = {}
        for entity_id in entity_ids:
            initial_state = initial_states.get(
                entity_id,
                self.entity_states.get(entity_id, None)
            )
            path = chain.simulate(n_steps, initial_state)
            results[entity_id] = path
            # Update stored state to final state
            self.entity_states[entity_id] = path[-1]
        
        # Convert to DataFrame
        return pd.DataFrame(results, index=[f"step_{i}" for i in range(n_steps)])
    
    def compute_default_probabilities(
        self,
        chain_name: str,
        horizon: int = 12,
        entity_ids: Optional[List[str]] = None
    ) -> pd.Series:
        """
        Compute default probabilities over time horizon using Markov chain.
        
        Args:
            chain_name: Name of the Markov chain to use
            horizon: Time horizon (number of steps)
            entity_ids: List of entity IDs (None = all entities)
            
        Returns:
            Series with default probabilities for each entity at the given horizon.
        """
        if chain_name not in self.markov_chains:
            raise ValueError(f"Markov chain '{chain_name}' not found")
        
        chain = self.markov_chains[chain_name]
        
        # Find default state
        default_states = ["D", "Default", "default"]
        default_idx = None
        for state in default_states:
            if state in chain.states:
                default_idx = chain.state_to_idx[state]
                break
        
        if default_idx is None:
            raise ValueError(f"No default state found in chain '{chain_name}'")
        
        if entity_ids is None:
            entity_ids = list(self.entities.keys())
        
        # Compute n-step transition probabilities
        transition_n = chain.n_step_transition(horizon)
        
        # Get default probabilities for each entity
        default_probs: Dict[str, float] = {}
        for entity_id in entity_ids:
            initial_state = self.entity_states.get(entity_id, None)
            if initial_state is None:
                # Use initial distribution
                prob = float(np.dot(chain.initial_distribution, transition_n[:, default_idx]))
            else:
                initial_idx = chain.state_to_idx[initial_state]
                prob = float(transition_n[initial_idx, default_idx])
            default_probs[entity_id] = prob
        
        return pd.Series(default_probs, name=f"P(default|{horizon} steps)")
    
    # -------------------------------------------------------------------------
    # Market states / regimes
    # -------------------------------------------------------------------------
    def add_market_state(self, market_state: MarketState) -> None:
        """Add a market state to the analysis."""
        self.market_states[market_state.state_name] = market_state
        # Also add as an entity for network analysis
        entity = Entity(
            id=f"market_{market_state.state_name}",
            name=f"Market State: {market_state.state_name}",
            entity_type="market_state",
            initial_risk_score=1.0 - market_state.base_stability,
            metadata={
                "description": market_state.description,
                "economic_indicators": market_state.economic_indicators,
                **market_state.metadata
            }
        )
        self.entities[entity.id] = entity
    
    def add_market_constraint(self, constraint: MarketConstraint) -> None:
        """Add a market constraint that affects state transitions."""
        self.market_constraints.append(constraint)
    
    def analyze_market_steady_state(
        self,
        chain_name: str,
        constraints: Optional[List[MarketConstraint]] = None,
        return_modified_matrix: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze market state steady state probabilities accounting for constraints.
        
        This method treats each market state as an entity with constraints and
        computes the steady state distribution that reflects how constraints
        affect transition probabilities.
        
        Args:
            chain_name: Name of the Markov chain representing market states
            constraints: Optional list of constraints (uses global constraints if None)
            return_modified_matrix: If True, also return the modified transition matrix
            
        Returns:
            Dictionary with:
            - steady_state: Stationary probability distribution
            - state_names: List of state names
            - unconstrained_steady_state: Steady state without constraints
            - modified_transition_matrix: (if return_modified_matrix=True)
            - constraint_impacts: Summary of how constraints affected transitions
        """
        if chain_name not in self.markov_chains:
            raise ValueError(f"Markov chain '{chain_name}' not found")
        
        chain = self.markov_chains[chain_name]
        
        # Use provided constraints or global constraints
        if constraints is None:
            constraints = self.market_constraints
        
        # Get unconstrained steady state
        unconstrained_steady = chain.stationary_distribution()
        
        # Get constrained steady state
        if constraints:
            constrained_steady, modified_matrix = chain.constrained_stationary_distribution(
                constraints,
                state_names=chain.states
            )
        else:
            constrained_steady = unconstrained_steady
            modified_matrix = chain.transition_matrix
        
        # Analyze constraint impacts
        constraint_impacts: Dict[str, List[Dict[str, Any]]] = {}
        if constraints:
            for constraint in constraints:
                impacts = []
                for i, from_state in enumerate(chain.states):
                    for j, to_state in enumerate(chain.states):
                        impact = constraint.get_transition_impact(from_state, to_state)
                        if impact != 1.0:
                            impacts.append({
                                "transition": f"{from_state}->{to_state}",
                                "impact": impact,
                                "base_prob": chain.transition_matrix[i, j],
                                "modified_prob": modified_matrix[i, j]
                            })
                if impacts:
                    constraint_impacts[constraint.name] = impacts
        
        result: Dict[str, Any] = {
            "steady_state": pd.Series(constrained_steady, index=chain.states, name="Probability"),
            "state_names": chain.states,
            "unconstrained_steady_state": pd.Series(unconstrained_steady, index=chain.states),
            "constraint_impacts": constraint_impacts,
            "n_constraints": len(constraints) if constraints else 0
        }
        
        if return_modified_matrix:
            result["modified_transition_matrix"] = pd.DataFrame(
                modified_matrix,
                index=chain.states,
                columns=chain.states
            )
        
        return result
    
    def compare_market_scenarios(
        self,
        chain_name: str,
        scenario_constraints: Dict[str, List[MarketConstraint]]
    ) -> pd.DataFrame:
        """
        Compare steady state probabilities across different constraint scenarios.
        
        Args:
            chain_name: Name of the Markov chain
            scenario_constraints: Dict mapping scenario names to constraint lists
            
        Returns:
            DataFrame with steady state probabilities for each scenario
        """
        if chain_name not in self.markov_chains:
            raise ValueError(f"Markov chain '{chain_name}' not found")
        
        results = {}
        for scenario_name, constraints in scenario_constraints.items():
            analysis = self.analyze_market_steady_state(chain_name, constraints)
            results[scenario_name] = analysis["steady_state"]
        
        return pd.DataFrame(results)
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    def summary(self) -> str:
        """Generate a summary of the Varda instance."""
        n_entities = len(self.entities)
        n_relationships = len(self.relationships)
        n_simulations = len(self.simulation_history)
        n_chains = len(self.markov_chains)
        n_market_states = len(self.market_states)
        n_constraints = len(self.market_constraints)
        n_deals = len(self.deals)
        n_tranches = len(self.tranches)
        
        entity_types: Dict[str, int] = {}
        for entity in self.entities.values():
            entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1
        
        deal_types: Dict[str, int] = {}
        for deal in self.deals.values():
            deal_type_name = deal.deal_type.value
            deal_types[deal_type_name] = deal_types.get(deal_type_name, 0) + 1
        
        summary = f"""
Varda Capital Markets Risk Lab: {self.name}
==========================================
Entities: {n_entities}
  {', '.join(f'{k}: {v}' for k, v in entity_types.items())}
Relationships: {n_relationships}
Deals: {n_deals}
  {', '.join(f'{k}: {v}' for k, v in deal_types.items()) if deal_types else 'None'}
Tranches: {n_tranches}
Markov Chains: {n_chains}
Market States: {n_market_states}
Market Constraints: {n_constraints}
Simulations Run: {n_simulations}

Systemic Risk Hubs: {len(self.identify_systemic_risk_hubs())}
        """
        return summary.strip()


# Example usage and demonstration
if __name__ == "__main__":
    # Create a Varda instance
    varda = Varda("Capital Markets Risk Lab Demo")
    
    # Add entities (issuers, banks, investors)
    varda.add_entity(Entity("issuer1", "TechCorp Inc", "issuer", initial_risk_score=0.2), initial_state="BBB")
    varda.add_entity(Entity("bank1", "BigBank", "bank", initial_risk_score=0.15), initial_state="AA")
    varda.add_entity(Entity("bank2", "GlobalBank", "bank", initial_risk_score=0.18), initial_state="A")
    varda.add_entity(Entity("sponsor1", "Private Equity Fund", "sponsor", initial_risk_score=0.25), initial_state="BB")
    
    # Create a high-yield bond deal
    hy_bond_tranche = Tranche(
        id="tranche_hy1",
        deal_id="deal_hy1",
        currency="USD",
        notional=500_000_000,
        coupon=0.08,
        spread_bps=400,
        maturity_years=5.0,
        rating="BB",
        pd_annual=0.03,
        lgd=0.60,
        seniority="senior"
    )
    
    hy_deal = CapitalMarketsDeal(
        id="deal_hy1",
        issuer_entity_id="issuer1",
        deal_type=DealType.DCM_HY,
        tranches=[hy_bond_tranche],
        bookrunners=["bank1", "bank2"],
        gross_fees=12_500_000,  # 2.5% of notional
        bank_share={"bank1": 0.60, "bank2": 0.40},
        pipeline_stage="priced",
        sector="Technology"
    )
    
    varda.add_deal(hy_deal)
    
    # Simple market regime chain for demonstration (Normal / Crisis)
    market_chain = create_market_regime_chain()
    varda.add_markov_chain("market_regimes", market_chain)
    
    # Example constraint: high inflation, tighter policy
    high_inflation = MarketConstraint(
        name="High Inflation",
        constraint_type="economic",
        value=4.0,
        impact_on_transitions={
            "Normal->Stressed": 1.5,
            "Stressed->Crisis": 1.8
        }
    )
    
    # Calibrate PD multiplier from regime
    pd_mult = varda.calibrate_pd_multiplier_from_regime(
        market_chain_name="market_regimes",
        scenario_constraints=[high_inflation],
        base_state="Normal",
        stressed_state="Crisis"
    )
    
    # Create a scenario with credit spread shock and regime-aware PD multiplier
    stress_scenario = CapitalMarketsScenario(
        name="Credit Spread Widening",
        description="200bps spread widening, PD multiplier from High Inflation regime",
        spread_shock_bps=200.0,
        pd_multiplier=pd_mult,
        horizon_years=1.0,
        market_constraints=[high_inflation]
    )
    
    # Run loss distribution simulation
    print("Running tranche loss distribution simulation...")
    loss_df = varda.simulate_tranche_loss_distribution(
        tranche_ids=["tranche_hy1"],
        scenario=stress_scenario,
        n_simulations=10000,
        random_seed=42
    )
    
    # Summarize losses (EL, UL, VaR, ES)
    loss_summary = varda.summarize_loss_distribution(loss_df)
    print("\nLoss Summary (EL/UL/VaR/ES):")
    print(loss_summary)
    
    # Deal-level risk/return summary
    deal_summary = varda.summarize_deal_risk_and_return("deal_hy1", loss_df)
    print("\nDeal Risk/Return Summary:")
    for k, v in deal_summary.items():
        print(f"  {k}: {v}")
    
    # Pipeline-level risk/return summary (trivial here: one deal)
    pipeline_summary = varda.summarize_pipeline_risk_and_return(
        deal_ids=["deal_hy1"],
        loss_df=loss_df
    )
    print("\nPipeline Risk/Return Summary:")
    print(pipeline_summary)
    
    # Compute fee-at-risk
    print("\nComputing fee-at-risk...")
    fee_at_risk = varda.compute_pipeline_fee_at_risk(
        deal_ids=["deal_hy1"],
        loss_df=loss_df,
        loss_threshold_ratio=0.02,
        fee_haircut_if_loss=0.5
    )
    
    for deal_id, bank_fees in fee_at_risk.items():
        print(f"\nDeal: {deal_id}")
        for bank_id in bank_fees.columns:
            mean_fee = bank_fees[bank_id].mean()
            p5_fee = bank_fees[bank_id].quantile(0.05)
            print(f"  {bank_id}: Mean Fee = ${mean_fee:,.2f}, 5th Pct = ${p5_fee:,.2f}")
    
    # Aggregate fee-at-risk per bank across deals
    per_bank_df = varda.aggregate_fee_at_risk(fee_at_risk)
    print("\nAggregate Fee-at-Risk per Bank:")
    for bank_id in per_bank_df.columns:
        series = per_bank_df[bank_id]
        print(
            f"  {bank_id}: Expected Fees = ${series.mean():,.2f}, "
            f"5th pct = ${series.quantile(0.05):,.2f}"
        )
    
    print("\n" + varda.summary())
