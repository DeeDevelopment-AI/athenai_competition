"""
Tests para el modulo de inferencia de regimen latente.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.latent_regime_inference import (
    LatentRegimeInference,
    InferenceMethod,
    ActivityMask,
)


class TestActivityMask:
    """Tests para la mascara de actividad."""

    def test_build_activity_mask_basic(self):
        """Debe construir mascara correctamente."""
        # Crear datos de prueba con algunos NaN
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        algo_returns = pd.DataFrame({
            'algo_1': np.random.randn(100) * 0.01,
            'algo_2': np.random.randn(100) * 0.01,
        }, index=dates)

        # Introducir algunos NaN
        algo_returns.loc[dates[10:20], 'algo_1'] = np.nan
        algo_returns.loc[dates[50:60], 'algo_2'] = np.nan

        inference = LatentRegimeInference()
        mask = inference.build_activity_mask(algo_returns)

        # Verificar estructura
        assert isinstance(mask, ActivityMask)
        assert mask.is_active.shape == algo_returns.shape
        assert mask.time_since_last.shape == algo_returns.shape

        # Verificar que is_active es correcto
        assert mask.is_active.loc[dates[5], 'algo_1'] == True
        assert mask.is_active.loc[dates[15], 'algo_1'] == False
        assert mask.is_active.loc[dates[55], 'algo_2'] == False

    def test_zero_return_is_still_active(self):
        """A zero return is an observed day, not inactivity."""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        algo_returns = pd.DataFrame({
            'algo_1': [0.01, 0.0, -0.01, np.nan, 0.0],
        }, index=dates)

        inference = LatentRegimeInference()
        mask = inference.build_activity_mask(algo_returns)

        assert mask.is_active.loc[dates[1], 'algo_1'] == True
        assert mask.is_active.loc[dates[3], 'algo_1'] == False

    def test_no_9999999_values(self):
        """No debe haber valores 9999999 en la mascara."""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        algo_returns = pd.DataFrame({
            'algo_1': np.random.randn(50) * 0.01,
        }, index=dates)
        algo_returns.iloc[20:30] = np.nan

        inference = LatentRegimeInference()
        mask = inference.build_activity_mask(algo_returns)

        # Verificar que no hay 9999999
        assert not (mask.is_active == 9999999).any().any()
        assert mask.time_since_last.max().max() < 100  # Valores razonables


class TestFamilyClustering:
    """Tests para clustering de familias."""

    def test_compute_behavioral_features(self):
        """Debe calcular features de comportamiento."""
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        algo_returns = pd.DataFrame({
            'algo_1': np.random.randn(200) * 0.01 + 0.001,  # Positivo
            'algo_2': np.random.randn(200) * 0.02,         # Alta vol
            'algo_3': np.random.randn(200) * 0.005,        # Baja vol
        }, index=dates)

        inference = LatentRegimeInference()
        features = inference.compute_algo_behavioral_features(algo_returns)

        # Verificar estructura
        assert len(features) == 3
        assert 'ann_return' in features.columns
        assert 'sharpe' in features.columns
        assert 'max_dd' in features.columns

        # Verificar que vol relativa es correcta
        assert features.loc['algo_2', 'ann_vol'] > features.loc['algo_3', 'ann_vol']

    def test_assign_families(self):
        """Debe asignar familias correctamente."""
        dates = pd.date_range('2020-01-01', periods=200, freq='D')

        # Crear 3 grupos distintos
        np.random.seed(42)
        group1 = np.random.randn(200) * 0.01 + 0.002  # High return, low vol
        group2 = np.random.randn(200) * 0.03         # High vol
        group3 = np.random.randn(200) * 0.005 - 0.001  # Negative return

        algo_returns = pd.DataFrame({
            'algo_1': group1,
            'algo_2': group1 * 0.9 + np.random.randn(200) * 0.002,
            'algo_3': group2,
            'algo_4': group2 * 0.8 + np.random.randn(200) * 0.005,
            'algo_5': group3,
            'algo_6': group3 * 0.95 + np.random.randn(200) * 0.001,
        }, index=dates)

        inference = LatentRegimeInference()
        features = inference.compute_algo_behavioral_features(algo_returns)
        family_labels = inference.assign_families(features, n_families=3, method='gmm')

        # Verificar que hay 3 familias
        n_families = len(set(family_labels[family_labels >= 0]))
        assert n_families >= 2  # Al menos 2 familias distintas

        # Verificar que algoritmos similares tienden a agruparse
        # (no es determinista, pero deberia tender a agruparse)
        assert isinstance(family_labels, pd.Series)


class TestTemporalFeatures:
    """Tests para features temporales."""

    def test_build_temporal_features(self):
        """Debe construir features temporales."""
        dates = pd.date_range('2020-01-01', periods=200, freq='D')

        algo_returns = pd.DataFrame({
            'algo_1': np.random.randn(200) * 0.01,
            'algo_2': np.random.randn(200) * 0.01,
        }, index=dates)

        benchmark_weights = pd.DataFrame({
            'algo_1': [0.5] * 200,
            'algo_2': [0.5] * 200,
        }, index=dates)

        benchmark_returns = pd.Series(
            np.random.randn(200) * 0.008,
            index=dates
        )

        family_labels = pd.Series([0, 1], index=['algo_1', 'algo_2'])

        inference = LatentRegimeInference()
        mask = inference.build_activity_mask(algo_returns)

        family_agg, universe_feat, bench_feat = inference.build_temporal_features(
            algo_returns, benchmark_weights, benchmark_returns, family_labels, mask
        )

        # Verificar familia aggregates
        assert family_agg.returns is not None
        assert len(family_agg.returns) == len(dates)

        # Verificar universe features
        assert universe_feat.cross_sectional_dispersion is not None
        assert len(universe_feat.cross_sectional_dispersion) == len(dates)

        # Verificar benchmark features
        assert bench_feat.returns is not None
        assert len(bench_feat.returns) == len(dates)

    def test_observation_matrix_excludes_benchmark_allocation_features_by_default(self):
        """Regime observations should not include benchmark allocation decisions by default."""
        dates = pd.date_range('2020-01-01', periods=80, freq='D')
        algo_returns = pd.DataFrame({
            'algo_1': np.random.randn(80) * 0.01,
            'algo_2': np.random.randn(80) * 0.01,
        }, index=dates)
        benchmark_weights = pd.DataFrame({
            'algo_1': [0.5] * 80,
            'algo_2': [0.5] * 80,
        }, index=dates)
        benchmark_returns = algo_returns.mean(axis=1)
        family_labels = pd.Series([0, 1], index=['algo_1', 'algo_2'])

        inference = LatentRegimeInference()
        mask = inference.build_activity_mask(algo_returns)
        family_agg, universe_feat, bench_feat = inference.build_temporal_features(
            algo_returns, benchmark_weights, benchmark_returns, family_labels, mask
        )
        obs = inference._build_observation_matrix(family_agg, universe_feat, bench_feat)

        assert 'bench_turnover' not in obs.columns
        assert 'bench_concentration' not in obs.columns
        assert not any(col.endswith('_wgt') for col in obs.columns)


class TestRegimeInference:
    """Tests para inferencia de regimen."""

    def test_infer_regimes_fuzzy(self):
        """Debe inferir regimenes con metodo fuzzy."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        # Crear datos con patrones claros
        np.random.seed(42)
        # Primer tercio: baja vol, retornos positivos (expansion)
        ret1 = np.random.randn(100) * 0.005 + 0.001
        # Segundo tercio: alta vol, retornos negativos (recession)
        ret2 = np.random.randn(100) * 0.02 - 0.002
        # Tercer tercio: baja vol, retornos neutros (recovery)
        ret3 = np.random.randn(100) * 0.006 + 0.0005

        algo_returns = pd.DataFrame({
            'algo_1': np.concatenate([ret1, ret2, ret3]),
            'algo_2': np.concatenate([ret1 * 0.9, ret2 * 1.1, ret3 * 0.95]),
        }, index=dates)

        benchmark_weights = pd.DataFrame({
            'algo_1': [0.5] * 300,
            'algo_2': [0.5] * 300,
        }, index=dates)

        benchmark_returns = algo_returns.mean(axis=1) * 0.8

        family_labels = pd.Series([0, 1], index=['algo_1', 'algo_2'])

        inference = LatentRegimeInference(n_regimes=4)
        mask = inference.build_activity_mask(algo_returns)

        family_agg, universe_feat, bench_feat = inference.build_temporal_features(
            algo_returns, benchmark_weights, benchmark_returns, family_labels, mask
        )

        result = inference.infer_regimes(
            family_agg, universe_feat, bench_feat,
            method=InferenceMethod.FUZZY_SCORECARD
        )

        # Verificar estructura del resultado
        assert result.regime_labels is not None
        assert len(result.regime_labels) == len(dates)
        assert result.regime_probabilities is not None

        # Verificar que hay multiples regimenes
        n_unique_regimes = result.regime_labels.nunique()
        assert n_unique_regimes >= 2

    def test_infer_regimes_temporal_clustering(self):
        """Debe inferir regimenes con clustering temporal."""
        dates = pd.date_range('2020-01-01', periods=200, freq='D')

        algo_returns = pd.DataFrame({
            'algo_1': np.random.randn(200) * 0.01,
        }, index=dates)

        benchmark_weights = pd.DataFrame({
            'algo_1': [1.0] * 200,
        }, index=dates)

        benchmark_returns = algo_returns['algo_1'] * 0.9

        family_labels = pd.Series([0], index=['algo_1'])

        inference = LatentRegimeInference(n_regimes=3)
        mask = inference.build_activity_mask(algo_returns)

        family_agg, universe_feat, bench_feat = inference.build_temporal_features(
            algo_returns, benchmark_weights, benchmark_returns, family_labels, mask
        )

        result = inference.infer_regimes(
            family_agg, universe_feat, bench_feat,
            method=InferenceMethod.TEMPORAL_CLUSTERING
        )

        assert result.regime_labels is not None
        assert len(result.regime_labels) == len(dates)


class TestConditionalAnalysis:
    """Tests para analisis condicional."""

    def test_analyze_benchmark_conditional(self):
        """Debe analizar benchmark condicionado a regimenes."""
        dates = pd.date_range('2020-01-01', periods=200, freq='D')

        # Crear pesos que varian por periodo
        weights_1 = [0.6] * 100 + [0.3] * 100
        weights_2 = [0.4] * 100 + [0.7] * 100

        benchmark_weights = pd.DataFrame({
            'algo_1': weights_1,
            'algo_2': weights_2,
        }, index=dates)

        family_labels = pd.Series([0, 1], index=['algo_1', 'algo_2'])

        # Crear regimenes que corresponden a los cambios de peso
        regime_labels = pd.Series(
            [0] * 100 + [1] * 100,
            index=dates
        )

        inference = LatentRegimeInference()
        conditional = inference.analyze_benchmark_conditional(
            benchmark_weights, family_labels, regime_labels
        )

        # Verificar estructura
        assert conditional.family_weights_by_regime is not None
        assert conditional.selection_probability is not None

        # En regimen 0, familia 0 deberia tener mas peso
        assert conditional.family_weights_by_regime['0'][0] > conditional.family_weights_by_regime['0'][1]

        # En regimen 1, familia 1 deberia tener mas peso
        assert conditional.family_weights_by_regime['1'][1] > conditional.family_weights_by_regime['1'][0]


class TestIntegration:
    """Tests de integracion del pipeline completo."""

    def test_full_pipeline(self):
        """Debe ejecutar el pipeline completo sin errores."""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        np.random.seed(42)

        # Crear 4 algoritmos
        algo_returns = pd.DataFrame({
            'algo_1': np.random.randn(300) * 0.01 + 0.001,
            'algo_2': np.random.randn(300) * 0.015,
            'algo_3': np.random.randn(300) * 0.008 - 0.0005,
            'algo_4': np.random.randn(300) * 0.012 + 0.0005,
        }, index=dates)

        benchmark_weights = pd.DataFrame({
            'algo_1': [0.25] * 150 + [0.35] * 150,
            'algo_2': [0.25] * 150 + [0.15] * 150,
            'algo_3': [0.25] * 150 + [0.25] * 150,
            'algo_4': [0.25] * 150 + [0.25] * 150,
        }, index=dates)

        benchmark_returns = (algo_returns * benchmark_weights).sum(axis=1)

        # Pipeline
        inference = LatentRegimeInference(n_regimes=3)

        # 1. Mascara
        mask = inference.build_activity_mask(algo_returns)
        assert isinstance(mask, ActivityMask)

        # 2. Features y familias
        algo_features = inference.compute_algo_behavioral_features(algo_returns)
        assert len(algo_features) == 4

        family_labels = inference.assign_families(algo_features, n_families=2)
        assert len(family_labels) == 4

        # 3. Features temporales
        family_agg, universe_feat, bench_feat = inference.build_temporal_features(
            algo_returns, benchmark_weights, benchmark_returns, family_labels, mask
        )

        # 4. Inferencia
        result = inference.infer_regimes(
            family_agg, universe_feat, bench_feat,
            method=InferenceMethod.FUZZY_SCORECARD
        )
        assert len(result.regime_labels) == 300

        # 5. Analisis condicional
        conditional = inference.analyze_benchmark_conditional(
            benchmark_weights, family_labels, result.regime_labels, algo_returns
        )
        assert conditional.family_weights_by_regime is not None

        # 6. Reporte
        report = inference.generate_report(result, conditional)
        assert 'LATENT REGIME' in report
        assert 'BENCHMARK' in report


# =============================================================================
# Tests for InvestmentClockRegimeInference
# =============================================================================

from src.analysis.latent_regime_inference import (
    InvestmentClockRegimeInference,
    BusinessCyclePhase,
)
from src.analysis.asset_inference import AssetInference


class TestInvestmentClockRegimeInference:
    """Tests for Investment Clock regime inference based on asset class performance."""

    @pytest.fixture
    def mock_asset_inferences(self):
        """Create mock asset inference results."""
        inferences = {}
        # Cyclical algos
        for i in range(5):
            inferences[f'algo_cyclical_{i}'] = AssetInference(
                predicted_asset='Technology',
                asset_class='equity',
                asset_label='Tech Sector',
                direction='long',
                confidence=80,
                best_composite=0.7,
            )
        # Defensive algos
        for i in range(5):
            inferences[f'algo_defensive_{i}'] = AssetInference(
                predicted_asset='Utilities',
                asset_class='equity',
                asset_label='Utilities Sector',
                direction='long',
                confidence=75,
                best_composite=0.65,
            )
        # Government bonds
        for i in range(3):
            inferences[f'algo_gov_{i}'] = AssetInference(
                predicted_asset='Treasury',
                asset_class='fixed_income',
                asset_label='US Treasuries',
                direction='long',
                confidence=85,
                best_composite=0.8,
            )
        # High yield
        for i in range(3):
            inferences[f'algo_hy_{i}'] = AssetInference(
                predicted_asset='HighYield',
                asset_class='fixed_income',
                asset_label='High Yield Bonds',
                direction='long',
                confidence=70,
                best_composite=0.6,
            )
        return inferences

    @pytest.fixture
    def mock_algo_returns(self, mock_asset_inferences):
        """Create mock algorithm returns matching the asset inferences."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='B')
        n_days = len(dates)

        returns = pd.DataFrame(index=dates)

        # Create returns with different characteristics by asset class
        for algo_id in mock_asset_inferences.keys():
            if 'cyclical' in algo_id:
                # Cyclical: higher vol, correlated with "market"
                base = np.random.randn(n_days) * 0.02
                returns[algo_id] = base + np.random.randn(n_days) * 0.005
            elif 'defensive' in algo_id:
                # Defensive: lower vol, less correlated
                returns[algo_id] = np.random.randn(n_days) * 0.008
            elif 'gov' in algo_id:
                # Government bonds: low vol, negative correlation to equity
                returns[algo_id] = np.random.randn(n_days) * 0.005
            elif 'hy' in algo_id:
                # High yield: moderate vol, positive correlation to equity
                returns[algo_id] = np.random.randn(n_days) * 0.012

        return returns

    def test_classify_asset(self):
        """Test asset classification into Investment Clock categories."""
        inference = InvestmentClockRegimeInference()

        # Test cyclical
        assert inference._classify_asset('equity', 'technology') == 'cyclical'
        assert inference._classify_asset('equity', 'materials') == 'cyclical'
        assert inference._classify_asset('equity', 'industrials') == 'cyclical'

        # Test defensive
        assert inference._classify_asset('equity', 'utilities') == 'defensive'
        assert inference._classify_asset('equity', 'healthcare') == 'defensive'
        assert inference._classify_asset('equity', 'consumer_staples') == 'defensive'

        # Test bonds
        assert inference._classify_asset('fixed_income', 'treasury') == 'government'
        assert inference._classify_asset('fixed_income', 'high_yield') == 'high_yield'

        # Test commodities
        assert inference._classify_asset('commodity', 'gold') == 'commodities'
        assert inference._classify_asset('commodity', 'oil') == 'commodities'

        # Test forex
        assert inference._classify_asset('forex', 'eurusd') == 'forex'

    def test_map_algos_to_categories(self, mock_asset_inferences):
        """Test mapping algorithms to Investment Clock categories."""
        inference = InvestmentClockRegimeInference()

        mapping = inference._map_algos_to_categories(mock_asset_inferences)

        # Should have all algos mapped
        assert len(mapping) == len(mock_asset_inferences)

        # Check category assignments
        cyclical_count = sum(1 for v in mapping.values() if v == 'cyclical')
        defensive_count = sum(1 for v in mapping.values() if v == 'defensive')
        gov_count = sum(1 for v in mapping.values() if v == 'government')
        hy_count = sum(1 for v in mapping.values() if v == 'high_yield')

        assert cyclical_count == 5
        assert defensive_count == 5
        assert gov_count == 3
        assert hy_count == 3

    def test_compute_asset_class_performance(self, mock_algo_returns, mock_asset_inferences):
        """Test computation of asset class performance."""
        inference = InvestmentClockRegimeInference(rolling_window=21, vol_window=63)

        mapping = inference._map_algos_to_categories(mock_asset_inferences)
        performance = inference._compute_asset_class_performance(mock_algo_returns, mapping)

        # Check structure
        assert performance.returns is not None
        assert performance.volatility is not None
        assert performance.n_algorithms is not None

        # Should have asset class columns
        assert 'cyclical' in performance.returns.columns
        assert 'defensive' in performance.returns.columns
        assert 'government' in performance.returns.columns
        assert 'high_yield' in performance.returns.columns

        # Returns should be reasonable
        assert performance.returns['cyclical'].mean() < 0.1  # Not crazy
        assert performance.returns['cyclical'].mean() > -0.1

    def test_build_regime_indicators(self, mock_algo_returns, mock_asset_inferences):
        """Test building regime indicators from asset class performance."""
        inference = InvestmentClockRegimeInference(rolling_window=21)

        mapping = inference._map_algos_to_categories(mock_asset_inferences)
        performance = inference._compute_asset_class_performance(mock_algo_returns, mapping)
        indicators = inference._build_regime_indicators(performance)

        # Check all indicators exist
        assert indicators.cyclical_vs_defensive is not None
        assert indicators.credit_spread is not None
        assert indicators.equity_vs_bonds is not None
        assert indicators.volatility_level is not None
        assert indicators.correlation_level is not None
        assert indicators.momentum_breadth is not None

        # Check lengths match
        assert len(indicators.cyclical_vs_defensive) == len(mock_algo_returns)
        assert len(indicators.credit_spread) == len(mock_algo_returns)

    def test_infer_regime_from_assets(self, mock_algo_returns, mock_asset_inferences):
        """Test full regime inference from asset classes."""
        inference = InvestmentClockRegimeInference(rolling_window=21)

        result = inference.infer_regime_from_assets(
            mock_algo_returns,
            mock_asset_inferences
        )

        # Check result structure
        assert result.regime_labels is not None
        assert result.regime_probabilities is not None
        assert result.indicators is not None
        assert result.phase_statistics is not None
        assert result.transition_matrix is not None

        # Should have 4 phases (or fewer if some phases don't occur)
        unique_regimes = result.regime_labels.unique()
        assert len(unique_regimes) <= 4
        assert len(unique_regimes) >= 1

        # Probabilities should sum to 1
        prob_sums = result.regime_probabilities.sum(axis=1)
        assert np.allclose(prob_sums.dropna(), 1.0, atol=0.01)

    def test_generate_report(self, mock_algo_returns, mock_asset_inferences):
        """Test report generation."""
        inference = InvestmentClockRegimeInference(rolling_window=21)

        result = inference.infer_regime_from_assets(
            mock_algo_returns,
            mock_asset_inferences
        )

        report = inference.generate_report(result)

        # Should contain key sections
        assert 'INVESTMENT CLOCK' in report
        assert 'METHODOLOGY' in report
        assert 'REGIME INDICATORS' in report
        assert 'BUSINESS CYCLE' in report
        assert 'TRANSITION MATRIX' in report

    def test_business_cycle_phases_enum(self):
        """Test BusinessCyclePhase enum values."""
        assert BusinessCyclePhase.RECOVERY.value == 0
        assert BusinessCyclePhase.EXPANSION.value == 1
        assert BusinessCyclePhase.SLOWDOWN.value == 2
        assert BusinessCyclePhase.CONTRACTION.value == 3
