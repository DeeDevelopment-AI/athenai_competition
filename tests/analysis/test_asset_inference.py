"""
Tests for src/analysis/asset_inference.py

Tests cover:
- Benchmark loading from various sources
- Fast correlation screening (Stage 1)
- Deep 6-signal analysis (Stage 2)
- Composite score calculation
- Multi-asset detection
- Confidence calculation
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.asset_inference import (
    AssetInferenceEngine,
    AssetInference,
    AssetExposure,
    BenchmarkLoader,
    _safe_float,
    _guess_asset_class,
    _guess_asset_class_from_name,
    _analyze_trading_pattern,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_algo_daily():
    """Create sample algorithm daily OHLC data."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    prices = 100 * (1 + np.random.randn(252).cumsum() * 0.01)

    return pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
    }, index=dates)


@pytest.fixture
def sample_algo_returns(sample_algo_daily):
    """Create sample algorithm returns."""
    return sample_algo_daily['close'].pct_change().dropna()


@pytest.fixture
def sample_benchmarks():
    """Create sample benchmark data."""
    np.random.seed(123)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')

    benchmarks = {}
    bench_returns = {}
    bench_meta = {}

    # Create several benchmarks with different characteristics
    for name, corr_factor, asset_class in [
        ('SP500', 0.7, 'indices'),
        ('EURUSD', 0.3, 'forex'),
        ('Gold', -0.2, 'commodities'),
        ('EQ_AAPL', 0.5, 'equity'),
    ]:
        # Create correlated returns
        base_returns = np.random.randn(252) * 0.01
        prices = 100 * (1 + base_returns).cumprod()

        benchmarks[name] = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
        }, index=dates)

        bench_returns[name] = pd.Series(base_returns, index=dates)

        bench_meta[name] = {
            'asset_class': asset_class,
            'ticker': name,
            'source': 'test',
            'start': str(dates[0].date()),
            'end': str(dates[-1].date()),
            'n_days': 252,
        }

    return benchmarks, bench_returns, bench_meta


@pytest.fixture
def inference_engine(sample_benchmarks):
    """Create AssetInferenceEngine with sample benchmarks."""
    benchmarks, bench_returns, bench_meta = sample_benchmarks
    return AssetInferenceEngine(
        benchmarks=benchmarks,
        bench_returns=bench_returns,
        bench_meta=bench_meta,
        top_n_candidates=5,
    )


# =============================================================================
# Tests for helper functions
# =============================================================================

class TestSafeFloat:
    """Tests for _safe_float helper function."""

    def test_normal_value(self):
        """Test with normal float value."""
        assert _safe_float(1.5) == 1.5

    def test_nan_value(self):
        """Test with NaN value."""
        assert _safe_float(np.nan) == 0.0
        assert _safe_float(np.nan, default=99.0) == 99.0

    def test_inf_value(self):
        """Test with inf value."""
        assert _safe_float(np.inf) == 0.0
        assert _safe_float(-np.inf) == 0.0

    def test_none_value(self):
        """Test with None value."""
        assert _safe_float(None) == 0.0


class TestGuessAssetClass:
    """Tests for _guess_asset_class function."""

    def test_forex_pairs(self):
        """Test forex pair detection."""
        assert _guess_asset_class('EURUSD') == 'forex'
        assert _guess_asset_class('GBPJPY') == 'forex'
        assert _guess_asset_class('AUDUSD') == 'forex'

    def test_commodities(self):
        """Test commodity detection."""
        assert _guess_asset_class('XAUUSD') == 'commodities'
        assert _guess_asset_class('WTIUSD') == 'commodities'

    def test_indices(self):
        """Test index detection."""
        assert _guess_asset_class('SPXUSD') == 'indices'
        assert _guess_asset_class('NASUSD') == 'indices'

    def test_crypto(self):
        """Test crypto detection."""
        assert _guess_asset_class('BTCUSD') == 'crypto'
        assert _guess_asset_class('ETHUSD') == 'crypto'

    def test_unknown(self):
        """Test unknown asset class."""
        assert _guess_asset_class('RANDOM') == 'other'


class TestGuessAssetClassFromName:
    """Tests for _guess_asset_class_from_name function."""

    def test_equity_prefix(self):
        """Test equity prefix detection."""
        assert _guess_asset_class_from_name('EQ_AAPL') == 'equity'

    def test_futures_prefix(self):
        """Test futures prefix detection."""
        assert _guess_asset_class_from_name('FUT_SP500_Fut') == 'indices'
        assert _guess_asset_class_from_name('FUT_Gold_Fut') == 'commodities'

    def test_forex_names(self):
        """Test forex name detection."""
        assert _guess_asset_class_from_name('EURUSD') == 'forex'

    def test_commodity_names(self):
        """Test commodity name detection."""
        assert _guess_asset_class_from_name('Gold') == 'commodities'
        assert _guess_asset_class_from_name('WTI_Oil') == 'commodities'


class TestAnalyzeTradingPattern:
    """Tests for _analyze_trading_pattern function."""

    def test_basic_pattern_analysis(self, sample_algo_daily):
        """Test basic trading pattern analysis."""
        pattern = _analyze_trading_pattern(sample_algo_daily)

        assert 'pattern' in pattern
        assert 'active_days_pct' in pattern
        assert 'avg_daily_move_pct' in pattern

    def test_short_series(self):
        """Test with very short series."""
        short_df = pd.DataFrame({
            'close': [100, 101, 102],
        }, index=pd.date_range('2020-01-01', periods=3))

        pattern = _analyze_trading_pattern(short_df)
        assert pattern['pattern'] == 'unknown'


# =============================================================================
# Tests for AssetInferenceEngine
# =============================================================================

class TestAssetInferenceEngineStage1:
    """Tests for Stage 1: Fast correlation screening."""

    def test_fast_screen_basic(self, inference_engine, sample_algo_returns):
        """Test basic fast screening."""
        scores = inference_engine._stage1_fast_screen(sample_algo_returns)

        assert isinstance(scores, dict)
        assert len(scores) > 0
        # All scores should be absolute correlations (0-1)
        assert all(0 <= s <= 1 for s in scores.values())

    def test_fast_screen_insufficient_data(self, inference_engine):
        """Test fast screen with insufficient data."""
        short_returns = pd.Series([0.01, 0.02, -0.01], index=pd.date_range('2020-01-01', periods=3))
        scores = inference_engine._stage1_fast_screen(short_returns)

        # Should return empty or very few results due to insufficient overlap
        assert len(scores) == 0 or all(scores.values())


class TestAssetInferenceEngineStage2:
    """Tests for Stage 2: Deep 6-signal analysis."""

    def test_deep_analysis_basic(self, inference_engine, sample_algo_returns, sample_benchmarks):
        """Test basic deep analysis."""
        _, bench_returns, _ = sample_benchmarks
        candidates = list(bench_returns.keys())[:3]

        signals = inference_engine._stage2_deep_analysis(sample_algo_returns, candidates)

        assert isinstance(signals, dict)
        for name in candidates:
            assert name in signals

    def test_deep_analysis_signals(self, inference_engine, sample_algo_returns, sample_benchmarks):
        """Test that all 6 signals are computed."""
        _, bench_returns, _ = sample_benchmarks
        candidates = list(bench_returns.keys())[:1]

        signals = inference_engine._stage2_deep_analysis(sample_algo_returns, candidates)
        sig = signals[candidates[0]]

        if not sig.get('skip', False):
            assert 'pearson_r' in sig
            assert 'spearman_r' in sig
            assert 'beta' in sig
            assert 'dd_corr' in sig
            assert 'vol_regime_corr' in sig
            assert 'directional_agreement' in sig
            assert 'composite_score' in sig

    def test_deep_analysis_composite_score_range(self, inference_engine, sample_algo_returns, sample_benchmarks):
        """Test that composite score is in valid range."""
        _, bench_returns, _ = sample_benchmarks
        candidates = list(bench_returns.keys())

        signals = inference_engine._stage2_deep_analysis(sample_algo_returns, candidates)

        for name, sig in signals.items():
            if not sig.get('skip', False):
                assert 0 <= sig['composite_score'] <= 1


class TestAssetInferenceEngineFull:
    """Tests for full inference pipeline."""

    def test_infer_basic(self, inference_engine, sample_algo_daily):
        """Test basic inference."""
        result = inference_engine.infer(sample_algo_daily)

        assert isinstance(result, AssetInference)
        assert result.predicted_asset is not None
        assert result.asset_class is not None
        assert 0 <= result.confidence <= 100

    def test_infer_with_returns(self, inference_engine, sample_algo_daily, sample_algo_returns):
        """Test inference with pre-computed returns."""
        result = inference_engine.infer(sample_algo_daily, sample_algo_returns)

        assert isinstance(result, AssetInference)

    def test_infer_insufficient_data(self, inference_engine):
        """Test inference with insufficient data."""
        short_df = pd.DataFrame({
            'close': [100, 101, 102],
        }, index=pd.date_range('2020-01-01', periods=3))

        result = inference_engine.infer(short_df)

        assert result.predicted_asset == 'unknown'
        assert result.confidence == 0

    def test_infer_top_matches(self, inference_engine, sample_algo_daily):
        """Test that top matches are returned."""
        result = inference_engine.infer(sample_algo_daily)

        assert isinstance(result.top_matches, list)
        # Should have some matches if inference succeeded
        if result.predicted_asset != 'unknown':
            assert len(result.top_matches) > 0

    def test_infer_trading_pattern(self, inference_engine, sample_algo_daily):
        """Test that trading pattern is analyzed."""
        result = inference_engine.infer(sample_algo_daily)

        assert isinstance(result.trading_pattern, dict)
        if result.trading_pattern:
            assert 'pattern' in result.trading_pattern


class TestAssetInferenceEngineConfidence:
    """Tests for confidence calculation."""

    def test_confidence_calculation(self, inference_engine, sample_algo_daily):
        """Test confidence is computed."""
        result = inference_engine.infer(sample_algo_daily)

        assert isinstance(result.confidence, int)
        assert 0 <= result.confidence <= 100

    def test_confidence_range(self):
        """Test confidence varies based on composite score."""
        # Create engine with controlled benchmarks
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        # High correlation benchmark
        base = np.random.randn(100) * 0.01
        algo_returns = pd.Series(base, index=dates)

        bench_returns = {
            'high_corr': pd.Series(base * 0.9 + np.random.randn(100) * 0.001, index=dates),
            'low_corr': pd.Series(np.random.randn(100) * 0.01, index=dates),
        }

        benchmarks = {name: pd.DataFrame({'close': (1 + ret).cumprod()}, index=dates)
                      for name, ret in bench_returns.items()}

        bench_meta = {name: {'asset_class': 'test', 'ticker': name, 'source': 'test'}
                      for name in bench_returns}

        engine = AssetInferenceEngine(
            benchmarks=benchmarks,
            bench_returns=bench_returns,
            bench_meta=bench_meta,
        )

        algo_df = pd.DataFrame({'close': (1 + algo_returns).cumprod()}, index=dates)
        result = engine.infer(algo_df)

        # Should have some confidence since there's a correlated benchmark
        assert result.confidence > 0


class TestAssetInferenceEngineMultiAsset:
    """Tests for multi-asset detection."""

    def test_significant_exposures(self, inference_engine, sample_algo_daily):
        """Test that significant exposures are detected."""
        result = inference_engine.infer(sample_algo_daily)

        assert isinstance(result.significant_exposures, list)
        # If inference succeeded, should have at least one exposure
        if result.predicted_asset != 'unknown':
            assert len(result.significant_exposures) >= 1

    def test_exposure_structure(self, inference_engine, sample_algo_daily):
        """Test AssetExposure structure."""
        result = inference_engine.infer(sample_algo_daily)

        for exp in result.significant_exposures:
            assert isinstance(exp, AssetExposure)
            assert exp.name is not None
            assert exp.asset_class is not None
            assert exp.direction in ['long', 'short', 'neutral']


class TestAssetInferenceEngineDirection:
    """Tests for direction detection."""

    def test_direction_detection(self, inference_engine, sample_algo_daily):
        """Test that direction is detected."""
        result = inference_engine.infer(sample_algo_daily)

        assert result.direction in ['long', 'short/inverse', 'uncorrelated', 'unknown']

    def test_inverse_detection(self):
        """Test inverse/short detection."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        # Create inversely correlated algo
        base = np.random.randn(100) * 0.01
        algo_returns = pd.Series(-base, index=dates)  # Inverse

        bench_returns = {'benchmark': pd.Series(base, index=dates)}
        benchmarks = {name: pd.DataFrame({'close': (1 + ret).cumprod()}, index=dates)
                      for name, ret in bench_returns.items()}
        bench_meta = {'benchmark': {'asset_class': 'test', 'ticker': 'BM', 'source': 'test'}}

        engine = AssetInferenceEngine(
            benchmarks=benchmarks,
            bench_returns=bench_returns,
            bench_meta=bench_meta,
        )

        algo_df = pd.DataFrame({'close': (1 + algo_returns).cumprod()}, index=dates)
        result = engine.infer(algo_df)

        # Should detect inverse relationship
        if result.predicted_asset != 'unknown':
            # At least one exposure should be short/inverse
            directions = [exp.direction for exp in result.significant_exposures]
            assert 'short' in directions or result.direction == 'short/inverse'


# =============================================================================
# Tests for BenchmarkLoader
# =============================================================================

class TestBenchmarkLoader:
    """Tests for BenchmarkLoader class."""

    def test_load_empty_directory(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = BenchmarkLoader(tmpdir)
            benchmarks, bench_returns, bench_meta = loader.load_all()

            assert benchmarks == {}
            assert bench_returns == {}
            assert bench_meta == {}

    def test_load_with_dat_ascii(self):
        """Test loading DAT_ASCII format files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a DAT_ASCII file
            dat_dir = Path(tmpdir) / 'forex'
            dat_dir.mkdir()

            # Create sample data
            dates = pd.date_range('2020-01-01', periods=100, freq='h')
            df = pd.DataFrame({
                'Gmt time': dates.strftime('%Y%m%d %H%M%S'),
                'Open': [100 + i * 0.001 for i in range(100)],
                'High': [100.01 + i * 0.001 for i in range(100)],
                'Low': [99.99 + i * 0.001 for i in range(100)],
                'Close': [100.005 + i * 0.001 for i in range(100)],
                'Volume': [1000] * 100,
            })
            df.to_csv(dat_dir / 'DAT_ASCII_EURUSD_M1_2020.csv', sep=';', index=False)

            loader = BenchmarkLoader(tmpdir)
            benchmarks, bench_returns, bench_meta = loader.load_all()

            assert len(benchmarks) >= 1
            assert 'EURUSD' in benchmarks or any('EUR' in k for k in benchmarks)


# =============================================================================
# Integration tests
# =============================================================================

class TestAssetInferenceIntegration:
    """Integration tests for asset inference."""

    def test_full_pipeline(self, sample_benchmarks):
        """Test full inference pipeline from scratch."""
        benchmarks, bench_returns, bench_meta = sample_benchmarks

        # Create algo that's correlated with SP500
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        sp500_returns = bench_returns['SP500']
        algo_returns = sp500_returns * 0.8 + pd.Series(np.random.randn(252) * 0.005, index=dates)
        algo_prices = 100 * (1 + algo_returns).cumprod()

        algo_daily = pd.DataFrame({
            'open': algo_prices * 0.99,
            'high': algo_prices * 1.01,
            'low': algo_prices * 0.98,
            'close': algo_prices,
        }, index=dates)

        engine = AssetInferenceEngine(
            benchmarks=benchmarks,
            bench_returns=bench_returns,
            bench_meta=bench_meta,
        )

        result = engine.infer(algo_daily)

        # Should identify SP500 as the best match
        assert result.predicted_asset == 'SP500'
        assert result.asset_class == 'indices'
        assert result.confidence > 30

    def test_inference_with_no_correlation(self, sample_benchmarks):
        """Test inference with uncorrelated algo."""
        benchmarks, bench_returns, bench_meta = sample_benchmarks

        # Create completely random algo
        np.random.seed(999)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        algo_returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
        algo_prices = 100 * (1 + algo_returns).cumprod()

        algo_daily = pd.DataFrame({
            'open': algo_prices * 0.99,
            'high': algo_prices * 1.01,
            'low': algo_prices * 0.98,
            'close': algo_prices,
        }, index=dates)

        engine = AssetInferenceEngine(
            benchmarks=benchmarks,
            bench_returns=bench_returns,
            bench_meta=bench_meta,
        )

        result = engine.infer(algo_daily)

        # Should have low confidence for random algo
        assert result.confidence < 50 or result.asset_class == 'unknown'
