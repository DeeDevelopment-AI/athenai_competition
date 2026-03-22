"""
Detección de regímenes de mercado basado en Investment Clock.

4 Fases del ciclo:
- EXPANSION: Crecimiento alto + momentum positivo → Risk-on agresivo
- SLOWDOWN: Crecimiento desacelerando + volatilidad subiendo → Transición a risk-off
- RECESSION: Crecimiento negativo + alta volatilidad → Risk-off / Crisis
- RECOVERY: Crecimiento mejorando + baja volatilidad → Transición a risk-on

Métodos disponibles:
- HEURISTIC: Reglas simples (vol + tendencia)
- FUZZY: Lógica difusa con funciones de membresía
- HMM: Hidden Markov Model multivariado
- CLUSTERING: GMM sobre features de mercado
"""

import logging
from enum import Enum
from typing import Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class RegimeMethod(Enum):
    """Métodos de detección de régimen."""
    HEURISTIC = "heuristic"
    FUZZY = "fuzzy"
    HMM = "hmm"
    CLUSTERING = "clustering"


# Investment Clock regime names
REGIME_EXPANSION = "expansion"
REGIME_SLOWDOWN = "slowdown"
REGIME_RECESSION = "recession"
REGIME_RECOVERY = "recovery"
REGIME_UNKNOWN = "unknown"

INVESTMENT_CLOCK_REGIMES = [REGIME_EXPANSION, REGIME_SLOWDOWN, REGIME_RECESSION, REGIME_RECOVERY]


@dataclass
class RegimeFeatures:
    """Features calculadas para detección de régimen."""
    trend: pd.Series          # Tendencia de retornos (proxy de crecimiento)
    volatility: pd.Series     # Volatilidad realizada
    momentum: pd.Series       # Cambio en momentum (aceleración)
    vol_change: pd.Series     # Cambio en volatilidad


class RegimeDetector:
    """
    Detecta regímenes de mercado usando distintos métodos.

    Basado en el modelo Investment Clock con 4 fases:
    - Expansion: crecimiento + baja volatilidad + momentum positivo
    - Slowdown: crecimiento decelerando + volatilidad subiendo
    - Recession: contracción + alta volatilidad
    - Recovery: mejora desde mínimos + volatilidad cayendo

    Uso:
        detector = RegimeDetector(method=RegimeMethod.FUZZY)
        regimes = detector.detect(returns)
    """

    def __init__(
        self,
        method: RegimeMethod = RegimeMethod.HEURISTIC,
        n_regimes: int = 4,
        # Window parameters
        vol_window: int = 21,
        trend_window: int = 63,
        momentum_window: int = 21,
        # Smoothing
        smoothing_window: int = 5,
    ):
        self.method = method
        self.n_regimes = n_regimes
        self.vol_window = vol_window
        self.trend_window = trend_window
        self.momentum_window = momentum_window
        self.smoothing_window = smoothing_window
        self._model = None
        self._features = None

    def detect(
        self,
        returns: pd.Series,
        features: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Detecta regímenes en la serie de retornos.

        Args:
            returns: Serie de retornos diarios.
            features: Features adicionales (opcional).

        Returns:
            Serie con etiquetas de régimen.
        """
        # Calcular features base
        self._features = self._calculate_features(returns)

        if self.method == RegimeMethod.HEURISTIC:
            regimes = self._detect_heuristic(returns)
        elif self.method == RegimeMethod.FUZZY:
            regimes = self._detect_fuzzy(returns)
        elif self.method == RegimeMethod.HMM:
            regimes = self._detect_hmm(returns)
        elif self.method == RegimeMethod.CLUSTERING:
            regimes = self._detect_clustering(returns)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Suavizar regímenes para evitar cambios muy frecuentes
        if self.smoothing_window > 1:
            regimes = self._smooth_regimes(regimes)

        return regimes

    def _calculate_features(self, returns: pd.Series) -> RegimeFeatures:
        """Calcula features para detección de régimen."""
        # Volatilidad realizada (rolling)
        volatility = returns.rolling(self.vol_window).std() * np.sqrt(252)

        # Tendencia (retorno rolling anualizado)
        trend = returns.rolling(self.trend_window).mean() * 252

        # Momentum: diferencia entre tendencia corta y larga
        short_trend = returns.rolling(self.momentum_window).mean() * 252
        long_trend = returns.rolling(self.trend_window).mean() * 252
        momentum = short_trend - long_trend

        # Cambio en volatilidad (derivada)
        vol_change = volatility.diff(self.momentum_window)

        return RegimeFeatures(
            trend=trend,
            volatility=volatility,
            momentum=momentum,
            vol_change=vol_change,
        )

    def _detect_heuristic(self, returns: pd.Series) -> pd.Series:
        """
        Clasificación heurística basada en volatilidad y tendencia.

        Mapping a Investment Clock:
        - risk_on_low_vol → expansion
        - risk_on_high_vol → slowdown (crecimiento pero con riesgo)
        - risk_off_low_vol → recovery (bajo rendimiento pero estable)
        - risk_off_high_vol → recession (crisis)
        """
        f = self._features

        # Umbrales (percentiles históricos)
        vol_threshold = f.volatility.expanding().quantile(0.5)
        trend_threshold = 0  # Positivo vs negativo

        # Clasificar
        high_vol = f.volatility > vol_threshold
        positive_trend = f.trend > trend_threshold

        regimes = pd.Series(index=returns.index, dtype=str)
        regimes[positive_trend & ~high_vol] = REGIME_EXPANSION
        regimes[positive_trend & high_vol] = REGIME_SLOWDOWN
        regimes[~positive_trend & ~high_vol] = REGIME_RECOVERY
        regimes[~positive_trend & high_vol] = REGIME_RECESSION

        # Rellenar NaN iniciales
        regimes = regimes.fillna(REGIME_UNKNOWN)

        return regimes

    def _detect_fuzzy(self, returns: pd.Series) -> pd.Series:
        """
        Detección de régimen con lógica difusa (fuzzy logic).

        Funciones de membresía para:
        - Trend: negative, neutral, positive
        - Volatility: low, medium, high
        - Momentum: falling, stable, rising
        - Vol Change: decreasing, stable, increasing

        Reglas difusas para cada fase del Investment Clock.
        """
        f = self._features

        # Calcular percentiles para normalización adaptativa
        trend_q25 = f.trend.expanding().quantile(0.25)
        trend_q75 = f.trend.expanding().quantile(0.75)
        vol_q25 = f.volatility.expanding().quantile(0.25)
        vol_q75 = f.volatility.expanding().quantile(0.75)
        mom_q25 = f.momentum.expanding().quantile(0.25)
        mom_q75 = f.momentum.expanding().quantile(0.75)

        regimes = pd.Series(index=returns.index, dtype=str)

        for i, date in enumerate(returns.index):
            if pd.isna(f.trend.iloc[i]) or pd.isna(f.volatility.iloc[i]):
                regimes.iloc[i] = REGIME_UNKNOWN
                continue

            # Get current values
            trend_val = f.trend.iloc[i]
            vol_val = f.volatility.iloc[i]
            mom_val = f.momentum.iloc[i] if not pd.isna(f.momentum.iloc[i]) else 0
            vol_chg = f.vol_change.iloc[i] if not pd.isna(f.vol_change.iloc[i]) else 0

            # Get adaptive thresholds
            t_q25 = trend_q25.iloc[i] if not pd.isna(trend_q25.iloc[i]) else -0.05
            t_q75 = trend_q75.iloc[i] if not pd.isna(trend_q75.iloc[i]) else 0.05
            v_q25 = vol_q25.iloc[i] if not pd.isna(vol_q25.iloc[i]) else 0.10
            v_q75 = vol_q75.iloc[i] if not pd.isna(vol_q75.iloc[i]) else 0.20
            m_q25 = mom_q25.iloc[i] if not pd.isna(mom_q25.iloc[i]) else -0.02
            m_q75 = mom_q75.iloc[i] if not pd.isna(mom_q75.iloc[i]) else 0.02

            # Fuzzy membership functions (using trapezoidal membership)
            # Trend memberships
            trend_positive = self._fuzzy_high(trend_val, t_q25, t_q75)
            trend_negative = self._fuzzy_low(trend_val, t_q25, t_q75)

            # Volatility memberships
            vol_high = self._fuzzy_high(vol_val, v_q25, v_q75)
            vol_low = self._fuzzy_low(vol_val, v_q25, v_q75)

            # Momentum memberships
            mom_rising = self._fuzzy_high(mom_val, m_q25, m_q75)
            mom_falling = self._fuzzy_low(mom_val, m_q25, m_q75)

            # Volatility change (is vol increasing or decreasing?)
            vol_increasing = 1.0 if vol_chg > 0.001 else (0.5 if vol_chg > -0.001 else 0.0)
            vol_decreasing = 1.0 if vol_chg < -0.001 else (0.5 if vol_chg < 0.001 else 0.0)

            # Fuzzy rules for Investment Clock phases
            # EXPANSION: positive trend + low vol + rising momentum
            expansion_score = min(trend_positive, vol_low, max(mom_rising, 0.5))

            # SLOWDOWN: positive trend + high vol OR rising vol + falling momentum
            slowdown_score = min(trend_positive, max(vol_high, vol_increasing * 0.7), mom_falling)
            # Alternative: strong positive trend but vol spiking
            slowdown_score = max(slowdown_score, min(trend_positive * 0.8, vol_high, vol_increasing))

            # RECESSION: negative trend + high vol
            recession_score = min(trend_negative, vol_high)

            # RECOVERY: improving momentum + low/decreasing vol + trend improving
            recovery_score = min(max(trend_negative * 0.5, 0.3), vol_low, mom_rising)
            # Alternative: trend still negative but momentum turning
            recovery_score = max(recovery_score, min(trend_negative, vol_decreasing, mom_rising))

            # Select regime with highest score
            scores = {
                REGIME_EXPANSION: expansion_score,
                REGIME_SLOWDOWN: slowdown_score,
                REGIME_RECESSION: recession_score,
                REGIME_RECOVERY: recovery_score,
            }

            best_regime = max(scores, key=scores.get)
            regimes.iloc[i] = best_regime

        return regimes

    def _fuzzy_high(self, value: float, q25: float, q75: float) -> float:
        """Fuzzy membership for 'high' - trapezoidal."""
        if value >= q75:
            return 1.0
        elif value <= q25:
            return 0.0
        else:
            return (value - q25) / (q75 - q25)

    def _fuzzy_low(self, value: float, q25: float, q75: float) -> float:
        """Fuzzy membership for 'low' - trapezoidal."""
        if value <= q25:
            return 1.0
        elif value >= q75:
            return 0.0
        else:
            return (q75 - value) / (q75 - q25)

    def _detect_hmm(self, returns: pd.Series) -> pd.Series:
        """
        Detección de regímenes con Hidden Markov Model multivariado.

        Usa múltiples features: returns, volatility, momentum.
        Los estados se ordenan según características para mapear a Investment Clock.
        """
        try:
            from hmmlearn import hmm
        except ImportError:
            logger.error("hmmlearn not installed. Install with: pip install hmmlearn")
            return self._detect_fuzzy(returns)

        f = self._features

        # Construir matriz de features multivariada
        features_df = pd.DataFrame({
            'returns': returns,
            'volatility': f.volatility,
            'trend': f.trend,
            'momentum': f.momentum,
        }).dropna()

        if len(features_df) < 100:
            logger.warning("Not enough data for HMM. Falling back to fuzzy.")
            return self._detect_fuzzy(returns)

        # Normalizar features
        scaler = StandardScaler()
        X = scaler.fit_transform(features_df.values)

        # Ajustar HMM con 4 componentes
        n_components = 4
        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=200,
            random_state=42,
            init_params="stmc",
        )

        try:
            model.fit(X)
            self._model = model

            # Predecir estados
            states = model.predict(X)

            # Caracterizar cada estado para asignar nombres
            state_profiles = []
            for state_id in range(n_components):
                state_mask = states == state_id
                if state_mask.sum() > 0:
                    state_features = features_df.iloc[state_mask]
                    state_profiles.append({
                        'state_id': state_id,
                        'mean_trend': state_features['trend'].mean(),
                        'mean_vol': state_features['volatility'].mean(),
                        'mean_momentum': state_features['momentum'].mean(),
                        'count': state_mask.sum(),
                    })

            # Mapear estados a regímenes del Investment Clock
            state_mapping = self._map_hmm_states_to_regimes(state_profiles)

            # Aplicar mapping
            named_states = [state_mapping.get(s, REGIME_UNKNOWN) for s in states]

            regimes = pd.Series(named_states, index=features_df.index)
            regimes = regimes.reindex(returns.index).fillna(REGIME_UNKNOWN)

            logger.info(f"HMM fitted with {n_components} states")
            for profile in state_profiles:
                regime = state_mapping.get(profile['state_id'], 'unknown')
                logger.info(
                    f"  State {profile['state_id']} -> {regime}: "
                    f"trend={profile['mean_trend']:.2%}, vol={profile['mean_vol']:.2%}, "
                    f"momentum={profile['mean_momentum']:.4f}, n={profile['count']}"
                )

            return regimes

        except Exception as e:
            logger.error(f"HMM fitting failed: {e}. Falling back to fuzzy.")
            return self._detect_fuzzy(returns)

    def _map_hmm_states_to_regimes(self, state_profiles: list[dict]) -> dict[int, str]:
        """
        Mapea estados HMM a regímenes del Investment Clock.

        Lógica:
        - EXPANSION: alto trend, baja vol, momentum positivo
        - SLOWDOWN: trend positivo pero vol subiendo o momentum cayendo
        - RECESSION: trend negativo, alta vol
        - RECOVERY: trend mejorando (momentum positivo), vol baja
        """
        if not state_profiles:
            return {}

        # Calcular scores para cada régimen por estado
        n_states = len(state_profiles)

        # Normalizar métricas
        trends = [p['mean_trend'] for p in state_profiles]
        vols = [p['mean_vol'] for p in state_profiles]
        moms = [p['mean_momentum'] for p in state_profiles]

        trend_min, trend_max = min(trends), max(trends)
        vol_min, vol_max = min(vols), max(vols)
        mom_min, mom_max = min(moms), max(moms)

        def normalize(val, vmin, vmax):
            if vmax == vmin:
                return 0.5
            return (val - vmin) / (vmax - vmin)

        # Score cada estado para cada régimen
        state_scores = {}
        for p in state_profiles:
            sid = p['state_id']
            t_norm = normalize(p['mean_trend'], trend_min, trend_max)
            v_norm = normalize(p['mean_vol'], vol_min, vol_max)
            m_norm = normalize(p['mean_momentum'], mom_min, mom_max)

            # EXPANSION: high trend, low vol, high momentum
            exp_score = t_norm * 0.4 + (1 - v_norm) * 0.3 + m_norm * 0.3

            # SLOWDOWN: high trend, high vol, low momentum
            slow_score = t_norm * 0.3 + v_norm * 0.4 + (1 - m_norm) * 0.3

            # RECESSION: low trend, high vol
            rec_score = (1 - t_norm) * 0.5 + v_norm * 0.5

            # RECOVERY: low/medium trend, low vol, high momentum
            recov_score = (1 - t_norm) * 0.2 + (1 - v_norm) * 0.4 + m_norm * 0.4

            state_scores[sid] = {
                REGIME_EXPANSION: exp_score,
                REGIME_SLOWDOWN: slow_score,
                REGIME_RECESSION: rec_score,
                REGIME_RECOVERY: recov_score,
            }

        # Asignar regímenes de forma greedy (cada régimen se asigna una vez)
        mapping = {}
        used_regimes = set()
        used_states = set()

        # Ordenar por mejor score
        all_assignments = []
        for sid, scores in state_scores.items():
            for regime, score in scores.items():
                all_assignments.append((score, sid, regime))

        all_assignments.sort(reverse=True)

        for score, sid, regime in all_assignments:
            if sid not in used_states and regime not in used_regimes:
                mapping[sid] = regime
                used_states.add(sid)
                used_regimes.add(regime)

            if len(mapping) == min(n_states, 4):
                break

        # Asignar estados restantes al régimen más probable
        for p in state_profiles:
            sid = p['state_id']
            if sid not in mapping:
                # Asignar al régimen disponible con mejor score
                best_regime = max(state_scores[sid], key=state_scores[sid].get)
                mapping[sid] = best_regime

        return mapping

    def _detect_clustering(self, returns: pd.Series) -> pd.Series:
        """
        Detección de regímenes con GMM clustering sobre features.
        """
        from sklearn.mixture import GaussianMixture

        f = self._features

        # Construir features
        features_df = pd.DataFrame({
            'trend': f.trend,
            'volatility': f.volatility,
            'momentum': f.momentum,
            'vol_change': f.vol_change,
        }).dropna()

        if len(features_df) < 50:
            logger.warning("Not enough data for clustering. Falling back to fuzzy.")
            return self._detect_fuzzy(returns)

        # Normalizar
        scaler = StandardScaler()
        X = scaler.fit_transform(features_df.values)

        # GMM
        model = GaussianMixture(
            n_components=4,
            covariance_type="full",
            random_state=42,
            n_init=5,
        )
        model.fit(X)
        self._model = model

        # Predecir
        states = model.predict(X)

        # Caracterizar y mapear (similar a HMM)
        state_profiles = []
        for state_id in range(4):
            state_mask = states == state_id
            if state_mask.sum() > 0:
                state_features = features_df.iloc[state_mask]
                state_profiles.append({
                    'state_id': state_id,
                    'mean_trend': state_features['trend'].mean(),
                    'mean_vol': state_features['volatility'].mean(),
                    'mean_momentum': state_features['momentum'].mean(),
                    'count': state_mask.sum(),
                })

        state_mapping = self._map_hmm_states_to_regimes(state_profiles)
        named_states = [state_mapping.get(s, REGIME_UNKNOWN) for s in states]

        regimes = pd.Series(named_states, index=features_df.index)
        regimes = regimes.reindex(returns.index).fillna(REGIME_UNKNOWN)

        logger.info(f"Clustering fitted with 4 regimes")
        return regimes

    def _smooth_regimes(self, regimes: pd.Series) -> pd.Series:
        """
        Suaviza regímenes para evitar cambios muy frecuentes.
        Usa voting de mayoría en ventana rolling.
        """
        def mode_in_window(window):
            if len(window) == 0:
                return REGIME_UNKNOWN
            counts = window.value_counts()
            return counts.idxmax()

        # Rolling.apply expects numeric data; regimes are strings, so use a manual window mode.
        result = []
        half_window = self.smoothing_window // 2

        for i in range(len(regimes)):
            start = max(0, i - half_window)
            end = min(len(regimes), i + half_window + 1)
            window = regimes.iloc[start:end]
            result.append(mode_in_window(window))

        return pd.Series(result, index=regimes.index)

    def get_regime_statistics(
        self, returns: pd.Series, regimes: pd.Series
    ) -> pd.DataFrame:
        """
        Calcula estadísticas por régimen.
        """
        aligned = pd.concat([returns, regimes], axis=1, join="inner")
        aligned.columns = ["returns", "regime"]

        stats = []
        for regime in INVESTMENT_CLOCK_REGIMES:
            regime_returns = aligned[aligned["regime"] == regime]["returns"]
            n_days = len(regime_returns)

            if n_days < 5:
                continue

            stats.append({
                "regime": regime,
                "n_days": n_days,
                "pct_time": n_days / len(aligned) * 100,
                "ann_return": regime_returns.mean() * 252,
                "ann_vol": regime_returns.std() * np.sqrt(252),
                "sharpe": (
                    regime_returns.mean() / regime_returns.std() * np.sqrt(252)
                    if regime_returns.std() > 0 else 0
                ),
                "max_dd": self._max_drawdown(regime_returns),
                "avg_duration_days": self._avg_regime_duration(regimes, regime),
            })

        # Ordenar por el ciclo del Investment Clock
        regime_order = {r: i for i, r in enumerate(INVESTMENT_CLOCK_REGIMES)}
        stats_df = pd.DataFrame(stats)
        if not stats_df.empty:
            stats_df['order'] = stats_df['regime'].map(regime_order)
            stats_df = stats_df.sort_values('order').drop('order', axis=1)
            stats_df = stats_df.set_index("regime")

        return stats_df

    def _max_drawdown(self, returns: pd.Series) -> float:
        """Calcula max drawdown."""
        if len(returns) == 0:
            return 0.0
        equity = (1 + returns).cumprod()
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        return drawdown.min()

    def _avg_regime_duration(self, regimes: pd.Series, target_regime: str) -> float:
        """Calcula duración media de un régimen."""
        is_target = regimes == target_regime
        durations = []

        current_duration = 0
        for in_regime in is_target:
            if in_regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0

        if current_duration > 0:
            durations.append(current_duration)

        return np.mean(durations) if durations else 0.0

    def get_transition_matrix(self, regimes: pd.Series) -> pd.DataFrame:
        """
        Calcula matriz de transición entre regímenes.
        """
        regimes = regimes[regimes != REGIME_UNKNOWN]

        # Usar orden del Investment Clock
        regime_order = [r for r in INVESTMENT_CLOCK_REGIMES if r in regimes.values]

        transitions = pd.DataFrame(
            0, index=regime_order, columns=regime_order, dtype=float
        )

        for i in range(len(regimes) - 1):
            from_regime = regimes.iloc[i]
            to_regime = regimes.iloc[i + 1]
            if from_regime in regime_order and to_regime in regime_order:
                transitions.loc[from_regime, to_regime] += 1

        # Normalizar por filas
        row_sums = transitions.sum(axis=1)
        transitions = transitions.div(row_sums, axis=0).fillna(0)

        return transitions

    def get_current_regime(self, regimes: pd.Series, lookback: int = 5) -> str:
        """
        Obtiene el régimen actual (moda de últimos N días).
        """
        recent = regimes.tail(lookback)
        if len(recent) == 0:
            return REGIME_UNKNOWN
        return recent.value_counts().idxmax()

    def plot_regimes(
        self,
        returns: pd.Series,
        regimes: pd.Series,
        figsize: tuple = (14, 8),
    ):
        """
        Visualiza regímenes sobre equity curve.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # Equity curve
        ax1 = axes[0]
        equity = (1 + returns).cumprod() * 100
        ax1.plot(equity.index, equity.values, 'k-', linewidth=1, alpha=0.7)

        # Colorear fondo por régimen
        regime_colors = {
            REGIME_EXPANSION: 'lightgreen',
            REGIME_SLOWDOWN: 'khaki',
            REGIME_RECESSION: 'lightcoral',
            REGIME_RECOVERY: 'lightblue',
            REGIME_UNKNOWN: 'lightgray',
        }

        for regime, color in regime_colors.items():
            mask = regimes == regime
            if mask.any():
                ax1.fill_between(
                    equity.index, equity.min(), equity.max(),
                    where=mask.reindex(equity.index).fillna(False),
                    alpha=0.3, color=color, label=regime
                )

        ax1.set_ylabel('Equity')
        ax1.set_title('Equity Curve con Regímenes (Investment Clock)')
        ax1.legend(loc='upper left', fontsize=8)

        # Volatilidad
        ax2 = axes[1]
        vol = returns.rolling(21).std() * np.sqrt(252)
        ax2.plot(vol.index, vol.values, 'b-', linewidth=1)
        ax2.axhline(vol.median(), color='red', linestyle='--', alpha=0.7)
        ax2.set_ylabel('Volatilidad (21d)')
        ax2.set_title('Volatilidad Realizada')

        # Trend
        ax3 = axes[2]
        trend = returns.rolling(63).mean() * 252
        ax3.plot(trend.index, trend.values, 'g-', linewidth=1)
        ax3.axhline(0, color='red', linestyle='--', alpha=0.7)
        ax3.set_ylabel('Trend (63d)')
        ax3.set_title('Tendencia Anualizada')

        plt.tight_layout()
        return fig
