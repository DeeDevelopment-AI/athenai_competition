"""
Trial tracking system for Phase 3 backtesting.

Logs all backtest trials to CSV and JSON files for reproducibility
and analysis.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


@dataclass
class TrialConfig:
    """Configuration for a single trial."""

    baseline_name: str
    feature_set: list[str]
    lookback_window: int
    rebalance_frequency: str
    max_weight: float = 0.40
    min_weight: float = 0.00
    max_turnover: float = 0.30
    max_exposure: float = 1.0
    shrinkage: float = 0.1
    extra_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary with numpy type handling."""
        d = asdict(self)
        d['feature_set'] = ','.join(self.feature_set) if self.feature_set else ''
        # Convert numpy types in extra_params
        if d.get('extra_params'):
            d['extra_params'] = {
                k: (int(v) if isinstance(v, np.integer) else
                    float(v) if isinstance(v, np.floating) else v)
                for k, v in d['extra_params'].items()
            }
        # Convert numpy types in main fields
        for key in ['lookback_window', 'max_weight', 'min_weight', 'max_turnover', 'max_exposure', 'shrinkage']:
            if key in d and isinstance(d[key], (np.integer, np.floating)):
                d[key] = float(d[key]) if isinstance(d[key], np.floating) else int(d[key])
        return d


@dataclass
class TrialMetrics:
    """Metrics from a trial."""

    # Training metrics
    sharpe_train: float = np.nan
    return_ann_train: float = np.nan
    vol_ann_train: float = np.nan
    max_dd_train: float = np.nan
    sortino_train: float = np.nan
    calmar_train: float = np.nan

    # Validation metrics
    sharpe_val: float = np.nan
    return_ann_val: float = np.nan
    vol_ann_val: float = np.nan
    max_dd_val: float = np.nan
    sortino_val: float = np.nan
    calmar_val: float = np.nan

    # Test metrics (out-of-sample)
    sharpe_test: float = np.nan
    return_ann_test: float = np.nan
    vol_ann_test: float = np.nan
    max_dd_test: float = np.nan
    sortino_test: float = np.nan
    calmar_test: float = np.nan

    # Relative metrics
    excess_return: float = np.nan
    tracking_error: float = np.nan
    information_ratio: float = np.nan
    beta_vs_benchmark: float = np.nan
    alpha_vs_benchmark: float = np.nan

    # Operational metrics
    turnover_ann: float = np.nan
    avg_holding_period: float = np.nan
    concentration_hhi: float = np.nan
    transaction_cost_estimate: float = np.nan

    def to_dict(self) -> dict:
        """Convert to dictionary with NaN handling."""
        d = asdict(self)
        # Convert NaN to None for JSON serialization
        for k, v in d.items():
            if isinstance(v, float) and np.isnan(v):
                d[k] = None
        return d


@dataclass
class Trial:
    """Complete trial record."""

    trial_id: int
    timestamp: str
    config: TrialConfig
    metrics: TrialMetrics
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: Optional[str] = None
    test_end: Optional[str] = None
    weight_history_path: Optional[str] = None
    notes: str = ""
    warnings: list[str] = field(default_factory=list)
    status: str = "completed"
    elapsed_seconds: float = 0.0

    def to_flat_dict(self) -> dict:
        """Convert to flat dictionary for CSV."""
        d = {
            'trial_id': self.trial_id,
            'timestamp': self.timestamp,
            'status': self.status,
            'elapsed_seconds': self.elapsed_seconds,
            'train_start': self.train_start,
            'train_end': self.train_end,
            'val_start': self.val_start,
            'val_end': self.val_end,
            'test_start': self.test_start,
            'test_end': self.test_end,
            'notes': self.notes,
            'n_warnings': len(self.warnings),
        }
        # Add config fields
        d.update({f'config_{k}': v for k, v in self.config.to_dict().items()})
        # Add metrics
        d.update(self.metrics.to_dict())
        return d

    def to_full_dict(self) -> dict:
        """Convert to full dictionary for JSON."""
        return {
            'trial_id': self.trial_id,
            'timestamp': self.timestamp,
            'status': self.status,
            'elapsed_seconds': self.elapsed_seconds,
            'train_start': self.train_start,
            'train_end': self.train_end,
            'val_start': self.val_start,
            'val_end': self.val_end,
            'test_start': self.test_start,
            'test_end': self.test_end,
            'notes': self.notes,
            'warnings': self.warnings,
            'config': self.config.to_dict(),
            'metrics': self.metrics.to_dict(),
            'weight_history_path': self.weight_history_path,
        }


class TrialTracker:
    """
    Tracks all backtest trials for Phase 3.

    Saves trials to:
    - outputs/reports/phase3_trials.csv (summary)
    - outputs/reports/trials/trial_{id}.json (detailed)
    """

    def __init__(
        self,
        output_dir: str = "outputs/reports",
        csv_filename: str = "phase3_trials.csv",
    ):
        self.output_dir = Path(output_dir)
        self.trials_dir = self.output_dir / "trials"
        self.csv_path = self.output_dir / csv_filename

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trials_dir.mkdir(parents=True, exist_ok=True)

        # Load existing trials or create new
        self.trials: list[Trial] = []
        self._load_existing()

    def _load_existing(self):
        """Load existing trials from CSV or JSON files."""
        self._next_id = 1

        # Try loading from CSV first
        if self.csv_path.exists():
            try:
                # Use on_bad_lines to skip corrupted rows
                df = pd.read_csv(self.csv_path, on_bad_lines='skip')
                if len(df) > 0 and 'trial_id' in df.columns:
                    self._next_id = int(df['trial_id'].max()) + 1
                    logger.info(f"Loaded {len(df)} existing trials from {self.csv_path}")
                    return
            except Exception as e:
                logger.warning(f"Could not load CSV trials: {e}")

        # Fallback: scan JSON trial files to get max ID
        if self.trials_dir.exists():
            try:
                json_files = list(self.trials_dir.glob("trial_*.json"))
                if json_files:
                    # Extract trial IDs from filenames
                    ids = []
                    for f in json_files:
                        try:
                            # Format: trial_0001.json
                            trial_id = int(f.stem.split('_')[1])
                            ids.append(trial_id)
                        except (ValueError, IndexError):
                            pass
                    if ids:
                        self._next_id = max(ids) + 1
                        logger.info(f"Found {len(ids)} existing trial JSON files, next ID: {self._next_id}")
            except Exception as e:
                logger.warning(f"Could not scan trial JSON files: {e}")

    def create_trial(
        self,
        config: TrialConfig,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
    ) -> Trial:
        """Create a new trial record."""
        trial = Trial(
            trial_id=self._next_id,
            timestamp=datetime.now().isoformat(),
            config=config,
            metrics=TrialMetrics(),
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
        )
        self._next_id += 1
        return trial

    def save_trial(self, trial: Trial):
        """Save a completed trial to CSV and JSON."""
        # Save to JSON
        json_path = self.trials_dir / f"trial_{trial.trial_id:04d}.json"
        with open(json_path, 'w') as f:
            json.dump(trial.to_full_dict(), f, indent=2, cls=NumpyEncoder)
        logger.info(f"Saved trial {trial.trial_id} to {json_path}")

        # Append to CSV
        flat = trial.to_flat_dict()
        df_row = pd.DataFrame([flat])

        if self.csv_path.exists():
            df_row.to_csv(self.csv_path, mode='a', header=False, index=False)
        else:
            df_row.to_csv(self.csv_path, index=False)

        self.trials.append(trial)
        logger.info(f"Appended trial {trial.trial_id} to {self.csv_path}")

    def log_warning(self, trial: Trial, warning: str):
        """Add a warning to a trial."""
        trial.warnings.append(warning)
        logger.warning(f"Trial {trial.trial_id}: {warning}")

    def get_trials_df(self) -> pd.DataFrame:
        """Get all trials as a DataFrame."""
        if self.csv_path.exists():
            return pd.read_csv(self.csv_path)
        return pd.DataFrame()

    def get_best_trials(
        self,
        metric: str = "sharpe_val",
        n: int = 3,
        baseline: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get top n trials by a metric."""
        df = self.get_trials_df()
        if df.empty:
            return df

        if baseline:
            df = df[df['config_baseline_name'] == baseline]

        return df.nlargest(n, metric)

    def summary(self) -> dict:
        """Generate summary statistics."""
        df = self.get_trials_df()
        if df.empty:
            return {'n_trials': 0}

        summary = {
            'n_trials': len(df),
            'n_completed': len(df[df['status'] == 'completed']),
            'n_failed': len(df[df['status'] == 'failed']),
            'baselines': df['config_baseline_name'].nunique(),
            'baseline_counts': df['config_baseline_name'].value_counts().to_dict(),
            'best_sharpe_val': df['sharpe_val'].max(),
            'best_trial_id': int(df.loc[df['sharpe_val'].idxmax(), 'trial_id']) if not df['sharpe_val'].isna().all() else None,
        }
        return summary

    def print_summary(self):
        """Print a formatted summary."""
        s = self.summary()
        print("\n" + "=" * 60)
        print("PHASE 3 TRIAL TRACKER SUMMARY")
        print("=" * 60)
        print(f"Total trials:     {s.get('n_trials', 0)}")
        print(f"Completed:        {s.get('n_completed', 0)}")
        print(f"Failed:           {s.get('n_failed', 0)}")
        print(f"Unique baselines: {s.get('baselines', 0)}")
        print("\nTrials per baseline:")
        for name, count in s.get('baseline_counts', {}).items():
            print(f"  {name}: {count}")
        if s.get('best_sharpe_val') is not None:
            print(f"\nBest validation Sharpe: {s['best_sharpe_val']:.4f} (trial {s['best_trial_id']})")
        print("=" * 60 + "\n")
