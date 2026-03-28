"""
Base runner class for phase scripts.

Provides common functionality:
- Argument parsing
- Logging setup
- Performance monitoring
- Report generation
- Data/output organization

Usage:
    class Phase1Runner(PhaseRunner):
        phase_name = "Phase 1: Data Loading"
        phase_number = 1

        def add_arguments(self, parser):
            parser.add_argument('--sample', type=int)

        def run(self, args):
            with self.step("Load data"):
                ...
"""

import argparse
import json
import logging
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import data_paths, output_paths, ensure_dir, ensure_parent_dir, PROJECT_ROOT
from src.utils.performance_monitor import (
    PhaseMonitor,
    detect_system_gpu,
    get_gpu_backend,
    get_memory_info,
    is_gpu_available,
)
from src.utils.device import DeviceManager


class PhaseRunner(ABC):
    """
    Abstract base class for phase scripts.

    Subclasses must implement:
    - phase_name: Human-readable phase name
    - phase_number: Phase number (1, 2, 3, etc.)
    - run(): Main execution logic
    """

    phase_name: str = "Unknown Phase"
    phase_number: int = 0

    def __init__(self):
        self.args: Optional[argparse.Namespace] = None
        self.monitor: Optional[PhaseMonitor] = None
        self.results: dict = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._run_id: Optional[str] = None  # set once in execute()

        # Paths
        self.dp = data_paths()
        self.op = output_paths()

    def setup_logging(self, verbose: bool = False, log_file: Optional[Path] = None):
        """Configure logging for the script."""
        level = logging.DEBUG if verbose else logging.INFO

        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Clear existing handlers
        root_logger.handlers = []

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        root_logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            ensure_parent_dir(log_file)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )
            file_handler.setFormatter(file_format)
            root_logger.addHandler(file_handler)

        # Reduce noise from libraries
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('numba').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)

    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with common arguments."""
        parser = argparse.ArgumentParser(
            description=f'{self.phase_name}',
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # Common arguments
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be done without executing'
        )
        parser.add_argument(
            '--output-dir', '-o',
            type=str,
            default=None,
            help='Override output directory'
        )
        parser.add_argument(
            '--log-file',
            type=str,
            default=None,
            help='Write logs to file'
        )
        parser.add_argument(
            '--no-report',
            action='store_true',
            help='Skip generating performance report'
        )

        # Add phase-specific arguments
        self.add_arguments(parser)

        return parser

    def add_arguments(self, parser: argparse.ArgumentParser):
        """Override to add phase-specific arguments."""
        pass

    @abstractmethod
    def run(self, args: argparse.Namespace) -> dict:
        """
        Main execution logic. Must be implemented by subclasses.

        Returns:
            dict: Results dictionary to be saved
        """
        pass

    def step(self, name: str):
        """Context manager for monitoring a step."""
        if self.monitor is None:
            raise RuntimeError("Monitor not initialized. Call execute() instead of run() directly.")
        return self.monitor.step(name)

    # ------------------------------------------------------------------
    # Run-tracking infrastructure
    # ------------------------------------------------------------------

    def _phase_root(self) -> Path:
        """Root output directory for this phase (parent of all run subdirs)."""
        if self.phase_number == 1:
            return self.op.data_pipeline.root
        elif self.phase_number == 2:
            return self.op.analysis.root
        elif self.phase_number == 3:
            return self.op.baselines.root
        elif self.phase_number == 4:
            return self.op.environment.root
        elif self.phase_number == 5:
            return self.op.rl_training.root
        elif self.phase_number == 6:
            return self.op.evaluation.root
        elif self.phase_number == 7:
            return self.op.swarm_pso.root
        elif self.phase_number == 8:
            return self.op.swarm_aco.root
        else:
            return self.op.root / f"phase{self.phase_number}"

    def _run_tag(self) -> str:
        """
        Short string summarising the key arguments for this run.
        Included in the run directory name so different invocations produce
        distinct, human-readable folders.  Override in each runner.
        """
        return ""

    def _make_run_id(self) -> str:
        """Build run_id = {YYYYMMDD_HHMMSS}[_{tag}]."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = self._run_tag()
        return f"{ts}_{tag}" if tag else ts

    # ------------------------------------------------------------------
    # Output / report directories
    # ------------------------------------------------------------------

    def get_output_dir(self) -> Path:
        """
        Output directory for data/artifact files.

        Phases 1 and 2: processed data goes to the canonical data/processed/
        locations (fixed paths consumed by downstream phases).
        All other phases: outputs go inside a timestamped run subdir.
        --output-dir always overrides.
        """
        if self.args and self.args.output_dir:
            return Path(self.args.output_dir)

        if self.phase_number == 1:
            return self.dp.processed.root
        elif self.phase_number == 2:
            return self.dp.processed.analysis.root
        else:
            root = self._phase_root()
            run_id = self._run_id or self._make_run_id()
            return root / run_id

    def _report_dir(self) -> Path:
        """
        Directory for phase reports (metrics JSON, results JSON, SUMMARY.md).

        All phases write reports into a timestamped run subdir under outputs/.
        Phases 1-2 keep their canonical *data* files at fixed paths but still
        write reports into a per-run subdir.
        --output-dir always overrides (used by tests).
        """
        if self.args and self.args.output_dir:
            return Path(self.args.output_dir)
        root = self._phase_root()
        run_id = self._run_id or self._make_run_id()
        return root / run_id

    def get_metrics_path(self) -> Path:
        """Get path for saving performance metrics."""
        return self._report_dir() / f"phase{self.phase_number}_metrics.json"

    def get_results_path(self) -> Path:
        """Get path for saving results."""
        return self._report_dir() / f"phase{self.phase_number}_results.json"

    def get_summary_path(self) -> Path:
        """Get path for saving markdown summary."""
        return self._report_dir() / f"PHASE{self.phase_number}_SUMMARY.md"

    def execute(self, args: Optional[list[str]] = None) -> dict:
        """
        Execute the phase with full monitoring and reporting.

        Args:
            args: Command line arguments (uses sys.argv if None)

        Returns:
            dict: Results dictionary
        """
        # Parse arguments
        parser = self.create_parser()
        self.args = parser.parse_args(args)

        # Compute run_id once (timestamp + key-arg tag) so that
        # get_output_dir() / _report_dir() are stable throughout the run.
        # Skip when --output-dir is set explicitly (tests / manual override).
        if not self.args.output_dir:
            self._run_id = self._make_run_id()

        # Setup logging
        log_file = Path(self.args.log_file) if self.args.log_file else None
        self.setup_logging(verbose=self.args.verbose, log_file=log_file)

        # Print header
        self._print_header()
        if self._run_id:
            self.logger.info(f"Run ID: {self._run_id}")

        # Dry run check
        if self.args.dry_run:
            self.logger.info("DRY RUN - No changes will be made")
            self.logger.info(f"Output directory: {self.get_output_dir()}")
            return {}

        # Create output directories
        output_dir = self.get_output_dir()
        ensure_dir(output_dir)

        # For phases 1-2, report dir differs from data output dir
        report_dir = self._report_dir()
        if report_dir != output_dir:
            ensure_dir(report_dir)

        # Write latest_run.txt so any downstream phase can find the latest run
        if self._run_id:
            phase_root = self._phase_root()
            ensure_dir(phase_root)
            (phase_root / "latest_run.txt").write_text(self._run_id)

        # Initialize monitor
        self.monitor = PhaseMonitor(self.phase_name)

        # Run the phase
        try:
            self.results = self.run(self.args)
            self.results['status'] = 'completed'
        except Exception as e:
            self.logger.error(f"Phase failed: {e}")
            self.results['status'] = 'failed'
            self.results['error'] = str(e)
            raise
        finally:
            # Generate performance report
            if not self.args.no_report:
                self._save_reports()

        return self.results

    def _print_header(self):
        """Print phase header with system info."""
        self.logger.info("=" * 70)
        self.logger.info(self.phase_name)
        self.logger.info("=" * 70)
        self.logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Output:  {self.get_output_dir()}")

        mem = get_memory_info()
        self.logger.info(f"Memory:  {mem['rss_mb']:.1f} MB (current)")

        if is_gpu_available():
            self.logger.info(f"GPU:     {get_gpu_backend()} available")
        else:
            system_gpu, source = detect_system_gpu()
            if system_gpu:
                device_info = DeviceManager.get_device_info()
                detail = device_info.diagnostic_message or "System GPU detected but runtime CUDA is unavailable"
                self.logger.info(f"GPU:     System GPU detected via {source}, but torch runtime is CPU-only")
                self.logger.info(f"GPU:     {detail}")
            else:
                self.logger.info("GPU:     Not available")

        self.logger.info("=" * 70)

    def _save_reports(self):
        """Save performance metrics and results."""
        # Save performance metrics
        metrics_path = self.get_metrics_path()
        ensure_parent_dir(metrics_path)
        self.monitor.save_report(metrics_path)

        # Save results
        results_path = self.get_results_path()
        ensure_parent_dir(results_path)

        # Add metadata to results
        self.results['timestamp'] = datetime.now().isoformat()
        self.results['phase'] = self.phase_number
        self.results['phase_name'] = self.phase_name

        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        self.logger.info(f"Results saved to: {results_path}")

        # Print summary
        print(self.monitor.format_summary())


class QuickRunner:
    """
    Simple runner for utility scripts that don't need full phase infrastructure.

    Usage:
        runner = QuickRunner("Data Migration")
        with runner:
            # do work
        print(runner.format_summary())
    """

    def __init__(self, name: str):
        self.name = name
        self.start_time = datetime.now()
        self.logger = logging.getLogger(name)

        # Setup basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )

    def __enter__(self):
        self.logger.info("=" * 60)
        self.logger.info(self.name)
        self.logger.info("=" * 60)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info("-" * 60)
        self.logger.info(f"Completed in {self._format_duration(elapsed)}")
        return False

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        else:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.0f}s"
