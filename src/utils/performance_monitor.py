"""
Performance monitoring utilities for script execution.

Provides comprehensive tracking of:
- Wall time and CPU time
- Memory usage (current, peak, delta)
- GPU memory and utilization (if available)

Usage:
    from src.utils.performance_monitor import PerformanceMonitor, monitor_function

    # Context manager
    with PerformanceMonitor("Phase 1") as pm:
        # Your code here
        pm.checkpoint("Data loading")
        # More code
        pm.checkpoint("Feature engineering")

    # Decorator
    @monitor_function
    def my_expensive_function():
        ...
"""

import gc
import logging
import os
import platform
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

import psutil

logger = logging.getLogger(__name__)


# =============================================================================
# GPU Detection
# =============================================================================

_GPU_AVAILABLE = False
_GPU_BACKEND = None

try:
    import torch
    if torch.cuda.is_available():
        _GPU_AVAILABLE = True
        _GPU_BACKEND = "cuda"
except ImportError:
    pass

if not _GPU_AVAILABLE:
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _GPU_AVAILABLE = True
            _GPU_BACKEND = "mps"
    except (ImportError, AttributeError):
        pass


def is_gpu_available() -> bool:
    """Check if GPU is available."""
    return _GPU_AVAILABLE


def get_gpu_backend() -> Optional[str]:
    """Get the GPU backend name (cuda, mps, or None)."""
    return _GPU_BACKEND


# =============================================================================
# Memory Utilities
# =============================================================================

def get_memory_info() -> dict:
    """Get current memory usage information."""
    process = psutil.Process()
    mem_info = process.memory_info()

    return {
        'rss_mb': mem_info.rss / (1024 * 1024),  # Resident Set Size
        'vms_mb': mem_info.vms / (1024 * 1024),  # Virtual Memory Size
        'percent': process.memory_percent(),
    }


def get_system_memory_info() -> dict:
    """Get system-wide memory information."""
    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / (1024 ** 3),
        'available_gb': mem.available / (1024 ** 3),
        'used_gb': mem.used / (1024 ** 3),
        'percent': mem.percent,
    }


def get_gpu_memory_info() -> Optional[dict]:
    """Get GPU memory information if available."""
    if not _GPU_AVAILABLE:
        return None

    try:
        import torch

        if _GPU_BACKEND == "cuda":
            return {
                'allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
                'reserved_mb': torch.cuda.memory_reserved() / (1024 * 1024),
                'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024),
                'device_name': torch.cuda.get_device_name(0),
                'device_count': torch.cuda.device_count(),
            }
        elif _GPU_BACKEND == "mps":
            # MPS doesn't have detailed memory API
            return {
                'backend': 'mps',
                'available': True,
            }
    except Exception as e:
        logger.debug(f"Failed to get GPU memory info: {e}")

    return None


def force_garbage_collection():
    """Force garbage collection and clear GPU cache if available."""
    gc.collect()

    if _GPU_AVAILABLE:
        try:
            import torch
            if _GPU_BACKEND == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass


# =============================================================================
# Checkpoint Data
# =============================================================================

@dataclass
class Checkpoint:
    """Single checkpoint in execution."""
    name: str
    timestamp: float
    elapsed_since_start: float
    elapsed_since_last: float
    memory_mb: float
    memory_delta_mb: float
    gpu_memory_mb: Optional[float] = None


@dataclass
class PerformanceReport:
    """Complete performance report for a monitored section."""
    name: str
    start_time: datetime
    end_time: datetime
    total_seconds: float
    cpu_time_seconds: float

    # Memory
    memory_start_mb: float
    memory_end_mb: float
    memory_peak_mb: float
    memory_delta_mb: float

    # GPU
    gpu_available: bool
    gpu_backend: Optional[str]
    gpu_memory_start_mb: Optional[float]
    gpu_memory_end_mb: Optional[float]
    gpu_memory_peak_mb: Optional[float]

    # System
    system_info: dict

    # Checkpoints
    checkpoints: list = field(default_factory=list)

    # Errors
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'total_seconds': self.total_seconds,
            'cpu_time_seconds': self.cpu_time_seconds,
            'memory': {
                'start_mb': self.memory_start_mb,
                'end_mb': self.memory_end_mb,
                'peak_mb': self.memory_peak_mb,
                'delta_mb': self.memory_delta_mb,
            },
            'gpu': {
                'available': self.gpu_available,
                'backend': self.gpu_backend,
                'memory_start_mb': self.gpu_memory_start_mb,
                'memory_end_mb': self.gpu_memory_end_mb,
                'memory_peak_mb': self.gpu_memory_peak_mb,
            } if self.gpu_available else None,
            'system': self.system_info,
            'checkpoints': [
                {
                    'name': cp.name,
                    'elapsed_since_start': cp.elapsed_since_start,
                    'elapsed_since_last': cp.elapsed_since_last,
                    'memory_mb': cp.memory_mb,
                    'memory_delta_mb': cp.memory_delta_mb,
                }
                for cp in self.checkpoints
            ],
            'errors': self.errors,
        }

    def format_summary(self) -> str:
        """Format a human-readable summary."""
        lines = [
            "",
            "=" * 70,
            f"PERFORMANCE REPORT: {self.name}",
            "=" * 70,
            "",
            "TIMING",
            "-" * 40,
            f"  Start:      {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  End:        {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Duration:   {self._format_duration(self.total_seconds)}",
            f"  CPU Time:   {self._format_duration(self.cpu_time_seconds)}",
            "",
            "MEMORY",
            "-" * 40,
            f"  Start:      {self.memory_start_mb:,.1f} MB",
            f"  End:        {self.memory_end_mb:,.1f} MB",
            f"  Peak:       {self.memory_peak_mb:,.1f} MB",
            f"  Delta:      {self.memory_delta_mb:+,.1f} MB",
        ]

        if self.gpu_available:
            lines.extend([
                "",
                f"GPU ({self.gpu_backend})",
                "-" * 40,
            ])
            if self.gpu_memory_start_mb is not None:
                lines.extend([
                    f"  Start:      {self.gpu_memory_start_mb:,.1f} MB",
                    f"  End:        {self.gpu_memory_end_mb:,.1f} MB",
                    f"  Peak:       {self.gpu_memory_peak_mb:,.1f} MB",
                ])

        if self.checkpoints:
            lines.extend([
                "",
                "CHECKPOINTS",
                "-" * 40,
            ])
            for cp in self.checkpoints:
                lines.append(
                    f"  [{self._format_duration(cp.elapsed_since_start):>10}] "
                    f"{cp.name:<30} "
                    f"(+{self._format_duration(cp.elapsed_since_last)}, "
                    f"{cp.memory_delta_mb:+.1f} MB)"
                )

        lines.extend([
            "",
            "SYSTEM",
            "-" * 40,
            f"  Platform:   {self.system_info.get('platform', 'N/A')}",
            f"  Python:     {self.system_info.get('python_version', 'N/A')}",
            f"  CPU Cores:  {self.system_info.get('cpu_count', 'N/A')}",
            f"  RAM:        {self.system_info.get('ram_gb', 0):.1f} GB",
        ])

        if self.errors:
            lines.extend([
                "",
                "ERRORS",
                "-" * 40,
            ])
            for error in self.errors:
                lines.append(f"  - {error}")

        lines.append("=" * 70)

        return "\n".join(lines)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 1:
            return f"{seconds * 1000:.1f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.0f}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


# =============================================================================
# Performance Monitor
# =============================================================================

class PerformanceMonitor:
    """
    Context manager for monitoring performance of code sections.

    Example:
        with PerformanceMonitor("My Task") as pm:
            # Do work
            pm.checkpoint("Step 1")
            # More work
            pm.checkpoint("Step 2")

        print(pm.report.format_summary())
    """

    def __init__(
        self,
        name: str,
        log_checkpoints: bool = True,
        collect_garbage: bool = True,
    ):
        self.name = name
        self.log_checkpoints = log_checkpoints
        self.collect_garbage = collect_garbage

        self._start_time: Optional[datetime] = None
        self._start_wall: Optional[float] = None
        self._start_cpu: Optional[float] = None
        self._start_memory: Optional[float] = None
        self._start_gpu_memory: Optional[float] = None

        self._last_checkpoint_time: Optional[float] = None
        self._last_checkpoint_memory: Optional[float] = None

        self._peak_memory: float = 0
        self._peak_gpu_memory: float = 0

        self._checkpoints: list[Checkpoint] = []
        self._errors: list[str] = []

        self.report: Optional[PerformanceReport] = None

    def __enter__(self) -> "PerformanceMonitor":
        """Start monitoring."""
        if self.collect_garbage:
            force_garbage_collection()

        self._start_time = datetime.now()
        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()

        mem_info = get_memory_info()
        self._start_memory = mem_info['rss_mb']
        self._peak_memory = self._start_memory

        self._last_checkpoint_time = self._start_wall
        self._last_checkpoint_memory = self._start_memory

        gpu_info = get_gpu_memory_info()
        if gpu_info and 'allocated_mb' in gpu_info:
            self._start_gpu_memory = gpu_info['allocated_mb']
            self._peak_gpu_memory = self._start_gpu_memory

        logger.info(f"Starting: {self.name}")
        logger.info(f"  Memory: {self._start_memory:.1f} MB")
        if self._start_gpu_memory is not None:
            logger.info(f"  GPU Memory: {self._start_gpu_memory:.1f} MB")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finish monitoring and generate report."""
        if self.collect_garbage:
            force_garbage_collection()

        end_time = datetime.now()
        end_wall = time.perf_counter()
        end_cpu = time.process_time()

        mem_info = get_memory_info()
        end_memory = mem_info['rss_mb']
        self._update_peak_memory()

        gpu_end_memory = None
        gpu_peak_memory = None
        gpu_info = get_gpu_memory_info()
        if gpu_info and 'allocated_mb' in gpu_info:
            gpu_end_memory = gpu_info['allocated_mb']
            gpu_peak_memory = gpu_info.get('max_allocated_mb', self._peak_gpu_memory)

        if exc_val is not None:
            self._errors.append(f"{exc_type.__name__}: {exc_val}")

        self.report = PerformanceReport(
            name=self.name,
            start_time=self._start_time,
            end_time=end_time,
            total_seconds=end_wall - self._start_wall,
            cpu_time_seconds=end_cpu - self._start_cpu,
            memory_start_mb=self._start_memory,
            memory_end_mb=end_memory,
            memory_peak_mb=self._peak_memory,
            memory_delta_mb=end_memory - self._start_memory,
            gpu_available=_GPU_AVAILABLE,
            gpu_backend=_GPU_BACKEND,
            gpu_memory_start_mb=self._start_gpu_memory,
            gpu_memory_end_mb=gpu_end_memory,
            gpu_memory_peak_mb=gpu_peak_memory,
            system_info=self._get_system_info(),
            checkpoints=self._checkpoints.copy(),
            errors=self._errors.copy(),
        )

        logger.info(f"Completed: {self.name}")
        logger.info(f"  Duration: {self.report._format_duration(self.report.total_seconds)}")
        logger.info(f"  Memory delta: {self.report.memory_delta_mb:+.1f} MB")

        return False  # Don't suppress exceptions

    def checkpoint(self, name: str):
        """Record a checkpoint."""
        now = time.perf_counter()
        mem_info = get_memory_info()
        current_memory = mem_info['rss_mb']

        self._update_peak_memory()

        elapsed_since_start = now - self._start_wall
        elapsed_since_last = now - self._last_checkpoint_time
        memory_delta = current_memory - self._last_checkpoint_memory

        gpu_memory = None
        gpu_info = get_gpu_memory_info()
        if gpu_info and 'allocated_mb' in gpu_info:
            gpu_memory = gpu_info['allocated_mb']
            if gpu_memory > self._peak_gpu_memory:
                self._peak_gpu_memory = gpu_memory

        cp = Checkpoint(
            name=name,
            timestamp=now,
            elapsed_since_start=elapsed_since_start,
            elapsed_since_last=elapsed_since_last,
            memory_mb=current_memory,
            memory_delta_mb=memory_delta,
            gpu_memory_mb=gpu_memory,
        )
        self._checkpoints.append(cp)

        self._last_checkpoint_time = now
        self._last_checkpoint_memory = current_memory

        if self.log_checkpoints:
            logger.info(
                f"  Checkpoint: {name} "
                f"(+{self.report._format_duration(elapsed_since_last) if self.report else f'{elapsed_since_last:.1f}s'}, "
                f"{memory_delta:+.1f} MB)"
            )

    def _update_peak_memory(self):
        """Update peak memory tracking."""
        mem_info = get_memory_info()
        if mem_info['rss_mb'] > self._peak_memory:
            self._peak_memory = mem_info['rss_mb']

    @staticmethod
    def _get_system_info() -> dict:
        """Get system information."""
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'ram_gb': psutil.virtual_memory().total / (1024 ** 3),
            'gpu_available': _GPU_AVAILABLE,
            'gpu_backend': _GPU_BACKEND,
        }


# =============================================================================
# Function Decorator
# =============================================================================

def monitor_function(
    name: Optional[str] = None,
    log_result: bool = False,
) -> Callable:
    """
    Decorator to monitor function performance.

    Example:
        @monitor_function()
        def my_function():
            ...

        @monitor_function(name="Custom Name", log_result=True)
        def another_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with PerformanceMonitor(func_name, log_checkpoints=False) as pm:
                result = func(*args, **kwargs)

            if log_result:
                logger.info(f"  Result type: {type(result).__name__}")

            return result

        return wrapper

    return decorator


# =============================================================================
# Step Monitor (for phase scripts)
# =============================================================================

@dataclass
class StepMetrics:
    """Metrics for a single step."""
    name: str
    duration_seconds: float
    memory_start_mb: float
    memory_end_mb: float
    memory_delta_mb: float
    status: str = "completed"
    error: Optional[str] = None


class PhaseMonitor:
    """
    Monitor for multi-step phase execution.

    Example:
        monitor = PhaseMonitor("Phase 1: Data Loading")

        with monitor.step("Load algorithms"):
            load_algorithms()

        with monitor.step("Process data"):
            process_data()

        print(monitor.format_summary())
        monitor.save_report("phase1_metrics.json")
    """

    def __init__(self, phase_name: str):
        self.phase_name = phase_name
        self.start_time = datetime.now()
        self.start_wall = time.perf_counter()
        self.start_memory = get_memory_info()['rss_mb']

        self.steps: list[StepMetrics] = []
        self.current_step: Optional[str] = None

        logger.info("=" * 70)
        logger.info(f"{phase_name}")
        logger.info("=" * 70)
        logger.info(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Initial memory: {self.start_memory:.1f} MB")

    @contextmanager
    def step(self, name: str):
        """Context manager for a single step."""
        self.current_step = name
        step_start = time.perf_counter()
        mem_start = get_memory_info()['rss_mb']

        logger.info(f"\n--- {name} ---")

        error = None
        status = "completed"

        try:
            yield
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            status = "failed"
            logger.error(f"Step failed: {error}")
            raise
        finally:
            step_end = time.perf_counter()
            mem_end = get_memory_info()['rss_mb']

            step_metrics = StepMetrics(
                name=name,
                duration_seconds=step_end - step_start,
                memory_start_mb=mem_start,
                memory_end_mb=mem_end,
                memory_delta_mb=mem_end - mem_start,
                status=status,
                error=error,
            )
            self.steps.append(step_metrics)

            logger.info(
                f"  Completed in {self._format_duration(step_metrics.duration_seconds)} "
                f"(memory: {step_metrics.memory_delta_mb:+.1f} MB)"
            )

            self.current_step = None

    def finish(self) -> dict:
        """Finish monitoring and return final metrics."""
        end_time = datetime.now()
        end_wall = time.perf_counter()
        end_memory = get_memory_info()['rss_mb']

        total_duration = end_wall - self.start_wall

        metrics = {
            'phase': self.phase_name,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_seconds': total_duration,
            'total_formatted': self._format_duration(total_duration),
            'memory_start_mb': self.start_memory,
            'memory_end_mb': end_memory,
            'memory_delta_mb': end_memory - self.start_memory,
            'n_steps': len(self.steps),
            'n_completed': sum(1 for s in self.steps if s.status == 'completed'),
            'n_failed': sum(1 for s in self.steps if s.status == 'failed'),
            'steps': [
                {
                    'name': s.name,
                    'duration_seconds': s.duration_seconds,
                    'duration_formatted': self._format_duration(s.duration_seconds),
                    'memory_delta_mb': s.memory_delta_mb,
                    'status': s.status,
                    'error': s.error,
                }
                for s in self.steps
            ],
            'system': {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'ram_gb': psutil.virtual_memory().total / (1024 ** 3),
                'gpu_available': _GPU_AVAILABLE,
                'gpu_backend': _GPU_BACKEND,
            },
        }

        logger.info("\n" + "=" * 70)
        logger.info(f"PHASE COMPLETE: {self.phase_name}")
        logger.info("=" * 70)
        logger.info(f"Total time: {metrics['total_formatted']}")
        logger.info(f"Memory delta: {metrics['memory_delta_mb']:+.1f} MB")
        logger.info(f"Steps: {metrics['n_completed']}/{metrics['n_steps']} completed")

        return metrics

    def format_summary(self) -> str:
        """Format a summary of all steps."""
        metrics = self.finish()
        lines = [
            "",
            "=" * 70,
            f"PHASE SUMMARY: {self.phase_name}",
            "=" * 70,
            "",
            f"Total Duration: {metrics['total_formatted']}",
            f"Memory Change:  {metrics['memory_delta_mb']:+.1f} MB",
            "",
            "STEPS",
            "-" * 70,
            f"{'#':<4} {'Step':<40} {'Time':>10} {'Memory':>10} {'Status':>8}",
            "-" * 70,
        ]

        for i, step in enumerate(self.steps, 1):
            lines.append(
                f"{i:<4} {step.name[:40]:<40} "
                f"{self._format_duration(step.duration_seconds):>10} "
                f"{step.memory_delta_mb:>+9.1f}MB "
                f"{step.status:>8}"
            )

        lines.append("-" * 70)
        lines.append("")

        return "\n".join(lines)

    def save_report(self, path: Path):
        """Save metrics to JSON file."""
        import json
        metrics = self.finish()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.info(f"Metrics saved to: {path}")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.0f}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
