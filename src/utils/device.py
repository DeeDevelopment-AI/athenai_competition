"""
GPU/CUDA device detection and management utilities.

Provides automatic device selection for PyTorch and stable-baselines3,
with graceful fallback to CPU when CUDA is not available.

Usage:
    from src.utils.device import get_device, get_torch_device, DeviceManager

    # For SB3 agents
    device = get_device()  # Returns "cuda" or "cpu"
    agent = PPO(..., device=device)

    # For PyTorch tensors
    torch_device = get_torch_device()  # Returns torch.device
    tensor = tensor.to(torch_device)

    # Check capabilities
    info = DeviceManager.get_device_info()
    print(info)
"""

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Try to import torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


@dataclass
class DeviceInfo:
    """Information about available compute devices."""

    has_cuda: bool
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str
    cuda_memory_total_gb: float
    cuda_memory_free_gb: float
    recommended_device: str
    torch_version: str
    cuda_version: str
    system_cuda_detected: bool = False
    system_cuda_source: str = ""
    diagnostic_message: str = ""

    def __str__(self) -> str:
        lines = [
            "Device Information:",
            f"  PyTorch version: {self.torch_version}",
            f"  CUDA available: {self.cuda_available}",
        ]
        if self.cuda_available:
            lines.extend([
                f"  CUDA version: {self.cuda_version}",
                f"  Device count: {self.cuda_device_count}",
                f"  Device name: {self.cuda_device_name}",
                f"  Memory total: {self.cuda_memory_total_gb:.2f} GB",
                f"  Memory free: {self.cuda_memory_free_gb:.2f} GB",
            ])
        elif self.system_cuda_detected:
            lines.append(f"  System CUDA detected via: {self.system_cuda_source}")
        lines.append(f"  Recommended device: {self.recommended_device}")
        if self.diagnostic_message:
            lines.append(f"  Note: {self.diagnostic_message}")
        return "\n".join(lines)


class DeviceManager:
    """
    Manages device selection and provides utilities for GPU/CPU operations.

    Thread-safe singleton pattern for consistent device usage across the application.
    """

    _instance = None
    _device: Optional[str] = None
    _torch_device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize device detection."""
        self._device = self._detect_best_device()
        if HAS_TORCH:
            self._torch_device = torch.device(self._device)
        logger.info(f"Device initialized: {self._device}")

    def _detect_best_device(self) -> str:
        """Detect the best available device."""
        # Check for explicit device override
        env_device = os.environ.get("RL_DEVICE", "").lower()
        if env_device in ("cpu", "cuda"):
            return env_device

        if not HAS_TORCH:
            return "cpu"

        # Check CUDA availability
        if torch.cuda.is_available():
            try:
                # Test CUDA actually works
                test_tensor = torch.zeros(1, device="cuda")
                del test_tensor
                return "cuda"
            except Exception as e:
                logger.warning(f"CUDA available but failed to use: {e}")
                return "cpu"

        return "cpu"

    @staticmethod
    def detect_system_cuda() -> tuple[bool, str]:
        """
        Detect whether the host system exposes NVIDIA/CUDA tools.

        This does not guarantee the current torch build can use CUDA.
        """
        if shutil.which("nvidia-smi"):
            return True, "nvidia-smi"
        if shutil.which("nvcc"):
            return True, "nvcc"
        return False, ""

    @property
    def device(self) -> str:
        """Get the selected device string ('cuda' or 'cpu')."""
        return self._device

    @property
    def torch_device(self):
        """Get the torch.device object."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not installed")
        return self._torch_device

    @classmethod
    def get_device_info(cls) -> DeviceInfo:
        """Get detailed information about available devices."""
        if not HAS_TORCH:
            return DeviceInfo(
                has_cuda=False,
                cuda_available=False,
                cuda_device_count=0,
                cuda_device_name="N/A",
                cuda_memory_total_gb=0,
                cuda_memory_free_gb=0,
                recommended_device="cpu",
                torch_version="N/A",
                cuda_version="N/A",
            )

        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
        cuda_device_name = ""
        cuda_memory_total_gb = 0.0
        cuda_memory_free_gb = 0.0
        cuda_version = ""
        system_cuda_detected, system_cuda_source = cls.detect_system_cuda()
        diagnostic_message = ""

        if cuda_available and cuda_device_count > 0:
            cuda_device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda or "N/A"
            try:
                props = torch.cuda.get_device_properties(0)
                cuda_memory_total_gb = props.total_memory / (1024**3)
                # Get free memory
                cuda_memory_free_gb = (
                    props.total_memory - torch.cuda.memory_allocated(0)
                ) / (1024**3)
            except Exception:
                pass

        manager = cls()

        if system_cuda_detected and not cuda_available:
            diagnostic_message = (
                "NVIDIA/CUDA is present on the system, but the current PyTorch build "
                "cannot use CUDA. This usually means torch was installed as CPU-only."
            )

        return DeviceInfo(
            has_cuda=True,
            cuda_available=cuda_available,
            cuda_device_count=cuda_device_count,
            cuda_device_name=cuda_device_name,
            cuda_memory_total_gb=cuda_memory_total_gb,
            cuda_memory_free_gb=cuda_memory_free_gb,
            recommended_device=manager.device,
            torch_version=torch.__version__,
            cuda_version=cuda_version,
            system_cuda_detected=system_cuda_detected,
            system_cuda_source=system_cuda_source,
            diagnostic_message=diagnostic_message,
        )

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None
        cls._device = None
        cls._torch_device = None


def get_device() -> str:
    """
    Get the recommended device string for use with stable-baselines3.

    Returns:
        'cuda' if GPU available and working, 'cpu' otherwise.

    Example:
        from stable_baselines3 import PPO
        from src.utils.device import get_device

        agent = PPO('MlpPolicy', env, device=get_device())
    """
    return DeviceManager().device


def get_torch_device():
    """
    Get the recommended torch.device object for PyTorch operations.

    Returns:
        torch.device('cuda') if GPU available, torch.device('cpu') otherwise.

    Example:
        from src.utils.device import get_torch_device

        device = get_torch_device()
        tensor = torch.zeros(100, device=device)
    """
    return DeviceManager().torch_device


def to_device(tensor_or_array, device: Optional[str] = None):
    """
    Move a tensor or array to the specified device.

    Args:
        tensor_or_array: PyTorch tensor or numpy array
        device: Target device ('cuda', 'cpu', or None for auto-select)

    Returns:
        Tensor on the target device
    """
    if not HAS_TORCH:
        return tensor_or_array

    if device is None:
        device = get_device()

    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.to(device)
    else:
        # Assume numpy array
        import numpy as np
        if isinstance(tensor_or_array, np.ndarray):
            return torch.from_numpy(tensor_or_array).to(device)
        else:
            return tensor_or_array


def ensure_cpu(tensor_or_array):
    """
    Ensure a tensor is on CPU (useful for numpy operations).

    Args:
        tensor_or_array: PyTorch tensor or numpy array

    Returns:
        Numpy array on CPU
    """
    if not HAS_TORCH:
        return tensor_or_array

    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().numpy()
    else:
        return tensor_or_array


def set_cuda_memory_fraction(fraction: float = 0.8) -> None:
    """
    Set the fraction of GPU memory to use.

    Useful for sharing GPU with other processes or avoiding OOM.

    Args:
        fraction: Fraction of total GPU memory to use (0.0 to 1.0)
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return

    torch.cuda.set_per_process_memory_fraction(fraction)
    logger.info(f"CUDA memory fraction set to {fraction:.1%}")


def clear_cuda_cache() -> None:
    """
    Clear CUDA cache to free up memory.

    Call this after training or when switching between models.
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return

    torch.cuda.empty_cache()
    logger.debug("CUDA cache cleared")


def get_memory_stats() -> dict:
    """
    Get current CUDA memory usage statistics.

    Returns:
        Dict with memory stats, or empty dict if CUDA not available
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return {}

    return {
        "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "cached_gb": torch.cuda.memory_reserved() / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
    }


# Convenience function for SB3 policy kwargs
def get_sb3_policy_kwargs(net_arch: Optional[list] = None) -> dict:
    """
    Get policy kwargs optimized for the current device.

    Args:
        net_arch: Network architecture, e.g., [256, 256]

    Returns:
        Dict of policy kwargs for SB3 agents
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for SB3 policy")

    if net_arch is None:
        net_arch = [256, 256]

    return {
        "net_arch": dict(pi=net_arch, vf=net_arch),
        "activation_fn": torch.nn.ReLU,
        "ortho_init": True,
    }
