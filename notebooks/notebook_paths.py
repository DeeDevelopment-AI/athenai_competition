from __future__ import annotations

from pathlib import Path


NOTEBOOKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOKS_DIR.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_ALGOS_DIR = RAW_DATA_DIR / "algoritmos"
RAW_BENCHMARK_DIR = RAW_DATA_DIR / "benchmark"

NOTEBOOK_DATA_DIR = NOTEBOOKS_DIR / "data"


def raw_path(*parts: str) -> Path:
    return RAW_DATA_DIR.joinpath(*parts)


def raw_algos_path(*parts: str) -> Path:
    return RAW_ALGOS_DIR.joinpath(*parts)


def raw_benchmark_path(*parts: str) -> Path:
    return RAW_BENCHMARK_DIR.joinpath(*parts)


def notebook_data_path(*parts: str) -> Path:
    return NOTEBOOK_DATA_DIR.joinpath(*parts)


def ensure_notebook_data_dir(*parts: str) -> Path:
    path = notebook_data_path(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_dir(*parts: str) -> Path:
    path = notebook_data_path(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
