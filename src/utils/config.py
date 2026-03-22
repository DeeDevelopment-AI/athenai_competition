"""
Configuración centralizada del proyecto.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

_config: Optional[dict] = None


def load_config(config_path: str | Path = "configs/default.yaml") -> dict:
    """
    Carga configuración desde archivo YAML.

    Args:
        config_path: Ruta al archivo de configuración.

    Returns:
        Diccionario con la configuración.
    """
    global _config

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return _config


def get_config(key: Optional[str] = None, default: Any = None) -> Any:
    """
    Obtiene valor de configuración.

    Args:
        key: Clave en formato "section.subsection.param" o None para todo.
        default: Valor por defecto si no existe.

    Returns:
        Valor de configuración o default.
    """
    global _config

    if _config is None:
        load_config()

    if key is None:
        return _config

    keys = key.split(".")
    value = _config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    return value


def merge_configs(*configs: dict) -> dict:
    """
    Combina múltiples configuraciones (las posteriores sobrescriben).

    Args:
        configs: Diccionarios de configuración.

    Returns:
        Configuración combinada.
    """
    result = {}
    for config in configs:
        _deep_merge(result, config)
    return result


def _deep_merge(base: dict, override: dict) -> None:
    """Merge recursivo de diccionarios."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
