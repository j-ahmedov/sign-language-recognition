"""YAML config loading with dotted access and command-line overrides.

Configs are plain dictionaries on disk (``configs/*.yaml``) and are snapshotted verbatim
into every results file, so a JSON result is always self-describing.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_config", "merge_configs", "get_in", "set_in", "apply_overrides"]


def load_config(*paths: str | Path) -> dict[str, Any]:
    """Load one or more YAML files, merging them left to right.

    Args:
        *paths: Paths to YAML files. Later files override earlier ones key by key.

    Returns:
        The merged configuration dictionary.

    Raises:
        FileNotFoundError: If any path does not exist.
    """
    merged: dict[str, Any] = {}
    for path in paths:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"config not found: {p}")
        with p.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        merged = merge_configs(merged, loaded)
    return merged


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either.

    Args:
        base: The base configuration.
        override: Values that take precedence.

    Returns:
        A new merged dictionary.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get_in(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read a nested value by dotted key, e.g. ``"encoder.d_model"``.

    Args:
        config: Configuration dictionary.
        dotted_key: Dot-separated path.
        default: Returned when the path is absent.

    Returns:
        The value at ``dotted_key`` or ``default``.
    """
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_in(config: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    """Return a copy of ``config`` with ``dotted_key`` set to ``value``.

    Args:
        config: Configuration dictionary.
        dotted_key: Dot-separated path; intermediate dicts are created as needed.
        value: Value to write.

    Returns:
        A new dictionary with the value set.
    """
    result = copy.deepcopy(config)
    node = result
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value
    return result


def _parse_scalar(raw: str) -> Any:
    """Parse an override value, repairing YAML 1.1's exponent rule.

    ``yaml.safe_load("1e-3")`` returns the *string* ``"1e-3"`` because YAML 1.1 only accepts
    a float exponent when it has a decimal point (``1.0e-3``). Silently passing a string
    learning rate into an optimizer is exactly the kind of bug that costs a day, so numeric
    -looking strings are promoted to float here.

    Args:
        raw: The right-hand side of a ``key=value`` override.

    Returns:
        The parsed value.
    """
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``key=value`` command-line overrides, parsing values as YAML scalars.

    Example:
        ``apply_overrides(cfg, ["train.lr=1e-3", "augment.enabled=false"])``

    Args:
        config: Configuration dictionary.
        overrides: Strings of the form ``dotted.key=value``.

    Returns:
        A new dictionary with all overrides applied.

    Raises:
        ValueError: If an override is not of the form ``key=value``.
    """
    result = config
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        result = set_in(result, key.strip(), _parse_scalar(raw))
    return result
