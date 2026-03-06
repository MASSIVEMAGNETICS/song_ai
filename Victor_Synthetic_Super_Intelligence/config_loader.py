"""Config Loader — YAML-based configuration management for Victor SSI.

Loads the default ``model_config.yaml`` and merges user-supplied overrides
so that every component receives a consistent, validated configuration dict.

Example::

    from Victor_Synthetic_Super_Intelligence.config_loader import load_config

    # Load defaults from configs/model_config.yaml
    cfg = load_config()

    # Merge runtime overrides
    cfg = load_config(overrides={"cognition": {"max_iterations": 10}})

    # Load from a custom path
    cfg = load_config(path="/etc/victor/config.yaml")
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Path to the bundled default configuration file.
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "configs", "model_config.yaml"
)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructively).

    Values in *override* take precedence.  Nested dicts are merged
    recursively rather than replaced wholesale.

    Args:
        base: The base dictionary (not modified).
        override: Overrides to apply on top of *base*.

    Returns:
        A new merged dictionary.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML file using the standard library (no PyYAML required)
    when the ``yaml`` package is unavailable.

    Falls back to a minimal hand-rolled parser for the simple key/value
    structure used by ``model_config.yaml``.  If PyYAML *is* installed it
    is preferred for full YAML support.

    Args:
        path: Absolute or relative path to the YAML file.

    Returns:
        Parsed configuration dict.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        import yaml  # type: ignore[import]

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}
    except ImportError:
        logger.debug(
            "PyYAML not installed; using built-in YAML parser for '%s'", path
        )
        return _parse_simple_yaml(path)


def _parse_simple_yaml(path: str) -> dict[str, Any]:
    """Minimal YAML parser sufficient for the default config file.

    Handles:
    * Top-level and one-level-nested keys
    * String, integer, float, and ``null`` values
    * Comments (lines starting with ``#``)

    This is *not* a general-purpose YAML parser — it is intentionally
    kept simple to avoid a PyYAML dependency.

    Args:
        path: Path to the YAML file.

    Returns:
        Nested dictionary parsed from the file.
    """
    result: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    current_key: str | None = None

    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            # Strip inline comments
            line = raw_line.split("#")[0].rstrip()
            if not line.strip():
                continue

            indent = len(line) - len(line.lstrip())

            if indent == 0:
                # Top-level key (section header or bare key/value)
                if ":" in line:
                    key, _, raw_val = line.partition(":")
                    key = key.strip()
                    raw_val = raw_val.strip()
                    if raw_val:
                        result[key] = _cast_yaml_value(raw_val)
                        current_section = None
                        current_key = None
                    else:
                        # Section header
                        result[key] = {}
                        current_section = result[key]
                        current_key = key
            elif indent == 2 and current_section is not None:
                # First-level nested key
                if ":" in line:
                    key, _, raw_val = line.partition(":")
                    key = key.strip()
                    raw_val = raw_val.strip()
                    if raw_val:
                        current_section[key] = _cast_yaml_value(raw_val)
                    else:
                        current_section[key] = {}
                        # Track sub-section for deeper nesting if needed
            elif indent == 4 and current_section is not None and current_key is not None:
                # Second-level nested key
                if ":" in line:
                    key, _, raw_val = line.partition(":")
                    key = key.strip()
                    raw_val = raw_val.strip()
                    # Find the correct sub-section
                    for sub_key, sub_val in current_section.items():
                        if isinstance(sub_val, dict):
                            # Place under the last empty sub-section
                            if len(sub_val) == 0 or key in sub_val:
                                sub_val[key] = _cast_yaml_value(raw_val) if raw_val else {}
                                break

    return result


def _cast_yaml_value(raw: str) -> Any:
    """Convert a raw YAML string value to a Python type.

    Handles ``null``, booleans, integers, floats, and quoted/unquoted
    strings.

    Args:
        raw: The raw string value from the YAML file (after the ``:``)

    Returns:
        The appropriately typed Python value.
    """
    stripped = raw.strip().strip('"').strip("'")

    if stripped.lower() == "null":
        return None
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False

    # Try integer
    try:
        return int(stripped)
    except ValueError:
        pass

    # Try float (handles scientific notation like 1.0e-4)
    try:
        return float(stripped)
    except ValueError:
        pass

    return stripped


def load_config(
    path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and return a fully-merged configuration dictionary.

    The function performs a three-phase merge:

    1. Load defaults from ``configs/model_config.yaml`` (or *path*).
    2. Apply any environment-variable overrides (prefixed ``VICTOR_``).
    3. Apply caller-supplied *overrides*.

    Environment variables
    ---------------------
    Any ``VICTOR_<SECTION>_<KEY>`` environment variable overrides the
    corresponding config value.  For example::

        VICTOR_COGNITION_MAX_ITERATIONS=10
        VICTOR_INTERFACES_API_PORT=9000

    Args:
        path: Optional path to a custom YAML config file.  Defaults to
            the bundled ``configs/model_config.yaml``.
        overrides: Optional dict of overrides to apply after the file is
            loaded.  Nested dicts are deep-merged.

    Returns:
        Merged configuration dictionary.
    """
    config_path = path or _DEFAULT_CONFIG_PATH
    try:
        config = _load_yaml(config_path)
    except FileNotFoundError:
        logger.warning(
            "Config file '%s' not found; using empty configuration.", config_path
        )
        config = {}

    # Apply environment-variable overrides (VICTOR_SECTION_KEY=value)
    config = _apply_env_overrides(config)

    # Apply caller-supplied overrides last
    if overrides:
        config = _deep_merge(config, overrides)

    logger.debug("Configuration loaded: %s", config)
    return config


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Scan environment variables for VICTOR_ prefixed overrides.

    Variables are expected to follow the pattern
    ``VICTOR_<SECTION>_<KEY>=<value>`` where *SECTION* and *KEY* are
    case-insensitive matches for top-level and second-level config keys.

    Args:
        config: The base configuration dictionary.

    Returns:
        Configuration with any environment overrides applied.
    """
    result = copy.deepcopy(config)
    prefix = "VICTOR_"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = env_key[len(prefix):].lower().split("_", 1)
        if len(parts) == 2:
            section, key = parts
            if section in result and isinstance(result[section], dict):
                result[section][key] = _cast_yaml_value(env_val)
                logger.debug(
                    "Env override: %s.%s = %r", section, key, env_val
                )
    return result


__all__ = ["load_config"]
