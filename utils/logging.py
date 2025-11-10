# utils/logging.py
from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Dict


def _flatten_mapping(mapping: Mapping[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in mapping.items():
        if prefix:
            new_key = f"{prefix}{sep}{key}"
        else:
            new_key = str(key)

        if isinstance(value, Mapping):
            flat.update(_flatten_mapping(value, new_key, sep=sep))
        else:
            flat[new_key] = value
    return flat


def log_config(logger: Any, config: Mapping[str, Any], extras: Dict[str, Any] | None = None) -> None:
    if logger is None:
        return

    params = _flatten_mapping(config)
    if extras:
        params.update(extras)

    try:
        logger.log_hyperparams(params)
        return
    except Exception:
        pass

    experiment = getattr(logger, "experiment", None)
    config_attr = getattr(experiment, "config", None)
    if config_attr is not None:
        try:
            config_attr.update(params, allow_val_change=True)  # type: ignore[arg-type]
        except TypeError:
            config_attr.update(params)  # type: ignore[call-arg]


def log_config_file(logger: Any, config_path: str, artifact_name: str | None = None, artifact_type: str = "config") -> None:
    """Attach a config file to the underlying experiment when supported."""
    if logger is None or not config_path:
        return

    if not os.path.isfile(config_path):
        return

    experiment = getattr(logger, "experiment", None)
    if experiment is None:
        return

    try:
        import wandb
    except ImportError:
        return

    name_root = artifact_name or Path(config_path).stem
    artifact = wandb.Artifact(name=name_root, type=artifact_type)
    artifact.add_file(str(config_path))
    try:
        experiment.log_artifact(artifact)
    except Exception:
        pass
