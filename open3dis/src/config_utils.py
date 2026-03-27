from copy import deepcopy
from typing import Any, Dict, Iterable, Optional

import yaml
from munch import Munch


def _load_yaml_mapping(path: str) -> Dict[str, Any]:
    with open(path, "r") as handle:
        data = yaml.safe_load(handle.read())

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    return data


def deep_merge_dicts(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml_config(config_path: str, overlays: Optional[Iterable[str]] = None) -> Munch:
    merged = _load_yaml_mapping(config_path)
    config_paths = [config_path]

    for overlay_path in overlays or []:
        merged = deep_merge_dicts(merged, _load_yaml_mapping(overlay_path))
        config_paths.append(overlay_path)

    cfg = Munch.fromDict(merged)
    cfg._config_paths = config_paths
    return cfg
