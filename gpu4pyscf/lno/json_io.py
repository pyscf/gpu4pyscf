"""JSON serialization for scientific results containing NumPy scalar types.

NumPy scalars can enter diagnostics from PySCF / CuPy array dimensions and
from reductions. Preserve JSON numeric types instead of converting to strings.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _json_default(value):
    if isinstance(value, np.generic):
        # np.int64 -> int, np.float64 -> float, np.bool_ -> bool.
        # Unsupported types such as complex remain unsupported.
        result = value.item()
        if isinstance(result, (str, int, float, bool)) or result is None:
            return result
        raise TypeError(f"Unsupported NumPy scalar for JSON: {type(value).__name__}")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dumps_results(data, *, indent=2):
    """Serialize NumPy-aware diagnostics; do not silently stringify numbers."""
    return json.dumps(data, indent=indent, default=_json_default, allow_nan=False) + "\n"


def write_results(path, data, *, indent=2):
    path = Path(path)
    payload = dumps_results(data, indent=indent)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf8")
    return path
