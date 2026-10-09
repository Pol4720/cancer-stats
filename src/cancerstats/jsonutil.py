"""Serialización JSON compacta y determinista de los resultados."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SIGNIFICANT = 7


def _round(x: float) -> float | None:
    if math.isnan(x) or math.isinf(x):
        return None
    if x == 0:
        return 0.0
    digits = SIGNIFICANT - math.floor(math.log10(abs(x))) - 1
    return round(x, max(digits, 0)) if digits < 15 else x


def to_jsonable(obj: Any) -> Any:
    """Convierte recursivamente tipos de NumPy y pandas a tipos nativos serializables.

    Los reales se redondean a 7 cifras significativas (precisión de sobra para informar y
    archivos más pequeños) y los NaN/infinitos pasan a ``null``.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, bool | np.bool_):
        return bool(obj)
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        return _round(float(obj))
    if obj is None or isinstance(obj, str):
        return obj
    if hasattr(obj, "to_dict"):
        return to_jsonable(obj.to_dict())
    return str(obj)


def dump(obj: Any, path: Path, indent: int | None = None) -> None:
    """Escribe JSON (UTF-8, sin escapar caracteres no ASCII)."""
    path.write_text(
        json.dumps(
            to_jsonable(obj),
            ensure_ascii=False,
            indent=indent,
            separators=(",", ":") if indent is None else None,
        ),
        encoding="utf-8",
    )


def load(path: Path) -> Any:
    """Lee JSON."""
    return json.loads(path.read_text(encoding="utf-8"))
