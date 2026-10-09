"""Formato de cifras, serialización y nombres usados por el informe y la sintaxis SPSS."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from cancerstats import jsonutil
from cancerstats.export.fmt import ci, integer, num, pcell, pvalue, tex_escape
from cancerstats.export.latex import alias, join_es
from cancerstats.export.spss import _name


def test_numbers_use_siunitx() -> None:
    assert num(1.23456, 3) == r"\num{1.235}"
    assert num(-0.0001, 2) == r"\num{0.00}"
    assert num(float("nan")) == "---"
    assert num(2.5, 1, signed=True) == r"\num{+2.5}"
    assert integer(3047.2) == r"\num{3047}"


def test_pvalues_never_print_zero() -> None:
    assert pvalue(1e-12) == r"$p < \num{0.001}$"
    assert pvalue(0.0456) == r"$p = \num{0.046}$"
    assert pcell(0.0) == r"$<\num{0.001}$"
    assert pcell(None) == "---"


def test_interval_uses_semicolon() -> None:
    assert ci(1.0, 2.25, 1) == r"$[\num{1.0};\ \num{2.2}]$"
    assert num(7, 2) == r"\num{7}"  # los enteros (conteos) no llevan decimales


def test_tex_escape() -> None:
    assert tex_escape("50 % & $ _x") == r"50 \% \& \$ \_x"
    assert tex_escape("R² ≥ 0") == r"$R^2$ $\geq$ 0"


def test_aliases_are_tex_safe() -> None:
    assert alias("PctBachDeg25_Over") == "grado25"
    assert alias("region[Medio Oeste]") == "medio-oeste"
    assert alias("PctBachDeg25_Over:region[Noreste]") == "grado25xnoreste"
    assert join_es(["a", "b", "c"]) == "a, b y c"
    assert join_es(["a"]) == "a"


def test_spss_names() -> None:
    assert _name("PctBachDeg25_Over:region[Oeste]") == "PctBachDeg25_Over_x_reg_OE"


def test_jsonable_handles_numpy_and_nan(tmp_path) -> None:  # type: ignore[no-untyped-def]
    obj = {
        "a": np.float64(math.pi),
        "b": np.int64(3),
        "c": float("nan"),
        "d": np.array([1.0, np.inf]),
        "e": pd.Series({"x": 1.5}),
        "f": np.bool_(True),
    }
    out = jsonutil.to_jsonable(obj)
    assert out == {"a": 3.141593, "b": 3, "c": None, "d": [1.0, None], "e": {"x": 1.5}, "f": True}
    path = tmp_path / "x.json"
    jsonutil.dump(obj, path)
    assert json.loads(path.read_text(encoding="utf-8")) == out


def test_text_columns_are_ragged() -> None:
    from cancerstats.export.latex import _ragged

    rag = r">{\raggedright\arraybackslash}"
    assert _ragged(r"l p{2cm} r X") == rf"l {rag}p{{2cm}} r {rag}X"
    assert _ragged("l r") == "l r"
