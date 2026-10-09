"""Formato de números y texto para LaTeX en castellano.

Los números se emiten dentro de ``\\num{...}`` de siunitx, configurado en el preámbulo con
coma decimal y espacio fino como separador de miles, de modo que el formato tipográfico se
decide en un único sitio.
"""

from __future__ import annotations

import math

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex_escape(text: object) -> str:
    """Escapa los caracteres especiales de LaTeX en un texto plano."""
    out = []
    for ch in str(text):
        out.append(_LATEX_SPECIAL.get(ch, ch))
    s = "".join(out)
    return (
        s.replace("≥", r"$\geq$")
        .replace("≤", r"$\leq$")
        .replace("×", r"$\times$")
        .replace("−", "\\textminus{}")
        .replace("σ²", r"$\sigma^2$")
        .replace("χ²", r"$\chi^2$")
        .replace("R²", r"$R^2$")
        .replace("λ", r"$\lambda$")
        .replace("→", r"$\to$")
        .replace("⇒", r"$\Rightarrow$")
        .replace("·", r"$\cdot$")
        .replace("μ", r"$\mu$")
        .replace("Σ", r"$\Sigma$")
        .replace("⊆", r"$\subseteq$")
        .replace("≈", r"$\approx$")
        .replace("′", "'")
    )


def num(x: float | int | None, digits: int = 2, signed: bool = False) -> str:
    """Número con siunitx; ``None``/NaN se representan con una raya."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "---"
    if isinstance(x, int) or (isinstance(x, float) and digits == 0):
        value = f"{round(float(x)):d}" if digits == 0 else str(x)
        return rf"\num{{{'+' if signed and float(x) > 0 else ''}{value}}}"
    text = f"{float(x):.{digits}f}"
    if text in (f"-{0:.{digits}f}",):
        text = f"{0:.{digits}f}"
    return rf"\num{{{'+' if signed and float(x) > 0 else ''}{text}}}"


def integer(x: float | int) -> str:
    """Entero con separador de miles."""
    return rf"\num{{{round(float(x)):d}}}"


def pct(x: float, digits: int = 1) -> str:
    """Porcentaje (el valor ya está en %)."""
    return rf"\SI{{{float(x):.{digits}f}}}{{\percent}}"


def pvalue(p: float | None, digits: int = 3, with_p: bool = True) -> str:
    """p-valor: «p < 0,001» si es muy pequeño; nunca «p = 0,000»."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "---"
    threshold = 10 ** (-digits)
    prefix = "p" if with_p else ""
    if p < threshold:
        return rf"${prefix} < \num{{{threshold:.{digits}f}}}$"
    return rf"${prefix} {'= ' if with_p else ''}\num{{{p:.{digits}f}}}$"


def pcell(p: float | None, digits: int = 3) -> str:
    """p-valor para una celda de tabla (sin «p =»)."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "---"
    threshold = 10 ** (-digits)
    if p < threshold:
        return rf"$<\num{{{threshold:.{digits}f}}}$"
    return rf"\num{{{p:.{digits}f}}}"


def ci(lo: float, hi: float, digits: int = 2) -> str:
    """Intervalo de confianza [a; b] con punto y coma (la coma es el separador decimal)."""
    return rf"$[{num(lo, digits)};\ {num(hi, digits)}]$"


def stars(p: float | None) -> str:
    """Marcas de significación convencionales."""
    if p is None:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
