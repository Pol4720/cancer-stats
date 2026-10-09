#!/usr/bin/env python3
"""Comprueba que la compilación del informe no deja advertencias ni cajas mal llenas.

Uso: python scripts/check_latex_log.py report/build/informe.log

Sale con código 1 y lista los problemas si el registro contiene advertencias de LaTeX o de
paquetes (incluidas las cifras que faltan en ``generado/resultados.tex``), cajas demasiado
llenas o demasiado vacías, caracteres que la fuente no tiene o referencias sin resolver.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERNS = [
    re.compile(r"Warning", re.IGNORECASE),
    re.compile(r"^(Overfull|Underfull) \\[hv]box"),
    re.compile(r"Missing character"),
    re.compile(r"undefined", re.IGNORECASE),
]
# Líneas informativas que contienen las palabras anteriores sin ser un problema.
IGNORE = [
    re.compile(r"^Package: infwarerr"),
    re.compile(r"^\(\S*infwarerr"),
]


def problems(log: str) -> list[str]:
    """Líneas problemáticas del registro (sin duplicados, en orden)."""
    # TeX parte las líneas a 79 caracteres: se reconstruyen antes de buscar.
    lines = log.replace("\r", "").split("\n")
    found: list[str] = []
    for i, line in enumerate(lines):
        if any(p.search(line) for p in IGNORE):
            continue
        if any(p.search(line) for p in PATTERNS):
            context = (
                line if len(line) < 79 else line + (lines[i + 1] if i + 1 < len(lines) else "")
            )
            if context not in found:
                found.append(context)
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    path = Path(argv[1])
    log = path.read_text(encoding="utf-8", errors="replace")
    found = problems(log)
    if found:
        print(f"{path}: {len(found)} problema(s) en la compilación:")
        for f in found:
            print(f"  · {f}")
        return 1
    print(f"{path}: compilación limpia (sin advertencias ni cajas mal llenas).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
