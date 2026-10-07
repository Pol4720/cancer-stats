"""Ingesta del fichero original.

El CSV de Kaggle llega con dos particularidades que una lectura ingenua no detecta:

1. Finales de línea CR (``\\r``), propios del Mac OS clásico, en lugar de LF o CRLF.
2. Codificación Mac Roman: el byte ``0x96`` es la «ñ» de *Doña Ana County* (Nuevo México).
   Leído como UTF-8 falla; leído como Windows-1252 se convierte en un guion largo.

Además, la columna ``Geography`` («Condado, Estado») se movió al final y su coma interna
la partió en dos columnas (``Geography`` y ``state``), dejando una columna vacía sin nombre
en su posición original.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS: tuple[str, ...] = (
    "avgAnnCount",
    "avgDeathsPerYear",
    "TARGET_deathRate",
    "incidenceRate",
    "medIncome",
    "popEst2015",
    "povertyPercent",
    "studyPerCap",
    "binnedInc",
    "MedianAge",
    "MedianAgeMale",
    "MedianAgeFemale",
    "AvgHouseholdSize",
    "PercentMarried",
    "PctNoHS18_24",
    "PctHS18_24",
    "PctSomeCol18_24",
    "PctBachDeg18_24",
    "PctHS25_Over",
    "PctBachDeg25_Over",
    "PctEmployed16_Over",
    "PctUnemployed16_Over",
    "PctPrivateCoverage",
    "PctPrivateCoverageAlone",
    "PctEmpPrivCoverage",
    "PctPublicCoverage",
    "PctPublicCoverageAlone",
    "PctWhite",
    "PctBlack",
    "PctAsian",
    "PctOtherRace",
    "PctMarriedHouseholds",
    "BirthRate",
    "Geography",
    "state",
)

TEXT_COLUMNS: tuple[str, ...] = ("binnedInc", "Geography", "state")


@dataclass
class IngestReport:
    """Qué se encontró al leer el fichero y qué se hizo."""

    path: str
    sha256: str
    n_bytes: int
    encoding: str
    line_ending: str
    n_rows: int
    n_columns_raw: int
    empty_columns: list[str] = field(default_factory=list)
    non_ascii_examples: list[str] = field(default_factory=list)
    missing_expected: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Representación serializable."""
        return dict(self.__dict__)


def sha256_bytes(data: bytes) -> str:
    """Huella SHA-256 en hexadecimal."""
    return hashlib.sha256(data).hexdigest()


def detect_line_ending(data: bytes) -> str:
    """Devuelve ``"CRLF"``, ``"LF"`` o ``"CR"`` según el separador de líneas dominante."""
    crlf = data.count(b"\r\n")
    lf = data.count(b"\n") - crlf
    cr = data.count(b"\r") - crlf
    counts = {"CRLF": crlf, "LF": lf, "CR": cr}
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] > 0 else "LF"


def detect_encoding(data: bytes) -> str:
    """Detecta la codificación entre UTF-8, Mac Roman y Windows-1252.

    Si el contenido es UTF-8 válido, se usa UTF-8. Si no lo es y las líneas terminan en CR
    (huella del Mac clásico), se asume Mac Roman; en otro caso, Windows-1252.
    """
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return "mac_roman" if detect_line_ending(data) == "CR" else "cp1252"
    return "utf-8"


def decode(data: bytes, encoding: str = "auto") -> tuple[str, str]:
    """Decodifica los bytes y normaliza los finales de línea a LF.

    Returns:
        El texto normalizado y la codificación efectivamente usada.
    """
    enc = detect_encoding(data) if encoding == "auto" else encoding
    text = data.decode(enc)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text, enc


def read_raw(path: Path, encoding: str = "auto") -> tuple[pd.DataFrame, IngestReport]:
    """Lee el CSV original y devuelve los datos junto con el informe de ingesta.

    Las columnas completamente vacías y sin nombre (artefacto de exportación) se eliminan
    aquí, porque no contienen información; quedan registradas en el informe.
    """
    data = path.read_bytes()
    text, enc = decode(data, encoding)
    df = pd.read_csv(io.StringIO(text), dtype=dict.fromkeys(TEXT_COLUMNS, "string"))

    empty = [c for c in df.columns if c.startswith("Unnamed") and df[c].isna().all()]
    df = df.drop(columns=empty)
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].str.strip()

    non_ascii = sorted({line for line in text.split("\n") if any(ord(ch) > 127 for ch in line)})
    examples = [
        line.rsplit(",", 2)[-2].strip() + ", " + line.rsplit(",", 1)[-1].strip()
        for line in non_ascii
    ][:5]

    report = IngestReport(
        path=str(path),
        sha256=sha256_bytes(data),
        n_bytes=len(data),
        encoding=enc,
        line_ending=detect_line_ending(data),
        n_rows=len(df),
        n_columns_raw=len(df.columns) + len(empty),
        empty_columns=empty,
        non_ascii_examples=examples,
        missing_expected=[c for c in EXPECTED_COLUMNS if c not in df.columns],
        unexpected_columns=[c for c in df.columns if c not in EXPECTED_COLUMNS],
    )
    if report.missing_expected:
        raise ValueError(f"Faltan columnas esperadas: {report.missing_expected}")
    df = df[list(EXPECTED_COLUMNS)]
    df.insert(0, "county_id", df["Geography"] + ", " + df["state"])
    return df, report
