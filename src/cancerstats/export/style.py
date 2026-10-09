"""Estilo gráfico común de las figuras del informe.

La paleta categórica, las tintas y las superficies son las de la paleta de referencia
validada para daltonismo (orden fijo de los tonos, nunca cíclico). Los números de los ejes
usan coma decimal, como el texto del informe.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

REGION_COLORS = {
    "Sur": SERIES[0],
    "Medio Oeste": SERIES[1],
    "Oeste": SERIES[2],
    "Noreste": SERIES[3],
}
REGION_ORDER = ["Sur", "Medio Oeste", "Oeste", "Noreste"]

WIDTH = 6.3  # pulgadas: ancho de texto del informe (16 cm)


FONT_DIR = Path(__file__).with_name("fonts")


def _font_family() -> list[str]:
    """Registra la Inter incluida en el paquete (TTF) para que las figuras sean idénticas en
    cualquier máquina; se descartan versiones OTF del sistema con el mismo nombre, que
    matplotlib no puede incrustar como TrueType."""
    fm = font_manager.fontManager
    fm.ttflist = [f for f in fm.ttflist if not (f.name == "Inter" and not f.fname.endswith(".ttf"))]
    for ttf in sorted(FONT_DIR.glob("Inter-*.ttf")):
        if not any(f.fname == str(ttf) for f in fm.ttflist):
            fm.addfont(str(ttf))
    return ["Inter", "DejaVu Sans"]


def setup() -> None:
    """Aplica el estilo a matplotlib."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": _font_family(),
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "text.color": INK,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
            "axes.unicode_minus": True,
            "mathtext.fontset": "dejavusans",
        }
    )


def _fmt(x: float, _pos: int | None = None) -> str:
    text = f"{x:,.0f}".replace(",", "\u2009") if abs(x) >= 10000 else f"{x:.6g}"
    return text.replace("-", "−").replace(".", ",")


def comma(ax: plt.Axes, which: str = "both") -> None:
    """Coma decimal en los ejes indicados."""
    fmt = ticker.FuncFormatter(_fmt)
    if which in ("both", "x"):
        ax.xaxis.set_major_formatter(fmt)
    if which in ("both", "y"):
        ax.yaxis.set_major_formatter(fmt)


def save(fig: plt.Figure, path: Path) -> None:
    """Guarda en PDF determinista (sin fecha de creación) y cierra la figura."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path, metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}
    )
    plt.close(fig)


def es(x: float, digits: int = 2) -> str:
    """Número con coma decimal para anotaciones."""
    return f"{x:.{digits}f}".replace("-", "−").replace(".", ",")
