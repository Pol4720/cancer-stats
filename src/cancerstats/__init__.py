"""cancerstats: análisis estadístico reproducible de la mortalidad por cáncer en EE. UU.

El paquete implementa, etapa por etapa, el proceso de inferencia estadística del curso
(planteamiento, depuración, estimación, contrastes de simplificación, diagnóstico y
previsión) sobre el conjunto de datos de Kaggle *Cancer Death Rates*.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cancerstats")
except PackageNotFoundError:  # pragma: no cover - sólo ocurre sin instalar el paquete
    __version__ = "0.0.0"

__all__ = ["__version__"]
