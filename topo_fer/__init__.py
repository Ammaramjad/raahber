"""TOPO-FER package initialization."""

from importlib.metadata import version

try:  # pragma: no cover
    __version__ = version("topo-fer")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
