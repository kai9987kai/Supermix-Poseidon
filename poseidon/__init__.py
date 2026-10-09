"""Supermix Poseidon: explicitly bounded, locally trained research system."""
__version__ = "0.5.0"

from .provenance import capture_sources as _capture_sources

_SOURCE_SNAPSHOT = _capture_sources()
