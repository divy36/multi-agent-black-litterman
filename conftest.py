"""Conftest."""

from __future__ import annotations

import urllib.request

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: slow test that calls a real, locally running Ollama "
        "instance and reads real project data files. Run explicitly with "
        "'pytest -m integration'; excluded from 'pytest -m \"not integration\"'.",
    )


def ollama_is_reachable(url: str = "http://localhost:11434/api/tags", timeout: float = 2.0) -> bool:
    """True if a local Ollama server responds. Used to skip integration
    tests with a clear reason rather than letting them fail with an
    opaque connection-refused error."""
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except Exception:
        return False
