from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import site
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(ROOT, "python")


def _load_upstream():
    for directory in site.getsitepackages():
        spec = importlib.machinery.PathFinder.find_spec("marisa_trie", [directory])
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        previous = sys.modules.get("marisa_trie")
        sys.modules["marisa_trie"] = module
        try:
            spec.loader.exec_module(module)
        finally:
            if previous is None:
                sys.modules.pop("marisa_trie", None)
            else:
                sys.modules["marisa_trie"] = previous
        return module
    return None


UPSTREAM = _load_upstream()
sys.path.insert(0, PYTHON)


@pytest.fixture(scope="session")
def upstream():
    if UPSTREAM is None:
        pytest.skip("upstream marisa-trie is not installed")
    return UPSTREAM
