from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_MARISA_TRIE_LIB") or os.path.join(
    ROOT, "dist", "libmojo-marisa-trie.so"
)

I = ctypes.c_int64

_SIGNATURES = {
    "mmtrie_lookup_many": ([I] * 15, I),
    "mmtrie_build": ([I] * 15, I),
    "mmtrie_simd_width": ([], I),
    "mmtrie_batch_grain": ([], I),
}


class BuildError(RuntimeError):
    pass


def build() -> str:
    if os.path.exists(LIB):
        return LIB
    script = os.path.join(ROOT, "build", "build.sh")
    if not os.path.exists(script):
        raise BuildError(f"compiled library not found at {LIB}")
    proc = subprocess.run(
        ["bash", script], cwd=ROOT, capture_output=True, text=True, timeout=1800
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(
    array: np.ndarray, dtype: np.dtype | type, *, writable: bool = True
) -> int:
    """Return an ABI-safe address while the caller keeps ``array`` alive."""
    expected = np.dtype(dtype)
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype != expected:
        raise TypeError(f"expected {expected} buffer, got {array.dtype}")
    if array.ndim != 1 or not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be one-dimensional and C-contiguous")
    if writable and not array.flags.writeable:
        raise ValueError("FFI output buffers must be writable")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("FFI buffers must have a non-null address")
    return address
