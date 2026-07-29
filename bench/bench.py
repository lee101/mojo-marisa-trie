"""Benchmark mojo-marisa-trie against upstream marisa-trie."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import math
import os
import platform
import site
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_upstream():
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
    raise RuntimeError("upstream marisa-trie is required for the benchmark")


upstream = load_upstream()
sys.path.insert(0, os.path.join(ROOT, "python"))
import mojo_marisa_trie as mm  # noqa: E402


def timeit(fn, repeat=5) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


N = 150_000
KEYS = [f"group/{i % 1000:04d}/item/{i:07d}" for i in range(N)]
KEYS.extend(f"group/{i:04d}" for i in range(1000))
QUERIES = KEYS[::2] + [f"missing/{i:07d}" for i in range(25_000)]
PREFIX_QUERIES = [f"group/{i % 1000:04d}/item/{i:07d}/tail" for i in range(20_000)]


def main() -> None:
    mm.Trie(["warmup"])["warmup"]
    ours = mm.Trie(KEYS)
    theirs = upstream.Trie(KEYS)

    ours_bulk = ours.key_ids(QUERIES)
    upstream_hits = np.fromiter(
        (theirs.get(key, -1) >= 0 for key in QUERIES), dtype=np.bool_
    )
    assert np.array_equal(ours_bulk >= 0, upstream_hits)
    assert [ours.prefixes(key) for key in PREFIX_QUERIES[:100]] == [
        theirs.prefixes(key) for key in PREFIX_QUERIES[:100]
    ]

    cases = [
        (
            "build Trie (151k keys)",
            lambda: mm.Trie(KEYS),
            lambda: upstream.Trie(KEYS),
            3,
        ),
        (
            "scalar get (100.5k queries)",
            lambda: sum(ours.get(key, -1) >= 0 for key in QUERIES),
            lambda: sum(theirs.get(key, -1) >= 0 for key in QUERIES),
            5,
        ),
        (
            "bulk key_ids (100.5k queries)",
            lambda: ours.key_ids(QUERIES),
            lambda: np.fromiter(
                (theirs.get(key, -1) for key in QUERIES), dtype=np.int32
            ),
            5,
        ),
        (
            "prefixes (20k queries)",
            lambda: [ours.prefixes(key) for key in PREFIX_QUERIES],
            lambda: [theirs.prefixes(key) for key in PREFIX_QUERIES],
            5,
        ),
    ]

    print(f"Machine: {cpu_name()}; {platform.platform()}")
    print()
    print("| operation | mojo-marisa-trie | marisa-trie 1.4.1 | upstream / Mojo |")
    print("|---|---:|---:|---:|")
    for name, mojo_fn, upstream_fn, repeat in cases:
        mojo_time = timeit(mojo_fn, repeat)
        upstream_time = timeit(upstream_fn, repeat)
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {upstream_time / mojo_time:.2f}x |"
        )

    print()
    print("| serialized trie | mojo-marisa-trie | marisa-trie 1.4.1 |")
    print("|---|---:|---:|")
    print(f"| 151k keys | {len(ours.tobytes()):,} B | {len(theirs.tobytes()):,} B |")
    print(f"Mojo topology arrays: {ours.nbytes:,} B")


if __name__ == "__main__":
    main()
