# mojo-marisa-trie

`mojo-marisa-trie` is a standalone Mojo port of the static lookup and prefix
traversal at the center of Python's
[`marisa-trie`](https://github.com/pytries/marisa-trie) package. It provides a
drop-in `marisa_trie` module for the covered API and a separate
`mojo_marisa_trie` import for projects that want to opt in explicitly.

This is a useful compatibility port, not a reimplementation of the MARISA C++
storage format. It uses a compact, path-compressed static radix trie. Mojo
builds that topology and performs batched lookup; Python supplies the covered
high-level compatibility API and handles scalar lookup and prefix results.

## Covered API

- `Trie` and `BinaryTrie`: construction, membership, `get`, `key_id`,
  `restore_key`, iteration, `keys`, `items`, `prefixes`,
  `iter_prefixes_with_ids`, and `has_keys_with_prefix`.
- `BytesTrie` and `RecordTrie`: duplicate-key value lists, lookup, iteration,
  prefix filtering, and `struct` packing.
- `StringTrie`: unique string mappings, keys, values, items, prefix items,
  `key_trie`, and `value_trie`.
- All five classes support `tobytes`/`frombytes`, `read`/`write`,
  `save`/`load`, `map`, `mmap`, and pickle round trips using this project's
  deterministic format.
- `Trie.key_ids(iterable)` and `BinaryTrie.key_ids(iterable)` are extensions
  that pack a query batch and traverse it in parallel in one FFI call.

The constructor accepts upstream tuning arguments so existing covered call
sites continue to work, but `num_tries`, cache, order, tail, and weights do not
change this implementation. Generated IDs are stable lexicographic IDs rather
than MARISA's internal node IDs.

Not covered:

- Reading or writing native MARISA files and byte strings.
- MARISA's rank/select topology, tail modes, weighting, and configurable
  multi-trie construction.
- True zero-copy memory mapping. `mmap` is API-compatible but eagerly loads
  this project's format.
- Upstream's internal iteration order. Results contain the same entries but
  are returned in lexicographic byte order.

## Install and run

The repository pins the tested Mojo nightly and installs the real upstream
package for parity tests.

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-marisa-trie.so`.

```python
from marisa_trie import RecordTrie, Trie

trie = Trie(["app", "apple", "application", "banana"])

assert "apple" in trie
assert trie.prefixes("application/json") == [
    "app",
    "application",
]
assert trie.restore_key(trie["banana"]) == "banana"

# Port extension: -1 marks a miss.
ids = trie.key_ids(["app", "missing", "banana"])
assert ids[1] == -1

records = RecordTrie("<I", [("red", (1,)), ("red", (2,)), ("blue", (3,))])
assert records["red"] == [(1,), (2,)]
```

Run that example inside the environment with `pixi run python example.py`, or
start `pixi run python` and paste it.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic, against `marisa-trie` 1.4.1. Each timing is the best of
five runs, except construction which is the best of three. The last column is
upstream time divided by Mojo time; values below `1.0x` mean this port is
slower.

| operation | mojo-marisa-trie | marisa-trie 1.4.1 | upstream / Mojo |
|---|---:|---:|---:|
| build Trie (151k keys) | 160.49 ms | 113.26 ms | 0.71x |
| scalar get (100.5k queries) | 53.19 ms | 54.68 ms | 1.03x |
| bulk key_ids (100.5k queries) | 51.46 ms | 58.73 ms | 1.14x |
| prefixes (20k queries) | 21.21 ms | 19.65 ms | 0.93x |

Scalar dictionary lookup was 1.03x faster and bulk lookup was 1.14x faster
than the corresponding upstream scalar loops on this run. Construction and
prefix lookup remained slower.

No GPU path is provided. Construction and traversal are branch-heavy,
low-arithmetic-intensity byte operations, so transfer and launch overhead would
dominate useful GPU work.

Storage is also less succinct than MARISA:

| serialized trie | mojo-marisa-trie | marisa-trie 1.4.1 |
|---|---:|---:|
| 151k keys | 4,668,021 B | 410,224 B |

The in-memory Mojo topology arrays for that trie occupy 3,330,905 bytes. These
numbers are intentionally reported rather than presenting this radix layout
as equivalent to MARISA's information-theoretic compression.

## How it works

Construction sorts the keys, packs their UTF-8 bytes (`BinaryTrie` keeps bytes
unchanged), and builds compressed radix segments directly in Mojo without an
intermediate byte-per-node trie. Nodes are assigned breadth-first. Five
caller-owned arrays cross the FFI boundary:

- an `int32` child-edge offset per node;
- `int32` segment offsets and lengths per edge;
- one contiguous byte pool containing path segments;
- an `int32` terminal ID per node, with `-1` for non-terminals.

Edges are stored in first-byte order. The child node for edge `e` is always
`e + 1`, so no child-pointer array is needed. During `key_ids`, Mojo
binary-searches the first byte, compares compressed segments in SIMD-width
chunks with a scalar tail, and advances to that implicit child. Scalar exact
lookup uses a lazily-created Python ID map, avoiding per-query FFI overhead.
Prefix lookup only tests key lengths that can be terminal. For `key_ids`,
Python concatenates all query bytes and supplies `int64` boundaries; Mojo
keeps batches of at most 4,096 queries serial and processes larger batches in
serial chunks. Boundary validation, common-prefix comparison, segment copying,
and segment comparison use SIMD-width blocks with scalar remainder loops.

ctypes passes every NumPy buffer as an integer address together with its
logical length. Python checks the exact dtype, one-dimensional C-contiguous
layout, writability, and non-null address before each call. The Mojo wrappers
validate capacities, offsets, and 32-bit topology bounds before rebuilding
`UnsafePointer[..., AnyOrigin[mut=True]]` values. Python keeps every allocation
alive for the synchronous call; the shared library neither allocates returned
memory nor retains pointers.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The test suite runs behavioral parity checks against the real upstream
package. The benchmark task holds `/tmp/mojo-bench.lock` to avoid overlapping
factory benchmarks.
