from __future__ import annotations

import bisect
import os
import struct
import warnings
from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from ._lib import addr, lib

TINY_CACHE = 2048
SMALL_CACHE = 1024
NORMAL_CACHE = 512
LARGE_CACHE = 256
HUGE_CACHE = 128
DEFAULT_CACHE = NORMAL_CACHE
MIN_NUM_TRIES = 1
MAX_NUM_TRIES = 127
DEFAULT_NUM_TRIES = 3
TEXT_TAIL = 4096
BINARY_TAIL = 8192
DEFAULT_TAIL = TEXT_TAIL
LABEL_ORDER = 65536
WEIGHT_ORDER = 131072
DEFAULT_ORDER = WEIGHT_ORDER
_I32_MAX = np.iinfo(np.int32).max

_MAGIC = b"MMTRIE\x01\x00"
_HEADER = struct.Struct("<8scIQ")
_U32 = struct.Struct("<I")


def _encode_blob(kind: bytes, metadata: bytes, pairs: Iterable[tuple[bytes, bytes]]) -> bytes:
    rows = list(pairs)
    chunks = [_HEADER.pack(_MAGIC, kind, len(metadata), len(rows)), metadata]
    for key, value in rows:
        chunks.extend((_U32.pack(len(key)), key, _U32.pack(len(value)), value))
    return b"".join(chunks)


def _decode_blob(data: bytes | bytearray | memoryview) -> tuple[bytes, bytes, list[tuple[bytes, bytes]]]:
    view = memoryview(data)
    if len(view) < _HEADER.size:
        raise ValueError("invalid mojo-marisa-trie data")
    magic, kind, metadata_len, count = _HEADER.unpack_from(view)
    if magic != _MAGIC:
        raise ValueError("invalid mojo-marisa-trie magic")
    pos = _HEADER.size
    end_metadata = pos + metadata_len
    if end_metadata > len(view):
        raise ValueError("truncated mojo-marisa-trie metadata")
    metadata = bytes(view[pos:end_metadata])
    pos = end_metadata
    rows: list[tuple[bytes, bytes]] = []
    for _ in range(count):
        if pos + 4 > len(view):
            raise ValueError("truncated mojo-marisa-trie key")
        key_len = _U32.unpack_from(view, pos)[0]
        pos += 4
        key_end = pos + key_len
        if key_end + 4 > len(view):
            raise ValueError("truncated mojo-marisa-trie key")
        key = bytes(view[pos:key_end])
        value_len = _U32.unpack_from(view, key_end)[0]
        pos = key_end + 4
        value_end = pos + value_len
        if value_end > len(view):
            raise ValueError("truncated mojo-marisa-trie value")
        rows.append((key, bytes(view[pos:value_end])))
        pos = value_end
    if pos != len(view):
        raise ValueError("trailing bytes in mojo-marisa-trie data")
    return kind, metadata, rows


def _restore(cls, data: bytes):
    obj = cls.__new__(cls)
    return obj.frombytes(data)


class _StaticTrie:
    _binary = False
    _kind = b"U"

    def __init__(
        self,
        arg=None,
        num_tries=DEFAULT_NUM_TRIES,
        binary=False,
        cache_size=DEFAULT_CACHE,
        order=DEFAULT_ORDER,
        weights=None,
    ):
        del num_tries, binary, cache_size, order, weights
        keys = [] if arg is None else list(arg)
        if self._binary:
            if not all(isinstance(key, bytes) for key in keys):
                raise TypeError("BinaryTrie keys must be bytes")
        elif not all(isinstance(key, str) for key in keys):
            raise TypeError("Trie keys must be str")
        ordered = sorted(set(keys))
        if len(ordered) >= 2**31:
            raise OverflowError("this implementation supports fewer than 2^31 keys")
        self._keys_by_id = ordered
        self._key_to_id = None
        self._key_lengths = sorted({len(key) for key in ordered})
        self._build_arrays(ordered)

    def _normalize_key(self, key):
        if self._binary:
            if not isinstance(key, bytes):
                raise TypeError("BinaryTrie keys must be bytes")
            return key
        if not isinstance(key, str):
            raise TypeError("Trie keys must be str")
        return key

    def _bytes(self, key) -> bytes:
        key = self._normalize_key(key)
        return key if self._binary else key.encode("utf-8")

    def _from_bytes(self, key: bytes):
        return key if self._binary else key.decode("utf-8")

    def _build_arrays(self, keys) -> None:
        encoded = keys if self._binary else list(map(str.encode, keys))
        if len(encoded) > (_I32_MAX - 1) // 2:
            raise OverflowError("too many keys for the int32 Mojo topology")
        key_offsets = np.empty(len(encoded) + 1, dtype=np.int64)
        key_offsets[0] = 0
        if encoded:
            lengths = np.fromiter(map(len, encoded), dtype=np.int64, count=len(encoded))
            np.cumsum(lengths, out=key_offsets[1:])
        raw = b"".join(encoded)
        if len(raw) > _I32_MAX:
            raise OverflowError("encoded keys exceed the int32 Mojo byte-pool limit")
        data = self._buffer(raw)

        max_nodes = max(1, 2 * len(encoded))
        node_lo = np.empty(max_nodes, dtype=np.int32)
        node_hi = np.empty(max_nodes, dtype=np.int32)
        node_depth = np.empty(max_nodes, dtype=np.int32)
        self._offsets = np.empty(max_nodes + 1, dtype=np.int32)
        self._segment_offsets = np.empty(max(1, max_nodes - 1), dtype=np.int32)
        self._segment_lengths = np.empty(max(1, max_nodes - 1), dtype=np.int32)
        self._labels = np.empty(max(1, len(raw)), dtype=np.uint8)
        self._terminal = np.empty(max_nodes, dtype=np.int32)
        stats = np.empty(2, dtype=np.int64)
        status = lib().mmtrie_build(
            addr(data, np.uint8, writable=False),
            len(raw),
            addr(key_offsets, np.int64, writable=False),
            len(encoded),
            addr(node_lo, np.int32),
            addr(node_hi, np.int32),
            addr(node_depth, np.int32),
            max_nodes,
            addr(self._offsets, np.int32),
            addr(self._segment_offsets, np.int32),
            addr(self._segment_lengths, np.int32),
            addr(self._labels, np.uint8),
            len(self._labels),
            addr(self._terminal, np.int32),
            addr(stats, np.int64),
        )
        if status:
            raise RuntimeError(f"Mojo trie construction failed (status {status})")

        node_count, pool_size = map(int, stats)
        self._offsets.resize(node_count + 1, refcheck=False)
        self._segment_offsets.resize(max(1, node_count - 1), refcheck=False)
        self._segment_lengths.resize(max(1, node_count - 1), refcheck=False)
        self._labels.resize(max(1, pool_size), refcheck=False)
        self._terminal.resize(node_count, refcheck=False)
        self._pool_size = pool_size
        self._edge_count = node_count - 1
        self._topology_addrs = tuple(
            map(
                lambda item: addr(item[0], item[1]),
                (
                    (self._offsets, np.int32),
                    (self._segment_offsets, np.int32),
                    (self._segment_lengths, np.int32),
                    (self._labels, np.uint8),
                    (self._terminal, np.int32),
                ),
            )
        )

    @staticmethod
    def _buffer(data: bytes) -> np.ndarray:
        return np.frombuffer(data if data else b"\0", dtype=np.uint8)

    def _id_map(self) -> dict:
        mapping = self._key_to_id
        if mapping is None:
            mapping = {key: index for index, key in enumerate(self._keys_by_id)}
            self._key_to_id = mapping
        return mapping

    def _lookup_id(self, key) -> int:
        key = self._normalize_key(key)
        return self._id_map().get(key, -1)

    def key_ids(self, keys: Iterable) -> np.ndarray:
        encoded = [self._bytes(key) for key in keys]
        query_offsets = np.empty(len(encoded) + 1, dtype=np.int64)
        query_offsets[0] = 0
        if encoded:
            lengths = np.fromiter(map(len, encoded), dtype=np.int64, count=len(encoded))
            np.cumsum(lengths, out=query_offsets[1:])
        raw = b"".join(encoded)
        data = self._buffer(raw)
        result = np.empty(len(encoded), dtype=np.int32)
        if encoded:
            status = lib().mmtrie_lookup_many(
                *self._topology_addrs,
                len(self._offsets) - 1,
                self._edge_count,
                self._pool_size,
                addr(data, np.uint8, writable=False),
                len(raw),
                addr(query_offsets, np.int64, writable=False),
                len(query_offsets),
                len(encoded),
                addr(result, np.int32),
                len(result),
            )
            if status:
                raise RuntimeError(f"Mojo batch lookup failed (status {status})")
        return result

    @property
    def nbytes(self) -> int:
        return (
            self._offsets.nbytes
            + self._edge_count * 8
            + self._pool_size
            + self._terminal.nbytes
        )

    def __len__(self) -> int:
        return len(self._keys_by_id)

    def __bool__(self) -> bool:
        return bool(self._keys_by_id)

    def __contains__(self, key) -> bool:
        try:
            key = self._normalize_key(key)
        except TypeError:
            return False
        return key in self._id_map()

    def __getitem__(self, key) -> int:
        return self.key_id(key)

    def get(self, key, default=None):
        try:
            key = self._normalize_key(key)
        except TypeError:
            return default
        found = self._id_map().get(key, -1)
        return found if found >= 0 else default

    def key_id(self, key) -> int:
        key = self._normalize_key(key)
        found = self._id_map().get(key, -1)
        if found < 0:
            raise KeyError(key)
        return found

    def restore_key(self, index: int):
        if not isinstance(index, int) or index < 0 or index >= len(self):
            raise KeyError(index)
        return self._keys_by_id[index]

    def __iter__(self):
        return self.iterkeys()

    def iterkeys(self, prefix=None):
        prefix = (b"" if self._binary else "") if prefix is None else self._normalize_key(prefix)
        start = bisect.bisect_left(self._keys_by_id, prefix)
        for key in self._keys_by_id[start:]:
            if not key.startswith(prefix):
                break
            yield key

    def keys(self, prefix=None) -> list:
        return list(self.iterkeys(prefix))

    def iteritems(self, prefix=b""):
        if not self._binary and prefix == b"":
            prefix = ""
        for key in self.iterkeys(prefix):
            yield key, self._lookup_id(key)

    def items(self, prefix=b"") -> list:
        return list(self.iteritems(prefix))

    def iter_prefixes_with_ids(self, key):
        key = self._normalize_key(key)
        key_len = len(key)
        mapping = self._id_map()
        for prefix_len in self._key_lengths:
            if prefix_len > key_len:
                break
            prefix = key[:prefix_len]
            index = mapping.get(prefix)
            if index is not None:
                yield prefix, index

    def iter_prefixes(self, key):
        for prefix, _ in self.iter_prefixes_with_ids(key):
            yield prefix

    def prefixes(self, key) -> list:
        key = self._normalize_key(key)
        key_len = len(key)
        mapping = self._id_map()
        return [
            prefix
            for prefix_len in self._key_lengths
            if prefix_len <= key_len
            and (prefix := key[:prefix_len]) in mapping
        ]

    def has_keys_with_prefix(self, prefix=None) -> bool:
        warnings.warn(
            "Trie.has_keys_with_prefix is deprecated; use Trie.iterkeys instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return next(self.iterkeys(prefix), None) is not None

    def tobytes(self) -> bytes:
        rows = ((self._bytes(key), b"") for key in self._keys_by_id)
        return _encode_blob(self._kind, b"", rows)

    def frombytes(self, data):
        kind, _, rows = _decode_blob(data)
        if kind != self._kind:
            raise ValueError("serialized trie has the wrong key type")
        self.__init__(self._from_bytes(key) for key, _ in rows)
        return self

    def write(self, f) -> None:
        f.write(self.tobytes())

    def read(self, f):
        return self.frombytes(f.read())

    def save(self, path) -> None:
        with open(os.fspath(path), "wb") as stream:
            self.write(stream)

    def load(self, path):
        with open(os.fspath(path), "rb") as stream:
            return self.read(stream)

    def mmap(self, path):
        return self.load(path)

    def map(self, buffer):
        return self.frombytes(buffer)

    def __reduce__(self):
        return _restore, (type(self), self.tobytes())

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self._keys_by_id == other._keys_by_id

    def __ne__(self, other) -> bool:
        return not self == other


class Trie(_StaticTrie):
    """A static trie mapping Unicode keys to generated integer IDs."""


class BinaryTrie(_StaticTrie):
    """A static trie mapping bytes keys to generated integer IDs."""

    _binary = True
    _kind = b"B"

    def iteritems(self, prefix=b""):
        return super().iteritems(prefix)

    def items(self, prefix=b"") -> list:
        return list(self.iteritems(prefix))


class BytesTrie:
    """A static trie mapping Unicode keys to lists of bytes values."""

    _kind = b"V"

    def __init__(self, arg=None, value_separator=b"\xff", **options):
        del value_separator
        grouped: dict[str, set[bytes]] = defaultdict(set)
        for key, value in (() if arg is None else arg):
            if not isinstance(key, str):
                raise TypeError("BytesTrie keys must be str")
            if not isinstance(value, bytes):
                raise TypeError("BytesTrie values must be bytes")
            grouped[key].add(value)
        self._values = {key: sorted(values) for key, values in grouped.items()}
        self._key_trie = Trie(self._values, **options)

    @property
    def nbytes(self) -> int:
        return self._key_trie.nbytes + sum(
            len(key.encode("utf-8")) + sum(map(len, values))
            for key, values in self._values.items()
        )

    def __len__(self) -> int:
        return sum(map(len, self._values.values()))

    def __bool__(self) -> bool:
        return bool(self._values)

    def __contains__(self, key) -> bool:
        return key in self._key_trie

    def __getitem__(self, key):
        values = self.get_value(key)
        if not values:
            raise KeyError(key)
        return values

    def get(self, key, default=None):
        values = self.get_value(key)
        return values if values else default

    def get_value(self, key) -> list[bytes]:
        if not isinstance(key, str):
            raise TypeError("BytesTrie keys must be str")
        return list(self._values.get(key, ()))

    def b_get_value(self, key) -> list[bytes]:
        if not isinstance(key, bytes):
            raise TypeError("key must be bytes")
        return self.get_value(key.decode("utf-8"))

    def __iter__(self):
        return self.iterkeys()

    def iterkeys(self, prefix=""):
        for key in self._key_trie.iterkeys(prefix):
            for _ in self._values[key]:
                yield key

    def keys(self, prefix="") -> list:
        return list(self.iterkeys(prefix))

    def iteritems(self, prefix=""):
        for key in self._key_trie.iterkeys(prefix):
            for value in self._values[key]:
                yield key, value

    def items(self, prefix="") -> list:
        return list(self.iteritems(prefix))

    def prefixes(self, key) -> list:
        return [prefix for prefix in self._key_trie.prefixes(key) if prefix]

    def has_keys_with_prefix(self, prefix="") -> bool:
        return next(self._key_trie.iterkeys(prefix), None) is not None

    def tobytes(self) -> bytes:
        rows = (
            (key.encode("utf-8"), value)
            for key in self._key_trie.keys()
            for value in self._values[key]
        )
        return _encode_blob(self._kind, b"", rows)

    def frombytes(self, data):
        kind, _, rows = _decode_blob(data)
        if kind != self._kind:
            raise ValueError("serialized trie has the wrong value type")
        self.__init__((key.decode("utf-8"), value) for key, value in rows)
        return self

    write = _StaticTrie.write
    read = _StaticTrie.read
    save = _StaticTrie.save
    load = _StaticTrie.load
    mmap = _StaticTrie.mmap
    map = _StaticTrie.map
    __reduce__ = _StaticTrie.__reduce__

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self._values == other._values

    def __ne__(self, other) -> bool:
        return not self == other


class RecordTrie(BytesTrie):
    """A static trie mapping Unicode keys to lists of struct-packed tuples."""

    _kind = b"R"

    def __init__(self, fmt, arg=None, **options):
        self.fmt = fmt
        self._struct = struct.Struct(fmt)
        packed = (
            ()
            if arg is None
            else ((key, self._struct.pack(*value)) for key, value in arg)
        )
        super().__init__(packed, **options)

    def get_value(self, key) -> list[tuple]:
        return [self._struct.unpack(value) for value in super().get_value(key)]

    def b_get_value(self, key) -> list[tuple]:
        if not isinstance(key, bytes):
            raise TypeError("key must be bytes")
        return [
            self._struct.unpack(value)
            for value in self._values.get(key.decode("utf-8"), ())
        ]

    def iteritems(self, prefix=""):
        for key, value in BytesTrie.iteritems(self, prefix):
            yield key, self._struct.unpack(value)

    def items(self, prefix="") -> list:
        return list(self.iteritems(prefix))

    def tobytes(self) -> bytes:
        rows = (
            (key.encode("utf-8"), value)
            for key in self._key_trie.keys()
            for value in self._values[key]
        )
        return _encode_blob(self._kind, self.fmt.encode("utf-8"), rows)

    def frombytes(self, data):
        kind, metadata, rows = _decode_blob(data)
        if kind != self._kind:
            raise ValueError("serialized trie has the wrong value type")
        fmt = metadata.decode("utf-8")
        unpacker = struct.Struct(fmt)
        self.__init__(
            fmt,
            ((key.decode("utf-8"), unpacker.unpack(value)) for key, value in rows),
        )
        return self


class StringTrie:
    """A static trie mapping unique Unicode keys to Unicode values."""

    _kind = b"S"

    def __init__(self, arg=None, **options):
        pairs = [] if arg is None else list(arg)
        values: dict[str, str] = {}
        for key, value in pairs:
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("StringTrie keys and values must be str")
            if key in values:
                raise ValueError("duplicate keys are not allowed")
            values[key] = value
        self._values = values
        self._key_trie = Trie(values, **options)

    @property
    def nbytes(self) -> int:
        return self._key_trie.nbytes + sum(
            len(key.encode("utf-8")) + len(value.encode("utf-8"))
            for key, value in self._values.items()
        )

    def __len__(self) -> int:
        return len(self._values)

    def __bool__(self) -> bool:
        return bool(self._values)

    def __contains__(self, key) -> bool:
        return key in self._key_trie

    def __getitem__(self, key) -> str:
        if key not in self._key_trie:
            raise KeyError(key)
        return self._values[key]

    def get(self, key, default=None):
        return self._values.get(key, default)

    def __iter__(self):
        return self.iterkeys()

    def iterkeys(self, prefix=""):
        return self._key_trie.iterkeys(prefix)

    def keys(self, prefix="") -> list:
        return list(self.iterkeys(prefix))

    def itervalues(self, prefix=""):
        for key in self.iterkeys(prefix):
            yield self._values[key]

    def values(self, prefix="") -> list:
        return list(self.itervalues(prefix))

    def iteritems(self, prefix=""):
        for key in self.iterkeys(prefix):
            yield key, self._values[key]

    def items(self, prefix="") -> list:
        return list(self.iteritems(prefix))

    def iter_prefixes(self, key):
        return self._key_trie.iter_prefixes(key)

    def prefixes(self, key) -> list:
        return list(self.iter_prefixes(key))

    def iter_prefix_items(self, key):
        for prefix in self.iter_prefixes(key):
            yield prefix, self._values[prefix]

    def prefix_items(self, key) -> list:
        return list(self.iter_prefix_items(key))

    @property
    def key_trie(self):
        return Trie(self._values)

    @property
    def value_trie(self):
        return Trie(self._values.values())

    def tobytes(self) -> bytes:
        rows = (
            (key.encode("utf-8"), self._values[key].encode("utf-8"))
            for key in self._key_trie.keys()
        )
        return _encode_blob(self._kind, b"", rows)

    def frombytes(self, data):
        kind, _, rows = _decode_blob(data)
        if kind != self._kind:
            raise ValueError("serialized trie has the wrong value type")
        self.__init__(
            (key.decode("utf-8"), value.decode("utf-8")) for key, value in rows
        )
        return self

    write = _StaticTrie.write
    read = _StaticTrie.read
    save = _StaticTrie.save
    load = _StaticTrie.load
    mmap = _StaticTrie.mmap
    map = _StaticTrie.map
    __reduce__ = _StaticTrie.__reduce__

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self._values == other._values

    def __ne__(self, other) -> bool:
        return not self == other
