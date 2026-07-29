from __future__ import annotations

import io
import pickle
import struct

import numpy as np
import pytest

import mojo_marisa_trie as mm
from mojo_marisa_trie._lib import addr, lib


TEXT_KEYS = [
    "",
    "a",
    "app",
    "apple",
    "application",
    "bar",
    "foo",
    "foobar",
    "naïve",
    "日本",
    "𐍈",
    "𐍈a",
]


def test_public_constants_match_upstream(upstream):
    names = [
        "TINY_CACHE",
        "SMALL_CACHE",
        "NORMAL_CACHE",
        "LARGE_CACHE",
        "HUGE_CACHE",
        "DEFAULT_CACHE",
        "MIN_NUM_TRIES",
        "MAX_NUM_TRIES",
        "DEFAULT_NUM_TRIES",
        "TEXT_TAIL",
        "BINARY_TAIL",
        "DEFAULT_TAIL",
        "LABEL_ORDER",
        "WEIGHT_ORDER",
        "DEFAULT_ORDER",
    ]
    assert {name: getattr(mm, name) for name in names} == {
        name: getattr(upstream, name) for name in names
    }


def test_trie_exact_lookup_and_restore_parity(upstream):
    ours = mm.Trie(TEXT_KEYS + ["foo"])
    theirs = upstream.Trie(TEXT_KEYS + ["foo"])
    assert len(ours) == len(theirs)
    assert sorted(ours.keys()) == sorted(theirs.keys())
    for key in TEXT_KEYS:
        assert (key in ours) == (key in theirs)
        assert ours.restore_key(ours[key]) == theirs.restore_key(theirs[key]) == key
        assert ours.get(key) == ours[key]
    assert ours.get("missing", 42) == theirs.get("missing", 42) == 42
    with pytest.raises(KeyError):
        ours["missing"]


@pytest.mark.parametrize("query", ["", "applejack", "application/x", "foobar!", "𐍈abc", "none"])
def test_trie_prefixes_match_upstream(upstream, query):
    ours = mm.Trie(TEXT_KEYS)
    theirs = upstream.Trie(TEXT_KEYS)
    assert ours.prefixes(query) == theirs.prefixes(query)
    assert list(ours.iter_prefixes(query)) == list(theirs.iter_prefixes(query))
    ours_pairs = list(ours.iter_prefixes_with_ids(query))
    theirs_pairs = list(theirs.iter_prefixes_with_ids(query))
    assert [key for key, _ in ours_pairs] == [key for key, _ in theirs_pairs]
    assert all(ours.restore_key(index) == key for key, index in ours_pairs)


@pytest.mark.parametrize("prefix", ["", "a", "app", "foo", "日", "𐍈", "missing"])
def test_trie_enumeration_parity(upstream, prefix):
    ours = mm.Trie(TEXT_KEYS)
    theirs = upstream.Trie(TEXT_KEYS)
    assert sorted(ours.keys(prefix)) == sorted(theirs.keys(prefix))
    assert sorted(ours.iterkeys(prefix)) == sorted(theirs.iterkeys(prefix))
    assert {key for key, _ in ours.items(prefix)} == {
        key for key, _ in theirs.items(prefix)
    }


def test_binary_trie_parity(upstream):
    keys = [b"", b"a", b"app", b"apple", b"\x00", b"\xff", b"\x00\xff", b"app"]
    ours = mm.BinaryTrie(keys)
    theirs = upstream.BinaryTrie(keys)
    assert sorted(ours.keys()) == sorted(theirs.keys())
    assert ours.prefixes(b"apple!") == theirs.prefixes(b"apple!")
    for key in keys:
        assert ours.restore_key(ours[key]) == theirs.restore_key(theirs[key]) == key
    assert ours.get(b"absent", -7) == theirs.get(b"absent", -7) == -7
    with pytest.deprecated_call():
        assert ours.has_keys_with_prefix()
    with pytest.deprecated_call():
        assert not ours.has_keys_with_prefix(b"missing")


def test_bulk_lookup_matches_scalar_and_reference():
    keys = [f"key/{i:05d}" for i in range(10_000)]
    queries = keys[::3] + ["missing", "", "key/99999"]
    trie = mm.Trie(keys)
    result = trie.key_ids(queries)
    expected = np.array([trie.get(key, -1) for key in queries], dtype=np.int32)
    assert np.array_equal(result, expected)
    assert all(
        index == -1 or trie.restore_key(int(index)) == key
        for key, index in zip(queries, result)
    )


def test_simd_segment_comparison_and_scalar_tail():
    width = int(lib().mmtrie_simd_width())
    assert width > 1
    keys = [
        b"a" * (width - 1),
        b"b" * width,
        b"c" * (width + 1),
    ] + [b"d" * (2 * width + 1) + suffix for suffix in (b"x", b"y")]
    trie = mm.BinaryTrie(keys)
    queries = keys + [
        b"a" * (width - 2) + b"z",
        b"b" * (width - 1) + b"z",
        b"c" * width + b"z",
        b"d" * (2 * width + 1) + b"z",
    ]
    result = trie.key_ids(queries)
    expected = np.array([trie.get(key, -1) for key in queries], dtype=np.int32)
    assert np.array_equal(result, expected)


def test_bulk_parallel_threshold_matches_serial_path():
    grain = int(lib().mmtrie_batch_grain())
    keys = [f"parallel/{i:05d}" for i in range(grain + 1)]
    trie = mm.Trie(keys)
    serial = trie.key_ids(keys[:grain])
    parallel = trie.key_ids(keys)
    assert np.array_equal(serial, np.arange(grain, dtype=np.int32))
    assert np.array_equal(parallel, np.arange(grain + 1, dtype=np.int32))


def test_ffi_rejects_invalid_lengths_and_buffer_layouts():
    trie = mm.Trie(["a", "ab"])
    data = np.frombuffer(b"a", dtype=np.uint8)
    query_offsets = np.array([0, 2], dtype=np.int64)
    result = np.empty(1, dtype=np.int32)
    status = lib().mmtrie_lookup_many(
        *trie._topology_addrs,
        len(trie._offsets) - 1,
        trie._edge_count,
        trie._pool_size,
        addr(data, np.uint8, writable=False),
        len(data),
        addr(query_offsets, np.int64, writable=False),
        len(query_offsets),
        1,
        addr(result, np.int32),
        len(result),
    )
    assert status != 0

    with pytest.raises(TypeError, match="expected int64"):
        addr(np.array([0], dtype=np.int32), np.int64)
    with pytest.raises(ValueError, match="contiguous"):
        addr(np.arange(6, dtype=np.int64)[::2], np.int64)
    readonly = np.frombuffer(b"x", dtype=np.uint8)
    with pytest.raises(ValueError, match="writable"):
        addr(readonly, np.uint8)


def test_bytes_trie_parity(upstream):
    data = [
        ("", b"empty"),
        ("foo", b"z"),
        ("bar", b"b"),
        ("foo", b"a"),
        ("foobar", b"x\x00y"),
        ("foo", b"a"),
        ("日本", b"\xff"),
    ]
    ours = mm.BytesTrie(data)
    theirs = upstream.BytesTrie(data)
    assert len(ours) == len(theirs)
    assert sorted(ours.items()) == sorted(theirs.items())
    assert sorted(ours.keys("fo")) == sorted(theirs.keys("fo"))
    assert ours["foo"] == theirs["foo"]
    assert ours.get("missing", 9) == theirs.get("missing", 9) == 9
    assert ours.prefixes("foobarbaz") == theirs.prefixes("foobarbaz")
    assert ours.b_get_value(b"foo") == ours.get_value("foo")
    assert list(ours.iterkeys("fo")) == ours.keys("fo")
    with pytest.raises(TypeError):
        ours.b_get_value("foo")
    assert ours.has_keys_with_prefix("fo")


def test_record_trie_parity(upstream):
    data = [
        ("foo", (1, 2.5)),
        ("bar", (3, -1.0)),
        ("foo", (2, 4.5)),
        ("日本", (65535, 0.25)),
    ]
    ours = mm.RecordTrie("<Hd", data)
    theirs = upstream.RecordTrie("<Hd", data)
    assert sorted(ours.items()) == sorted(theirs.items())
    assert ours["foo"] == theirs["foo"]
    assert ours.get("missing", None) is theirs.get("missing", None)
    assert ours.prefixes("foobar") == theirs.prefixes("foobar")
    assert ours.b_get_value(b"foo") == ours.get_value("foo")


def test_string_trie_parity(upstream):
    data = [
        ("", "root"),
        ("foo", "one"),
        ("foobar", "two"),
        ("bar", "one"),
        ("日本", "value"),
    ]
    ours = mm.StringTrie(data)
    theirs = upstream.StringTrie(data)
    assert sorted(ours.items()) == sorted(theirs.items())
    assert ours.prefix_items("foobar!") == theirs.prefix_items("foobar!")
    assert sorted(ours.values()) == sorted(theirs.values())
    assert sorted(ours.key_trie.keys()) == sorted(theirs.key_trie.keys())
    assert sorted(ours.value_trie.keys()) == sorted(theirs.value_trie.keys())
    assert list(ours.iter_prefixes("foobar!")) == theirs.prefixes("foobar!")
    assert list(ours.iter_prefix_items("foobar!")) == ours.prefix_items("foobar!")
    assert list(ours.itervalues("foo")) == ours.values("foo")
    assert ours.get("missing", "fallback") == "fallback"
    with pytest.raises(ValueError):
        mm.StringTrie([("same", "a"), ("same", "b")])


@pytest.mark.parametrize(
    "factory",
    [
        lambda: mm.Trie(TEXT_KEYS),
        lambda: mm.BinaryTrie([b"", b"a", b"abc", b"\xff"]),
        lambda: mm.BytesTrie([("a", b"1"), ("a", b"2")]),
        lambda: mm.RecordTrie("<ii", [("a", (1, 2)), ("b", (3, 4))]),
        lambda: mm.StringTrie([("a", "x"), ("ab", "y")]),
    ],
)
def test_persistence_roundtrips(factory, tmp_path):
    original = factory()
    from_bytes = type(original).__new__(type(original)).frombytes(original.tobytes())
    assert from_bytes == original
    assert pickle.loads(pickle.dumps(original)) == original

    stream = io.BytesIO()
    original.write(stream)
    stream.seek(0)
    assert type(original).__new__(type(original)).read(stream) == original

    path = tmp_path / "trie.mmtrie"
    original.save(path)
    assert type(original).__new__(type(original)).load(path) == original
    assert type(original).__new__(type(original)).mmap(path) == original
    assert type(original).__new__(type(original)).map(memoryview(original.tobytes())) == original


def test_corrupt_serialization_is_rejected():
    with pytest.raises(ValueError, match="magic"):
        mm.Trie().frombytes(b"not a trie" + b"\0" * 30)
    data = mm.Trie(["a"]).tobytes()
    with pytest.raises(ValueError, match="truncated"):
        mm.Trie().frombytes(data[:-1])


def test_drop_in_module_name():
    import marisa_trie

    assert marisa_trie.Trie(["a"])["a"] == 0
    assert marisa_trie.Trie is mm.Trie


def test_layout_is_compact_and_static():
    keys = [f"common/prefix/{i:06d}" for i in range(20_000)]
    trie = mm.Trie(keys)
    assert trie.nbytes < sum(map(len, keys))
    assert trie._offsets.dtype == np.int32
    assert trie._labels.dtype == np.uint8
    assert trie._terminal.dtype == np.int32
    with pytest.raises(TypeError):
        trie["new"] = 1


def test_record_packing_matches_struct():
    trie = mm.RecordTrie(">Ih", [("x", (0xDEADBEEF, -12))])
    assert trie["x"] == [struct.unpack(">Ih", struct.pack(">Ih", 0xDEADBEEF, -12))]
