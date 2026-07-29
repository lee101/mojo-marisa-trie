from std.algorithm import parallelize
from std.sys import simd_width_of

comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime BATCH_GRAIN = 4096
comptime W = simd_width_of[DType.float64]()


def common_prefix_length(
    data: BPtr,
    first_start: Int,
    last_start: Int,
    start: Int,
    limit: Int,
) -> Int:
    var pos = start
    while pos + W <= limit:
        var first = data.load[width=W](first_start + pos)
        var last = data.load[width=W](last_start + pos)
        var differences = first.ne(last).select(
            SIMD[DType.uint8, W](1), SIMD[DType.uint8, W](0)
        )
        if differences.reduce_add() != 0:
            for lane in range(W):
                if first[lane] != last[lane]:
                    return pos + lane
        pos += W
    while pos < limit:
        if data[first_start + pos] != data[last_start + pos]:
            break
        pos += 1
    return pos


def copy_bytes(source: BPtr, source_start: Int, destination: BPtr, count: Int):
    var pos = 0
    while pos + W <= count:
        destination.store(pos, source.load[width=W](source_start + pos))
        pos += W
    while pos < count:
        destination[pos] = source[source_start + pos]
        pos += 1


def segments_equal(
    pool: BPtr,
    pool_start: Int,
    key: BPtr,
    key_start: Int,
    count: Int,
) -> Bool:
    var pos = 0
    while pos + W <= count:
        var left = pool.load[width=W](pool_start + pos)
        var right = key.load[width=W](key_start + pos)
        var differences = left.ne(right).select(
            SIMD[DType.uint8, W](1), SIMD[DType.uint8, W](0)
        )
        if differences.reduce_add() != 0:
            return False
        pos += W
    while pos < count:
        if pool[pool_start + pos] != key[key_start + pos]:
            return False
        pos += 1
    return True


def find_edge(
    offsets: I32Ptr, segment_offsets: I32Ptr, pool: BPtr, node: Int, label: UInt8
) -> Int:
    var lo = Int(offsets[node])
    var hi = Int(offsets[node + 1])
    while lo < hi:
        var mid = (lo + hi) // 2
        if pool[Int(segment_offsets[mid])] < label:
            lo = mid + 1
        else:
            hi = mid
    if (
        lo < Int(offsets[node + 1])
        and pool[Int(segment_offsets[lo])] == label
    ):
        return lo
    return -1


def lookup(
    offsets: I32Ptr,
    segment_offsets: I32Ptr,
    segment_lengths: I32Ptr,
    pool: BPtr,
    terminal: I32Ptr,
    key: BPtr,
    key_len: Int,
) -> Int:
    var node = 0
    var pos = 0
    while pos < key_len:
        var edge = find_edge(offsets, segment_offsets, pool, node, key[pos])
        if edge < 0:
            return -1
        var segment_start = Int(segment_offsets[edge])
        var segment_len = Int(segment_lengths[edge])
        if pos + segment_len > key_len:
            return -1
        if not segments_equal(pool, segment_start, key, pos, segment_len):
            return -1
        pos += segment_len
        node = edge + 1
    return Int(terminal[node])


def prefix_node(
    offsets: I32Ptr,
    segment_offsets: I32Ptr,
    segment_lengths: I32Ptr,
    pool: BPtr,
    key: BPtr,
    key_len: Int,
) -> Int:
    var node = 0
    var pos = 0
    while pos < key_len:
        var edge = find_edge(offsets, segment_offsets, pool, node, key[pos])
        if edge < 0:
            return -1
        var segment_start = Int(segment_offsets[edge])
        var segment_len = Int(segment_lengths[edge])
        if pos + segment_len > key_len:
            return -1
        if not segments_equal(pool, segment_start, key, pos, segment_len):
            return -1
        pos += segment_len
        node = edge + 1
    return node


@export("mmtrie_lookup_many")
def mmtrie_lookup_many(
    offsets_addr: Int,
    segment_offsets_addr: Int,
    segment_lengths_addr: Int,
    pool_addr: Int,
    terminal_addr: Int,
    node_count: Int,
    edge_count: Int,
    pool_size: Int,
    data_addr: Int,
    data_size: Int,
    query_offsets_addr: Int,
    query_offsets_count: Int,
    count: Int,
    ids_addr: Int,
    ids_count: Int,
) abi("C") -> Int:
    if (
        offsets_addr == 0 or segment_offsets_addr == 0
        or segment_lengths_addr == 0 or pool_addr == 0
        or terminal_addr == 0 or data_addr == 0
        or query_offsets_addr == 0 or ids_addr == 0
    ):
        return 1
    if (
        node_count <= 0 or edge_count != node_count - 1 or pool_size < 0
        or node_count > 2147483647 or pool_size > 2147483647
        or data_size < 0 or count < 0
        or query_offsets_count != count + 1 or ids_count < count
    ):
        return 2
    var offsets = I32Ptr(unsafe_from_address=offsets_addr)
    var segment_offsets = I32Ptr(unsafe_from_address=segment_offsets_addr)
    var segment_lengths = I32Ptr(unsafe_from_address=segment_lengths_addr)
    var pool = BPtr(unsafe_from_address=pool_addr)
    var terminal = I32Ptr(unsafe_from_address=terminal_addr)
    var data = BPtr(unsafe_from_address=data_addr)
    var query_offsets = I64Ptr(unsafe_from_address=query_offsets_addr)
    var ids = I32Ptr(unsafe_from_address=ids_addr)
    if query_offsets[0] != 0 or Int(query_offsets[count]) != data_size:
        return 3
    for i in range(count):
        if query_offsets[i] < 0 or query_offsets[i] > query_offsets[i + 1]:
            return 3
    if offsets[0] != 0 or Int(offsets[node_count]) != edge_count:
        return 4
    for node in range(node_count):
        if offsets[node] < 0 or offsets[node] > offsets[node + 1]:
            return 4
    for edge in range(edge_count):
        var start = Int(segment_offsets[edge])
        var length = Int(segment_lengths[edge])
        if start < 0 or length <= 0 or start + length > pool_size:
            return 5
    @parameter
    def lookup_chunk(chunk: Int):
        var first = chunk * BATCH_GRAIN
        var last = min(first + BATCH_GRAIN, count)
        for i in range(first, last):
            var start = Int(query_offsets[i])
            ids[i] = Int32(
                lookup(
                    offsets,
                    segment_offsets,
                    segment_lengths,
                    pool,
                    terminal,
                    data + start,
                    Int(query_offsets[i + 1]) - start,
                )
            )

    var chunks = (count + BATCH_GRAIN - 1) // BATCH_GRAIN
    if chunks > 1:
        parallelize[lookup_chunk](chunks, min(chunks, 8))
    elif count > 0:
        lookup_chunk(0)
    return 0


@export("mmtrie_build")
def mmtrie_build(
    data_addr: Int,
    data_size: Int,
    key_offsets_addr: Int,
    count: Int,
    node_lo_addr: Int,
    node_hi_addr: Int,
    node_depth_addr: Int,
    node_capacity: Int,
    offsets_addr: Int,
    segment_offsets_addr: Int,
    segment_lengths_addr: Int,
    pool_addr: Int,
    pool_capacity: Int,
    terminal_addr: Int,
    stats_addr: Int,
) abi("C") -> Int:
    if (
        data_addr == 0 or key_offsets_addr == 0 or node_lo_addr == 0
        or node_hi_addr == 0 or node_depth_addr == 0 or offsets_addr == 0
        or segment_offsets_addr == 0 or segment_lengths_addr == 0
        or pool_addr == 0 or terminal_addr == 0 or stats_addr == 0
    ):
        return 1
    if (
        count < 0 or data_size < 0 or pool_capacity < data_size
        or count > 1073741823 or data_size > 2147483647
        or node_capacity < 1 or (count > 0 and node_capacity < 2 * count)
    ):
        return 2
    var data = BPtr(unsafe_from_address=data_addr)
    var key_offsets = I64Ptr(unsafe_from_address=key_offsets_addr)
    if key_offsets[0] != 0 or Int(key_offsets[count]) != data_size:
        return 3
    for i in range(count):
        if key_offsets[i] < 0 or key_offsets[i] > key_offsets[i + 1]:
            return 3
    var node_lo = I32Ptr(unsafe_from_address=node_lo_addr)
    var node_hi = I32Ptr(unsafe_from_address=node_hi_addr)
    var node_depth = I32Ptr(unsafe_from_address=node_depth_addr)
    var offsets = I32Ptr(unsafe_from_address=offsets_addr)
    var segment_offsets = I32Ptr(unsafe_from_address=segment_offsets_addr)
    var segment_lengths = I32Ptr(unsafe_from_address=segment_lengths_addr)
    var pool = BPtr(unsafe_from_address=pool_addr)
    var terminal = I32Ptr(unsafe_from_address=terminal_addr)
    var stats = I64Ptr(unsafe_from_address=stats_addr)

    node_lo[0] = 0
    node_hi[0] = Int32(count)
    node_depth[0] = 0
    terminal[0] = -1
    if count > 0 and key_offsets[1] == 0:
        terminal[0] = 0

    var node_count = 1
    var edge_count = 0
    var pool_size = 0
    var node = 0
    while node < node_count:
        offsets[node] = Int32(edge_count)
        var lo = Int(node_lo[node])
        var hi = Int(node_hi[node])
        var depth = Int(node_depth[node])
        var scan = lo
        if (
            scan < hi
            and Int(key_offsets[scan + 1] - key_offsets[scan]) == depth
        ):
            scan += 1
        while scan < hi:
            var first_start = Int(key_offsets[scan])
            var label = data[first_start + depth]
            var end = scan + 1
            while (
                end < hi
                and data[Int(key_offsets[end]) + depth] == label
            ):
                end += 1

            var last_start = Int(key_offsets[end - 1])
            var first_len = Int(key_offsets[scan + 1]) - first_start
            var last_len = Int(key_offsets[end]) - last_start
            var limit = min(first_len, last_len)
            var common = common_prefix_length(
                data, first_start, last_start, depth + 1, limit
            )
            var segment_len = common - depth
            segment_offsets[edge_count] = Int32(pool_size)
            segment_lengths[edge_count] = Int32(segment_len)
            copy_bytes(data, first_start + depth, pool + pool_size, segment_len)

            node_lo[node_count] = Int32(scan)
            node_hi[node_count] = Int32(end)
            node_depth[node_count] = Int32(common)
            terminal[node_count] = -1
            if first_len == common:
                terminal[node_count] = Int32(scan)
            node_count += 1
            edge_count += 1
            pool_size += segment_len
            scan = end
        node += 1
    offsets[node_count] = Int32(edge_count)
    stats[0] = Int64(node_count)
    stats[1] = Int64(pool_size)
    return 0


@export("mmtrie_simd_width")
def mmtrie_simd_width() abi("C") -> Int:
    return W


@export("mmtrie_batch_grain")
def mmtrie_batch_grain() abi("C") -> Int:
    return BATCH_GRAIN
