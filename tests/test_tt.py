import chess

from Fushi.search.tt import NodeType, TranspositionTable


def _make_key(n: int) -> int:
    """Return a synthetic 64-bit Zobrist-like key."""
    return n & 0xFFFF_FFFF_FFFF_FFFF


def _store(
    tt: TranspositionTable,
    key: int,
    depth: int = 4,
    score: int = 100,
    node_type: NodeType = NodeType.EXACT,
    best_move: chess.Move | None = None,
) -> None:
    tt.store(key, depth, score, node_type, best_move)


class TestNodeType:
    def test_values(self):
        assert NodeType.EXACT == 0
        assert NodeType.LOWERBOUND == 1
        assert NodeType.UPPERBOUND == 2

    def test_ordering(self):
        assert NodeType.EXACT < NodeType.LOWERBOUND < NodeType.UPPERBOUND


class TestInit:
    def test_num_entries_is_power_of_two(self):
        tt = TranspositionTable(size_mb=1)
        n = tt.num_entries
        assert n >= 1
        assert (n & (n - 1)) == 0, f"{n} is not a power of two"

    def test_mask_consistent_with_num_entries(self):
        tt = TranspositionTable(size_mb=1)
        assert tt._mask == tt.num_entries - 1

    def test_initial_counters_zero(self):
        tt = TranspositionTable(size_mb=1)
        assert tt.hits == 0
        assert tt.stores == 0

    def test_size_mb_property(self):
        tt = TranspositionTable(size_mb=1)
        assert 0 < tt.size_mb <= 2

    def test_repr_contains_num_entries(self):
        tt = TranspositionTable(size_mb=1)
        assert str(tt.num_entries) in repr(tt)


class TestStoreProbe:
    def setup_method(self):
        self.tt = TranspositionTable(size_mb=1)

    def test_probe_miss_on_empty_table(self):
        assert self.tt.probe(_make_key(1)) is None

    def test_store_then_probe_returns_entry(self):
        key = _make_key(42)
        _store(self.tt, key, depth=5, score=200, node_type=NodeType.EXACT)
        entry = self.tt.probe(key)
        assert entry is not None
        assert entry.key == key
        assert entry.depth == 5
        assert entry.score == 200
        assert entry.node_type == NodeType.EXACT

    def test_probe_increments_hits(self):
        key = _make_key(7)
        _store(self.tt, key)
        assert self.tt.hits == 0
        self.tt.probe(key)
        assert self.tt.hits == 1
        self.tt.probe(key)
        assert self.tt.hits == 2

    def test_store_increments_stores(self):
        assert self.tt.stores == 0
        _store(self.tt, _make_key(1))
        assert self.tt.stores == 1
        _store(self.tt, _make_key(2))
        assert self.tt.stores == 2

    def test_best_move_stored_and_returned(self):
        key = _make_key(99)
        move = chess.Move.from_uci("e2e4")
        _store(self.tt, key, best_move=move)
        entry = self.tt.probe(key)
        assert entry is not None
        assert entry.best_move == move

    def test_best_move_none_stored_correctly(self):
        key = _make_key(55)
        _store(self.tt, key, best_move=None)
        entry = self.tt.probe(key)
        assert entry is not None
        assert entry.best_move is None


class TestCollisionDetection:
    def test_different_key_same_slot_returns_none(self):
        tt = TranspositionTable(size_mb=1)
        mask = tt._mask

        # Construct two keys that hash to the same slot but are different.
        key_a = _make_key(0)
        key_b = key_a ^ (mask + 1)  # differs in high bits, same low bits
        assert (key_a & mask) == (key_b & mask), "keys must alias to same slot"
        assert key_a != key_b

        _store(tt, key_a, depth=3, score=50)
        # Probing with key_b should miss because full key doesn't match.
        assert tt.probe(key_b) is None

    def test_same_key_returns_entry(self):
        tt = TranspositionTable(size_mb=1)
        key = _make_key(123)
        _store(tt, key, depth=3, score=50)
        assert tt.probe(key) is not None


class TestReplacementStrategy:
    def setup_method(self):
        self.tt = TranspositionTable(size_mb=1)

    def _alias_key(self, base_key: int) -> int:
        """Return a key that shares the same slot as *base_key* but is different."""
        mask = self.tt._mask
        return base_key ^ (mask + 1)

    def test_empty_slot_always_accepts(self):
        key = _make_key(10)
        _store(self.tt, key, depth=1, score=10)
        assert self.tt.probe(key) is not None

    def test_shallower_same_age_does_not_replace(self):
        """A new entry with depth < existing depth should be dropped."""
        key = _make_key(20)
        _store(self.tt, key, depth=6, score=100)
        # Try to overwrite with a shallower search.
        _store(self.tt, key, depth=3, score=999)
        entry = self.tt.probe(key)
        assert entry is not None
        assert entry.depth == 6  # original deeper entry survives
        assert entry.score == 100

    def test_equal_depth_same_age_replaces(self):
        """depth >= existing.depth → replace (boundary condition)."""
        key = _make_key(21)
        _store(self.tt, key, depth=5, score=111)
        _store(self.tt, key, depth=5, score=222)
        entry = self.tt.probe(key)
        assert entry is not None
        assert entry.score == 222

    def test_deeper_same_age_replaces(self):
        """A deeper search should overwrite a shallower entry."""
        key = _make_key(30)
        _store(self.tt, key, depth=3, score=50)
        _store(self.tt, key, depth=7, score=75)
        entry = self.tt.probe(key)
        assert entry is not None
        assert entry.depth == 7
        assert entry.score == 75

    def test_stale_entry_replaced_by_shallower_new_search(self):
        """After new_search(), existing entries are stale → replaced freely."""
        key = _make_key(40)
        _store(self.tt, key, depth=8, score=300)
        self.tt.new_search()
        # Shallow entry from new search should evict the stale deep one.
        _store(self.tt, key, depth=1, score=999)
        entry = self.tt.probe(key)
        assert entry is not None
        assert entry.score == 999
        assert entry.depth == 1

    def test_alias_key_same_slot_replaced_by_stale_logic(self):
        """Stale entry at a slot is replaced even when it is a different key."""
        key_a = _make_key(50)
        key_b = self._alias_key(key_a)
        _store(self.tt, key_a, depth=8, score=500)
        self.tt.new_search()
        _store(self.tt, key_b, depth=1, score=111)
        # slot now holds key_b
        assert self.tt.probe(key_b) is not None
        assert self.tt.probe(key_a) is None  # key_a evicted


class TestClear:
    def test_clear_removes_all_entries(self):
        tt = TranspositionTable(size_mb=1)
        for i in range(10):
            _store(tt, _make_key(i))
        tt.clear()
        for i in range(10):
            assert tt.probe(_make_key(i)) is None

    def test_clear_resets_counters(self):
        tt = TranspositionTable(size_mb=1)
        _store(tt, _make_key(1))
        tt.probe(_make_key(1))
        tt.clear()
        assert tt.hits == 0
        assert tt.stores == 0

    def test_clear_resets_age(self):
        tt = TranspositionTable(size_mb=1)
        tt.new_search()
        tt.new_search()
        assert tt._age == 2
        tt.clear()
        assert tt._age == 0

    def test_store_works_after_clear(self):
        tt = TranspositionTable(size_mb=1)
        _store(tt, _make_key(5))
        tt.clear()
        _store(tt, _make_key(5), score=42)
        entry = tt.probe(_make_key(5))
        assert entry is not None
        assert entry.score == 42


class TestNewSearch:
    def test_increments_age(self):
        tt = TranspositionTable(size_mb=1)
        assert tt._age == 0
        tt.new_search()
        assert tt._age == 1
        tt.new_search()
        assert tt._age == 2

    def test_resets_hits_and_stores(self):
        tt = TranspositionTable(size_mb=1)
        _store(tt, _make_key(1))
        tt.probe(_make_key(1))
        # hits == 1, stores == 1
        tt.new_search()
        assert tt.hits == 0
        assert tt.stores == 0

    def test_old_entries_still_probeable(self):
        """new_search() should NOT wipe the table — stale entries are still accessible."""
        tt = TranspositionTable(size_mb=1)
        key = _make_key(99)
        _store(tt, key, depth=5, score=123)
        tt.new_search()
        entry = tt.probe(key)
        assert entry is not None
        assert entry.score == 123

    def test_age_stored_in_entry(self):
        tt = TranspositionTable(size_mb=1)
        tt.new_search()  # age → 1
        key = _make_key(77)
        _store(tt, key)
        entry = tt.probe(key)
        assert entry is not None
        assert entry.age == 1
