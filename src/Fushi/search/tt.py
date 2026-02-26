"""
Transposition Table (TT) implementation using Zobrist hashing.

Reference: https://www.chessprogramming.org/Transposition_Table

The TT caches previously evaluated positions to avoid redundant computation,
especially valuable in iterative deepening where shallower iterations can seed
deeper iterations with useful bounds and best-move hints.

Entry types (NodeType):
  EXACT      — PV-node: exact score returned from a full minimax search.
  LOWERBOUND — Cut-node: score is a lower bound (failed high, beta cutoff).
  UPPERBOUND — All-node: score is an upper bound (failed low, alpha cutoff).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import chess
import chess.polyglot


class NodeType(enum.IntEnum):
    """Type of transposition table entry, reflecting alpha-beta search node type."""

    EXACT = 0
    """The stored score is exact (PV-node). Safe to use directly."""

    LOWERBOUND = 1
    """Score is a lower bound (beta-cutoff/Cut-node). Usable when score >= beta."""

    UPPERBOUND = 2
    """Score is an upper bound (all-node). Usable when score <= alpha."""


@dataclass(slots=True)
class TTEntry:
    """A single entry in the transposition table."""

    key: int
    """Full 64-bit Zobrist key for collision detection."""

    depth: int
    """Remaining search depth when this entry was stored."""

    score: int
    """Evaluated score (centipawns), relative to side to move."""

    node_type: NodeType
    """Whether the score is exact, a lower bound, or an upper bound."""

    best_move: chess.Move | None
    """Best move found at this node (hash move); used for move ordering."""

    age: int
    """Search age when stored; stale entries may be replaced even if deeper."""


def zobrist_hash(board: chess.Board) -> int:
    return chess.polyglot.zobrist_hash(board)


class TranspositionTable:
    """
    Fixed-size transposition table backed by a flat list.

    Addressing: ``key & mask`` maps a 64-bit Zobrist key to a slot.
    Collision detection: the full key is stored and checked on probe.

    Replacement strategy: depth-preferred with aging.
    A new entry replaces an existing one when any of the following holds:
      - The slot is empty.
      - The existing entry is from a previous search (age < current age).
      - The new entry was searched to at least as great a depth.
    """

    # Conservative estimate of bytes per Python TTEntry object (slots=True helps).
    _BYTES_PER_ENTRY: int = 160

    def __init__(self, size_mb: int = 16) -> None:
        """
        Initialise the table.

        Args:
            size_mb: Approximate memory budget in megabytes.
        """
        capacity = (size_mb * 1024 * 1024) // self._BYTES_PER_ENTRY

        # Round down to the largest power of two that fits in the budget.
        num_entries = 1
        while num_entries * 2 <= capacity:
            num_entries *= 2

        self._num_entries: int = num_entries
        self._mask: int = num_entries - 1
        self._table: list[TTEntry | None] = [None] * num_entries
        self._age: int = 0

        # Diagnostic counters (reset each search via new_search).
        self.probes: int = 0
        self.hits: int = 0
        self.exact_hits: int = 0
        self.lowerbound_hits: int = 0
        self.upperbound_hits: int = 0
        self.stores: int = 0

    def probe(self, key: int) -> TTEntry | None:
        """
        Look up *key* in the table.

        Returns the matching :class:`TTEntry` on a hit, or ``None`` on a miss.
        A *hit* requires that the stored full key equals *key* (type-2 collision
        check).
        """
        self.probes += 1
        entry = self._table[key & self._mask]
        if entry is not None and entry.key == key:
            self.hits += 1
            if entry.node_type == NodeType.EXACT:
                self.exact_hits += 1
            elif entry.node_type == NodeType.LOWERBOUND:
                self.lowerbound_hits += 1
            else:
                self.upperbound_hits += 1
            return entry
        return None

    def store(
        self,
        key: int,
        depth: int,
        score: int,
        node_type: NodeType,
        best_move: chess.Move | None = None,
    ) -> None:
        """
        Store a search result in the table.

        The entry may be silently dropped if replacement strategy decides
        that the existing deeper entry from the current search is more valuable.

        Args:
            key:       64-bit Zobrist key of the position.
            depth:     Remaining depth at which the position was searched.
            score:     Score relative to the side to move.
            node_type: EXACT, LOWERBOUND, or UPPERBOUND.
            best_move: Best (or refutation) move, if known.
        """
        idx = key & self._mask
        existing = self._table[idx]

        # Replacement conditions (depth-preferred + aging):
        if (
            existing is None
            or existing.age < self._age  # stale entry → replace freely
            or depth >= existing.depth  # deeper search → replace
        ):
            self._table[idx] = TTEntry(
                key=key,
                depth=depth,
                score=score,
                node_type=node_type,
                best_move=best_move,
                age=self._age,
            )
            self.stores += 1

    def clear(self) -> None:
        """Wipe all entries and reset age / counters."""
        self._table = [None] * self._num_entries
        self._age = 0
        self.probes = 0
        self.hits = 0
        self.exact_hits = 0
        self.lowerbound_hits = 0
        self.upperbound_hits = 0
        self.stores = 0

    def new_search(self) -> None:
        """
        Signal the start of a new root search.

        Increments the age counter so that entries from the previous root
        position are considered stale and can be replaced by shallower entries
        from the new search.  This is deliberately *not* a full clear — old
        entries may still be probed and used until they are naturally evicted.
        """
        self._age += 1
        self.probes = 0
        self.hits = 0
        self.exact_hits = 0
        self.lowerbound_hits = 0
        self.upperbound_hits = 0
        self.stores = 0

    @property
    def num_entries(self) -> int:
        """Total number of slots in the table."""
        return self._num_entries

    @property
    def size_mb(self) -> float:
        """Approximate memory usage in megabytes."""
        return self._num_entries * self._BYTES_PER_ENTRY / (1024 * 1024)

    def __repr__(self) -> str:
        return (
            f"TranspositionTable(num_entries={self._num_entries}, "
            f"size≈{self.size_mb:.1f} MB, age={self._age})"
        )
