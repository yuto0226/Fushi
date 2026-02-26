from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import chess
import chess.polyglot

from . import BookReader


class PolyglotBookReader(BookReader):
    """
    Opening book backed by one or more Polyglot ``.bin`` files.

    Accepts a single path, a list of paths, or use :meth:`from_dir` to
    automatically discover every ``.bin`` file under a directory.

    When multiple files are loaded, entries for the same move are merged
    by summing their weights, so books with broader coverage naturally
    complement each other.

    Parameters
    ----------
    paths:
        One path or a list of paths to ``.bin`` files.
    weight_random:
        When ``True`` (default) a move is drawn probabilistically
        proportional to its combined weight, mimicking human-like variety.
        When ``False`` the move with the highest total weight is always
        chosen (deterministic / strongest).
    """

    def __init__(
        self,
        paths: str | Path | Sequence[str | Path],
        *,
        weight_random: bool = True,
    ) -> None:
        if isinstance(paths, (str, Path)):
            self._paths = [Path(paths)]
        else:
            self._paths = [Path(p) for p in paths]
        self._weight_random = weight_random

    @classmethod
    def from_dir(
        cls,
        directory: str | Path,
        *,
        weight_random: bool = True,
    ) -> "PolyglotBookReader":
        """
        Create a reader from **all** ``.bin`` files found recursively
        under *directory*.

        Raises ``FileNotFoundError`` if the directory does not exist.
        Returns an instance with an empty path list (always miss) if no
        ``.bin`` files are present.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Book directory not found: {directory}")
        bins = sorted(directory.rglob("*.bin"))
        return cls(bins, weight_random=weight_random)

    def probe(self, board: chess.Board) -> chess.Move | None:
        """Return a book move for *board*, or ``None`` if not in any book."""
        if not self._paths:
            return None

        # Aggregate entries from all books: merge weights for the same move.
        combined: dict[chess.Move, int] = {}
        for path in self._paths:
            try:
                with chess.polyglot.open_reader(path) as reader:
                    for entry in reader.find_all(board):
                        combined[entry.move] = (
                            combined.get(entry.move, 0) + entry.weight
                        )
            except (OSError, KeyError):
                continue

        # Filter to legal moves only (guards against castling encoding edge cases).
        legal = {m: w for m, w in combined.items() if m in board.legal_moves}
        if not legal:
            return None

        if self._weight_random:
            return self._weighted_choice(legal)
        else:
            return max(legal, key=lambda m: legal[m])

    @staticmethod
    def _weighted_choice(weighted: dict[chess.Move, int]) -> chess.Move:
        total = sum(weighted.values())
        if total == 0:
            return random.choice(list(weighted.keys()))  # uniform fallback
        r = random.randrange(total)
        cumulative = 0
        for move, weight in weighted.items():
            cumulative += weight
            if r < cumulative:
                return move
        return next(reversed(weighted))  # unreachable, but type-safe
