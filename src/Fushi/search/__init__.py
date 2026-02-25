from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import chess

if TYPE_CHECKING:
    from .tt import TranspositionTable

StopCondition = Callable[[], bool]


@dataclass
class SearchInfo:
    depth: int
    score: int  # centipawns
    nodes: int
    time_ms: int
    pv: list[chess.Move] = field(default_factory=list)
    seldepth: int | None = None  # quiescence deepest depth

    def to_uci(self) -> str:
        pv_str = " ".join(m.uci() for m in self.pv)
        parts = [
            f"depth {self.depth}",
            *([] if self.seldepth is None else [f"seldepth {self.seldepth}"]),
            f"score cp {self.score}",
            f"nodes {self.nodes}",
            f"time {self.time_ms}",
            *([] if not self.pv else [f"pv {pv_str}"]),
        ]
        return "info " + " ".join(parts)


@dataclass
class SearchResult:
    best_move: chess.Move | None
    score: int
    depth: int
    nodes: int
    pv: list[chess.Move] = field(default_factory=list)


InfoCallback = Callable[[SearchInfo], None]


class Searcher(ABC):
    def __init__(self, tt: TranspositionTable | None = None) -> None:
        self._tt: TranspositionTable | None = tt

    @abstractmethod
    def search(
        self,
        board: chess.Board,
        *,
        on_info: InfoCallback | None = None,
        stop_condition: StopCondition | None = None,
    ) -> SearchResult:
        """
        Search for the best move.
        Returns a SearchResult where the score is relative to the side to move at the root.
        """
        ...


from .alphabeta import AlphaBetaSearcher as AlphaBetaSearcher  # noqa: E402
from .basic import BasicSearcher as BasicSearcher  # noqa: E402
from .book import BookSearcher as BookSearcher  # noqa: E402
from .dfs import BruteForceSearcher as BruteForceSearcher  # noqa: E402
from .minmax import MinMaxSearcher as MinMaxSearcher  # noqa: E402
from .tt import NodeType as NodeType  # noqa: E402
from .tt import TTEntry as TTEntry  # noqa: E402
from .tt import TranspositionTable as TranspositionTable  # noqa: E402
from .tt import zobrist_hash as zobrist_hash  # noqa: E402
