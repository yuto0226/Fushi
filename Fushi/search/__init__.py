from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import chess


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
    @abstractmethod
    def search(
        self,
        board: chess.Board,
        *,
        on_info: InfoCallback | None = None,
    ) -> SearchResult:
        """
        Search for the best move.
        Returns a SearchResult where the score is relative to the side to move at the root.
        """
        ...


from .basic import BasicSearcher as BasicSearcher  # noqa: E402
