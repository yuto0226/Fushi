from abc import ABC, abstractmethod

import chess

PIECE_VALUES = {
    chess.KING: 10000,
    chess.QUEEN: 1000,
    chess.ROOK: 525,
    chess.BISHOP: 350,
    chess.KNIGHT: 350,
    chess.PAWN: 100,
}


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, board: chess.Board) -> int:
        """
        Evaluate the board from an absolute perspective.
        Score > 0 means White is winning.
        Score < 0 means Black is winning.
        """
        ...


from .shannon import ShannonEvaluator as ShannonEvaluator  # noqa: E402


class WeightedEvaluator(Evaluator):
    def __init__(self, components: list[tuple[Evaluator, float]]) -> None:
        if not components:
            raise ValueError("At least one evaluator is required")
        self._components = components

    def evaluate(self, board: chess.Board) -> int:
        return round(sum(e.evaluate(board) * w for e, w in self._components))


# TODO: PhaseEvaluator
