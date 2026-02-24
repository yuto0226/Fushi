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
    def evaluate(self, board: chess.Board) -> int: ...


from .shannon import ShannonEvaluator as ShannonEvaluator  # noqa: E402
