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
from .pst import (  # noqa: E402
    PSTEvaluator as PSTEvaluator,
    CLASSICAL_TABLES as CLASSICAL_TABLES,
    FUSHI_MG_TABLES as FUSHI_MG_TABLES,
    FUSHI_EG_TABLES as FUSHI_EG_TABLES,
    PESTO_MG_TABLES as PESTO_MG_TABLES,  # deprecated alias
    PESTO_EG_TABLES as PESTO_EG_TABLES,  # deprecated alias
)
from .king_safety import KingSafetyEvaluator as KingSafetyEvaluator  # noqa: E402


class WeightedEvaluator(Evaluator):
    def __init__(self, components: list[tuple[Evaluator, float]]) -> None:
        if not components:
            raise ValueError("At least one evaluator is required")
        self._components = components

    def evaluate(self, board: chess.Board) -> int:
        return round(sum(e.evaluate(board) * w for e, w in self._components))


class PhaseEvaluator(Evaluator):
    """
    Tapered evaluator that interpolates between a middlegame and an endgame
    evaluator based on the remaining material on the board.

    The game phase is derived from the count of major/minor pieces:
      phase = 4 * queens + 2 * rooks + 1 * bishops + 1 * knights
    clamped to [0, 24] (24 = full middlegame, 0 = pure endgame).

    Final score:
      (mg_score * phase + eg_score * (24 - phase)) / 24

    Typical usage::

        from Fushi.evaluate import (
            PhaseEvaluator, WeightedEvaluator,
            ShannonEvaluator, PSTEvaluator,
            FUSHI_MG_TABLES, FUSHI_EG_TABLES,
        )

        eval = PhaseEvaluator(
            mg=WeightedEvaluator([
                (ShannonEvaluator(), 1.0),
                (PSTEvaluator(FUSHI_MG_TABLES), 1.0),
            ]),
            eg=WeightedEvaluator([
                (ShannonEvaluator(), 1.0),
                (PSTEvaluator(FUSHI_EG_TABLES), 1.0),
            ]),
        )
    """

    _PHASE_WEIGHTS: dict[chess.PieceType, int] = {
        chess.QUEEN: 4,
        chess.ROOK: 2,
        chess.BISHOP: 1,
        chess.KNIGHT: 1,
    }
    _MAX_PHASE = 24

    def __init__(self, mg: Evaluator, eg: Evaluator) -> None:
        self._mg = mg
        self._eg = eg

    def evaluate(self, board: chess.Board) -> int:
        phase = self._compute_phase(board)
        mg_score = self._mg.evaluate(board)
        eg_score = self._eg.evaluate(board)
        return round(
            (mg_score * phase + eg_score * (self._MAX_PHASE - phase)) / self._MAX_PHASE
        )

    def _compute_phase(self, board: chess.Board) -> int:
        phase = 0
        for piece_type, weight in self._PHASE_WEIGHTS.items():
            phase += len(board.pieces(piece_type, chess.WHITE)) * weight
            phase += len(board.pieces(piece_type, chess.BLACK)) * weight
        return min(phase, self._MAX_PHASE)
