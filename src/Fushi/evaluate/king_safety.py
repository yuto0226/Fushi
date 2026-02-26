"""
King safety evaluator focused on castling incentives.

Scoring model (centipawns, from White's perspective):

  CASTLED_BONUS          King is on a known castled square (g1/c1 etc.)
  RIGHTS_BONUS           Each castling right still available
  UNCASTLED_PENALTY      King on e-file with *no* castling rights remaining
                         (lost castling rights without castling)

The UNCASTLED_PENALTY is intentionally large so the search strongly prefers
moves that lead to castling over moves that burn the rights.
"""

from __future__ import annotations

import chess

from . import Evaluator

CASTLED_BONUS: int = 60  # reward for king already on a castled square
RIGHTS_BONUS: int = 15  # reward per available castling right (max 30)
UNCASTLED_PENALTY: int = 50  # extra penalty when rights are gone but king exposed

# Squares that indicate White / Black have completed castling
_WHITE_CASTLED = frozenset({chess.G1, chess.C1})
_BLACK_CASTLED = frozenset({chess.G8, chess.C8})

# e-file squares (king sitting here in middlegame is dangerous)
_WHITE_KING_CENTER = chess.E1
_BLACK_KING_CENTER = chess.E8


class KingSafetyEvaluator(Evaluator):
    """
    Evaluate king safety with an emphasis on castling.

    Combine with material and PST evaluators via :class:`WeightedEvaluator`::

        WeightedEvaluator([
            (ShannonEvaluator(),    1.0),
            (PSTEvaluator(tables), 1.0),
            (KingSafetyEvaluator(), 1.0),
        ])

    Args:
        castled_bonus:    Centipawn bonus when king is on a castled square.
        rights_bonus:     Centipawn bonus per remaining castling right.
        uncastled_penalty: Extra centipawn penalty when castling rights are
                          gone but king is still on the e-file.
    """

    def __init__(
        self,
        castled_bonus: int = CASTLED_BONUS,
        rights_bonus: int = RIGHTS_BONUS,
        uncastled_penalty: int = UNCASTLED_PENALTY,
    ) -> None:
        super().__init__()
        self._castled_bonus = castled_bonus
        self._rights_bonus = rights_bonus
        self._uncastled_penalty = uncastled_penalty

    def evaluate(self, board: chess.Board) -> int:
        return self._score_color(board, chess.WHITE) - self._score_color(
            board, chess.BLACK
        )

    def _score_color(self, board: chess.Board, color: chess.Color) -> int:
        king_sq = board.king(color)
        if king_sq is None:
            return 0

        castled_squares = _WHITE_CASTLED if color == chess.WHITE else _BLACK_CASTLED
        center_sq = _WHITE_KING_CENTER if color == chess.WHITE else _BLACK_KING_CENTER

        score = 0

        if king_sq in castled_squares:
            score += self._castled_bonus

        if board.has_kingside_castling_rights(color):
            score += self._rights_bonus
        if board.has_queenside_castling_rights(color):
            score += self._rights_bonus

        # King is stuck on the e-file with rights gone → dangerous
        no_rights = not board.has_kingside_castling_rights(
            color
        ) and not board.has_queenside_castling_rights(color)
        if no_rights and king_sq == center_sq and king_sq not in castled_squares:
            score -= self._uncastled_penalty

        return score
