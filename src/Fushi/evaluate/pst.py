"""
Piece-Square Table (PST) evaluator.

Tables are stored from White's visual perspective:
  index 0  = a8 (top-left when viewed from White's side)
  index 63 = h1 (bottom-right)

Lookup rules:
  White piece at square sq  →  table[sq ^ 56]   (flip rank)
  Black piece at square sq  →  table[sq]         (already mirrored)

Fushi MG/EG tables, derived from PeSTO with the following modifications:

  Bishop MG
  ---------
  - b4/b5 (pin bishop, Spanish/Nimzo):  5 → 20  (severely undervalued, caused
    Black to prefer passive Bd6 over the tactically correct Bb4 pin)
  - g4/g5 (active attacking bishop):    7 → 15  (undervalued vs passive squares)
  - f4/f5 (London System / active):    12 → 20  (key active development square)
  - d3/d6 (often a "bad bishop" spot): 15 →  8  (reduced to discourage bishop
    placement in front of own center pawns)
  - f3/f6 (knight's natural square):   27 → 14  (bishop PST was higher than
    knight PST for the same square, causing incorrect piece placement)

  King EG
  -------
  Replaced PeSTO's asymmetric king EG values with a clean radial gradient:
  corners = -80, edges = −60 / −40 / −30, center (d4-e5) = +40.
  Effect: the engine is strongly incentivised to centralise its own king while
  simultaneously pushing the opponent's king toward the corners/edges to set
  up mating nets.

References:
  PeSTO tables – https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function
"""

from __future__ import annotations

import chess

from . import Evaluator

# Maps each PieceType to a 64-element list of centipawn bonuses.
PieceSquareTables = dict[chess.PieceType, list[int]]

# fmt: off
_FUSHI_MG: PieceSquareTables = {
    chess.PAWN: [
          0,   0,   0,   0,   0,   0,   0,   0,
         98, 134,  61,  95,  68, 126,  34, -11,
         -6,   7,  26,  31,  65,  56,  25, -20,
        -14,  13,   6,  21,  23,  12,  17, -23,
        -27,  -2,  -5,  12,  17,   6,  10, -25,
        -26,  -4,  -4, -10,   3,   3,  33, -12,
        -35,  -1, -20, -23, -15,  24,  38, -22,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    chess.KNIGHT: [
        -167, -89, -34, -49,  61, -97, -15, -107,
         -73, -41,  72,  36,  23,  62,   7,  -17,
         -47,  60,  37,  65,  84, 129,  73,   44,
          -9,  17,  19,  53,  37,  69,  18,   22,
         -13,   4,  16,  13,  28,  19,  21,   -8,
         -23,  -9,  12,  10,  19,  17,  25,  -16,
         -29, -53, -12,  -3,  -1,  18, -14,  -19,
        -105, -21, -58, -33, -17, -28, -19,  -23,
    ],
    chess.BISHOP: [
        -29,   4, -82, -37, -25, -42,   7,  -8,
        -26,  16, -18, -13,  30,  59,  18, -47,
        -16,  37,  43,  40,  35,  50,  37,  -2,
         -4,  20,  19,  50,  37,  37,  15,  -2,  # b5/b4 pin: 5→20; g5/g4 active: 7→15
         -6,  13,  13,  26,  34,  20,  10,   4,  # f4/f5 London/active: 12→20
          0,  15,  15,   8,  14,  14,  18,  10,  # d3/d6 passive: 15→8; f3/f6 knight sq: 27→14
          4,  15,  16,   0,   7,  21,  33,   1,
        -33,  -3, -14, -21, -13, -12, -39, -21,
    ],
    chess.ROOK: [
         32,  42,  32,  51,  63,   9,  31,  43,
         27,  32,  58,  62,  80,  67,  26,  44,
         -5,  19,  26,  36,  17,  45,  61,  16,
        -24, -11,   7,  26,  24,  35,  -8, -20,
        -36, -26, -12,  -1,   9,  -7,   6, -23,
        -45, -25, -16, -17,   3,   0,  -5, -33,
        -44, -16, -20,  -9,  -1,  11,  -6, -71,
        -19, -13,   1,  17,  16,   7, -37, -26,
    ],
    chess.QUEEN: [
        -28,   0,  29,  12,  59,  44,  43,  45,
        -24, -39,  -5,   1, -16,  57,  28,  54,
        -13, -17,   7,   8,  29,  56,  47,  57,
        -27, -27, -16, -16,  -1,  17,  -2,   1,
         -9, -26,  -9, -10,  -2,  -4,   3,  -3,
        -14,   2, -11,  -2,  -5,   2,  14,   5,
        -35,  -8,  11,   2,   8,  15,  -3,   1,
         -1, -18,  -9,  10, -15, -25, -31, -50,
    ],
    chess.KING: [
        -65,  23,  16, -15, -56, -34,   2,  13,
         29,  -1, -20,  -7,  -8,  -4, -38, -29,
         -9,  24,   2, -16, -20,   6,  22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49,  -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
          1,   7,  -8, -64, -43, -16,   9,   8,
        -15,  36,  12, -54,   8, -28,  24,  14,
    ],
}

_FUSHI_EG: PieceSquareTables = {
    chess.PAWN: [
          0,   0,   0,   0,   0,   0,   0,   0,
        178, 173, 158, 134, 147, 132, 165, 187,
         94, 100,  85,  67,  56,  53,  82,  84,
         32,  24,  13,   5,  -2,   4,  17,  17,
         13,   9,  -3,  -7,  -7,  -8,   3,  -1,
          4,   7,  -6,   1,   0,  -5,  -1,  -8,
         13,   8,   8,  10,  13,   0,   2,  -7,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    chess.KNIGHT: [
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25,  -8, -25,  -2,  -9, -25, -24, -52,
        -24, -20,  10,   9,  -1,  -9, -19, -41,
        -17,   3,  22,  22,  22,  11,   8, -18,
        -18,  -6,  16,  25,  16,  17,   4, -18,
        -23,  -3,  -1,  15,  10,  -3, -20, -22,
        -42, -20, -10,  -5,  -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64,
    ],
    chess.BISHOP: [
        -14, -21, -11,  -8,  -7,  -9, -17, -24,
         -8,  -4,   7, -12,  -3, -13,  -4, -14,
          2,  -8,   0,  -1,  -2,   6,   0,   4,
         -3,   9,  12,   9,  14,  10,   3,   2,
         -6,   3,  13,  19,   7,  10,  -3,  -9,
        -12,  -3,   8,  10,  13,   3,  -7, -15,
        -14, -18,  -7,  -1,   4,  -9, -15, -27,
        -23,  -9, -23,  -5,  -9, -16,  -5, -17,
    ],
    chess.ROOK: [
         13,  10,  18,  15,  12,  12,   8,   5,
         11,  13,  13,  11,  -3,   3,   8,   3,
          7,   7,   7,   5,   4,  -3,  -5,  -3,
          4,   3,  13,   1,   2,   1,  -1,   2,
          3,   5,   8,   4,  -5,  -6,  -8, -11,
         -4,   0,  -5,  -1,  -7, -12,  -8, -16,
         -6,  -6,   0,   2,  -9,  -9, -11,  -3,
         -9,   2,   3,  -1,  -5, -13,   4, -20,
    ],
    chess.QUEEN: [
         -9,  22,  22,  27,  27,  19,  10,  20,
        -17,  20,  32,  41,  58,  25,  30,   0,
        -20,   6,   9,  49,  47,  35,  19,   9,
          3,  22,  24,  45,  57,  40,  57,  36,
        -18,  28,  19,  47,  31,  34,  39,  23,
        -16, -27,  15,   6,   9,  17,  10,   5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43,  -5, -32, -20, -41,
    ],
    chess.KING: [
        # Radial gradient: corners = -80, edges taper to 0, centre = +40.
        # A king forced to the corner suffers a large penalty; a centralised
        # king earns a large bonus.  The same table is used for both sides
        # (via the sq^56 flip), so the opponent is equally punished for
        # retreating to the corner while our king marches to the centre.
        -80, -60, -40, -30, -30, -40, -60, -80,  # rank 8
        -60, -30,   0,  10,  10,   0, -30, -60,  # rank 7
        -40,   0,  20,  30,  30,  20,   0, -40,  # rank 6
        -30,  10,  30,  40,  40,  30,  10, -30,  # rank 5
        -30,  10,  30,  40,  40,  30,  10, -30,  # rank 4
        -40,   0,  20,  30,  30,  20,   0, -40,  # rank 3
        -60, -30,   0,  10,  10,   0, -30, -60,  # rank 2
        -80, -60, -40, -30, -30, -40, -60, -80,  # rank 1
    ],
}

# fmt: on

#: Fushi middlegame tables.
FUSHI_MG_TABLES: PieceSquareTables = _FUSHI_MG

#: Fushi endgame tables – used together with FUSHI_MG_TABLES via PhaseEvaluator.
FUSHI_EG_TABLES: PieceSquareTables = _FUSHI_EG

#: Alias for a sensible default single-phase table set.
CLASSICAL_TABLES: PieceSquareTables = _FUSHI_MG

# Backwards-compatible aliases (deprecated, prefer FUSHI_* names).
PESTO_MG_TABLES = FUSHI_MG_TABLES
PESTO_EG_TABLES = FUSHI_EG_TABLES

_PHASE_WEIGHTS: dict[chess.PieceType, int] = {
    chess.QUEEN: 4,
    chess.ROOK: 2,
    chess.BISHOP: 1,
    chess.KNIGHT: 1,
}
_MAX_PHASE = 24  # 2 queens * 4 + 4 rooks * 2 + 4 bishops * 1 + 4 knights * 1


def _game_phase(board: chess.Board) -> int:
    """Return a phase value in [0, 24].  24 = full middlegame, 0 = endgame."""
    phase = 0
    for piece_type, weight in _PHASE_WEIGHTS.items():
        phase += len(board.pieces(piece_type, chess.WHITE)) * weight
        phase += len(board.pieces(piece_type, chess.BLACK)) * weight
    return min(phase, _MAX_PHASE)


class PSTEvaluator(Evaluator):
    """
    Evaluate a position using a set of piece-square tables.

    The score is purely positional (no material count); combine with
    :class:`ShannonEvaluator` or another material evaluator via
    :class:`WeightedEvaluator`.

    Args:
        tables: A mapping from :data:`chess.PieceType` to a 64-element list
                of centipawn bonuses.  The list is indexed from *White's*
                visual perspective: index 0 = a8, index 63 = h1.
    """

    def __init__(self, tables: PieceSquareTables) -> None:
        super().__init__()
        self._tables = tables

    def evaluate(self, board: chess.Board) -> int:
        score = 0
        for piece_type, table in self._tables.items():
            for sq in board.pieces(piece_type, chess.WHITE):
                score += table[sq ^ 56]
            for sq in board.pieces(piece_type, chess.BLACK):
                score -= table[sq]
        return score
