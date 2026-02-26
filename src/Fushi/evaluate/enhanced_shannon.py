"""
Enhanced Shannon Evaluator — extends :class:`ShannonEvaluator` with four
additional heuristics:

1. **Passed pawns** – rank-scaled bonus that grows as the pawn approaches
   promotion.  A pawn on the 7th rank earns ~120 cp; one just out of its
   starting square earns nothing yet.

2. **Pawn shield** – rewards intact pawn cover in front of a castled king
   (g1/c1 for White, g8/c8 for Black).  Penalises open and semi-open files
   aimed at the castled king.

3. **Bishop pair** – flat bonus when a side retains both bishops.  Two bishops
   outperform bishop+knight or two knights in open positions.

4. **Rook activity** – bonuses for rooks on open / semi-open files and for
   rooks that have reached the opponent's 7th rank.

All values are centipawns from White's absolute perspective, consistent with
the base :class:`ShannonEvaluator` contract.
"""

from __future__ import annotations

import chess

from .shannon import ShannonEvaluator

# ---------------------------------------------------------------------------
# Tuning constants (centipawns)
# ---------------------------------------------------------------------------

# Passed pawn bonus indexed by the pawn's *relative rank* (own perspective).
# rel_rank = actual_rank for White; 7 − actual_rank for Black.
# Index 0 / 1 are unused (no pawn can stand on its own back-rank/1st rank).
_PASSED_PAWN_BONUS: list[int] = [0, 0, 10, 20, 30, 50, 80, 120]

# Pawn shield — score per file in the king's vicinity
_PAWN_SHIELD_NEAR: int = 15  # pawn on the 2nd rank in front of king
_PAWN_SHIELD_FAR: int = 8  # pawn pushed to the 3rd rank (weaker)
_PAWN_SHIELD_OPEN_PENALTY: int = 25  # open file toward the king
_PAWN_SHIELD_SEMI_OPEN_PENALTY: int = 15  # semi-open file toward the king

# Bishop pair
_BISHOP_PAIR_BONUS: int = 30

# Rook activity
_ROOK_OPEN_FILE: int = 25
_ROOK_SEMI_OPEN_FILE: int = 15
_ROOK_SEVENTH_RANK: int = 25


class EnhancedShannonEvaluator(ShannonEvaluator):
    """Shannon evaluator augmented with passed-pawn, pawn-shield,
    bishop-pair, and rook-activity terms.

    Drop-in replacement for :class:`ShannonEvaluator`::

        from Fushi.evaluate import EnhancedShannonEvaluator
        evaluator = EnhancedShannonEvaluator()
    """

    def evaluate(self, board: chess.Board) -> int:
        base = super().evaluate(board)
        bonus = (
            self._evaluate_passed_pawns(board)
            + self._evaluate_pawn_shield(board)
            + self._evaluate_bishop_pair(board)
            + self._evaluate_rooks(board)
        )
        return base + bonus

    # ------------------------------------------------------------------
    # 1. Passed pawns
    # ------------------------------------------------------------------

    def _evaluate_passed_pawns(self, board: chess.Board) -> int:
        return self._passed_pawn_score(board, chess.WHITE) - self._passed_pawn_score(
            board, chess.BLACK
        )

    def _passed_pawn_score(self, board: chess.Board, color: chess.Color) -> int:
        score = 0
        enemy_pawns = board.pieces(chess.PAWN, not color)
        for sq in board.pieces(chess.PAWN, color):
            if self._is_passed(sq, color, enemy_pawns):
                rank = chess.square_rank(sq)
                rel_rank = rank if color == chess.WHITE else 7 - rank
                score += _PASSED_PAWN_BONUS[rel_rank]
        return score

    @staticmethod
    def _is_passed(sq: int, color: chess.Color, enemy_pawns: chess.SquareSet) -> bool:
        """Return True if the pawn on *sq* is a passed pawn for *color*."""
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        # Check own file and both adjacent files for enemy pawns *ahead*.
        for ep_sq in enemy_pawns:
            ep_file = chess.square_file(ep_sq)
            if abs(ep_file - file) > 1:
                continue
            ep_rank = chess.square_rank(ep_sq)
            # "Ahead" means a higher rank for White, lower rank for Black.
            if color == chess.WHITE and ep_rank > rank:
                return False
            if color == chess.BLACK and ep_rank < rank:
                return False
        return True

    # ------------------------------------------------------------------
    # 2. Pawn shield
    # ------------------------------------------------------------------

    def _evaluate_pawn_shield(self, board: chess.Board) -> int:
        return self._pawn_shield_score(board, chess.WHITE) - self._pawn_shield_score(
            board, chess.BLACK
        )

    def _pawn_shield_score(self, board: chess.Board, color: chess.Color) -> int:
        king_sq = board.king(color)
        if king_sq is None:
            return 0

        king_rank = chess.square_rank(king_sq)
        king_file = chess.square_file(king_sq)
        back_rank = 0 if color == chess.WHITE else 7

        # Shield is only relevant when the king sits on the back rank and is
        # on a flank (not d/e file where the king is in the centre).
        if king_rank != back_rank:
            return 0
        if king_file in (3, 4):  # d- or e-file — no shield expected
            return 0

        direction = 1 if color == chess.WHITE else -1
        score = 0

        for f in range(max(0, king_file - 1), min(8, king_file + 2)):
            near_sq = chess.square(f, back_rank + direction)  # 2nd rank
            far_sq = chess.square(f, back_rank + 2 * direction)  # 3rd rank

            near_piece = board.piece_at(near_sq)
            far_piece = board.piece_at(far_sq)

            if (
                near_piece is not None
                and near_piece.piece_type == chess.PAWN
                and near_piece.color == color
            ):
                score += _PAWN_SHIELD_NEAR  # intact shield pawn
            elif (
                far_piece is not None
                and far_piece.piece_type == chess.PAWN
                and far_piece.color == color
            ):
                score += _PAWN_SHIELD_FAR  # pushed one step, weaker cover
            else:
                # No own pawn directly ahead — evaluate how open the file is.
                file_mask = chess.BB_FILES[f]
                own_on_file = bool(board.pieces(chess.PAWN, color) & file_mask)
                enemy_on_file = bool(board.pieces(chess.PAWN, not color) & file_mask)
                if not own_on_file:
                    if not enemy_on_file:
                        score -= _PAWN_SHIELD_OPEN_PENALTY  # fully open file
                    else:
                        score -= _PAWN_SHIELD_SEMI_OPEN_PENALTY  # semi-open file

        return score

    # ------------------------------------------------------------------
    # 3. Bishop pair
    # ------------------------------------------------------------------

    def _evaluate_bishop_pair(self, board: chess.Board) -> int:
        score = 0
        if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
            score += _BISHOP_PAIR_BONUS
        if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
            score -= _BISHOP_PAIR_BONUS
        return score

    # ------------------------------------------------------------------
    # 4. Rook activity
    # ------------------------------------------------------------------

    def _evaluate_rooks(self, board: chess.Board) -> int:
        return self._rook_score(board, chess.WHITE) - self._rook_score(
            board, chess.BLACK
        )

    def _rook_score(self, board: chess.Board, color: chess.Color) -> int:
        score = 0
        # Rank index for the opponent's 7th rank (index 6 for White, 1 for Black).
        seventh_rank = 6 if color == chess.WHITE else 1

        for sq in board.pieces(chess.ROOK, color):
            file = chess.square_file(sq)
            file_mask = chess.BB_FILES[file]
            own_pawns = bool(board.pieces(chess.PAWN, color) & file_mask)
            enemy_pawns = bool(board.pieces(chess.PAWN, not color) & file_mask)

            if not own_pawns:
                if not enemy_pawns:
                    score += _ROOK_OPEN_FILE  # fully open file
                else:
                    score += _ROOK_SEMI_OPEN_FILE  # semi-open file

            if chess.square_rank(sq) == seventh_rank:
                score += _ROOK_SEVENTH_RANK  # dominant 7th-rank rook

        return score
