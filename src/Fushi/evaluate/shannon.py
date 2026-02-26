import sys

import chess

from . import Evaluator, PIECE_VALUES


class ShannonEvaluator(Evaluator):
    def __init__(self):
        super().__init__()

    def evaluate(self, board: chess.Board) -> int:
        """
        f(p) = 200(K-K')
            + 9(Q-Q')
            + 5(R-R')
            + 3(B-B' + N-N')
            + 1(P-P')
            - 0.5(D-D' + S-S' + I-I')
            + 0.1(M-M') + ...

        KQRBNP = number of kings, queens, rooks, bishops, knights and pawns
        D,S,I = doubled, blocked and isolated pawns
        M = Mobility (the number of legal moves)

        Returns:
            Absolute score (positive for White, negative for Black).
        """
        if board.is_checkmate():
            return -sys.maxsize if board.turn == chess.WHITE else sys.maxsize

        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0.0
        score += self._evaluate_material(board)
        score -= 0.5 * self._evaluate_pawn_structure(board)
        score += 0.1 * self._evaluate_mobility(board)

        return round(score)

    def _evaluate_material(self, board: chess.Board) -> float:
        score = 0.0

        for piece_type in [
            chess.KING,
            chess.QUEEN,
            chess.ROOK,
            chess.BISHOP,
            chess.KNIGHT,
            chess.PAWN,
        ]:
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))

            score += PIECE_VALUES[piece_type] * (white_count - black_count)

        return score

    def _evaluate_pawn_structure(self, board: chess.Board) -> float:
        white_penalty = self._count_pawn_weaknesses(board, chess.WHITE)
        black_penalty = self._count_pawn_weaknesses(board, chess.BLACK)

        return white_penalty - black_penalty

    def _count_pawn_weaknesses(self, board: chess.Board, color: chess.Color) -> int:
        pawns = board.pieces(chess.PAWN, color)
        weakness = 0

        files = [0] * 8
        for square in pawns:
            files[chess.square_file(square)] += 1

        doubled = sum(max(0, count - 1) for count in files)
        weakness += doubled

        isolated = 0
        for file_idx in range(8):
            if files[file_idx] > 0:
                left_has_pawn = file_idx > 0 and files[file_idx - 1] > 0
                right_has_pawn = file_idx < 7 and files[file_idx + 1] > 0
                if not left_has_pawn and not right_has_pawn:
                    isolated += files[file_idx]
        weakness += isolated

        blocked = 0
        opponent = not color
        for square in pawns:
            if color == chess.WHITE:
                front_square = square + 8
            else:
                front_square = square - 8

            if 0 <= front_square < 64:
                blocker = board.piece_at(front_square)
                # Shannon's definition: a pawn is *blocked* when an *enemy*
                # piece occupies the square directly in front of it.
                # Own pieces forming a pawn chain are not a weakness.
                if blocker is not None and blocker.color == opponent:
                    blocked += 1
        weakness += blocked

        return weakness

    def _evaluate_mobility(self, board: chess.Board) -> float:
        # A null move is semantically illegal when in check (you cannot pass
        # while your king is attacked).  python-chess won't raise but it would
        # measure an artificially compressed mobility for the side in check,
        # skewing the score.  Return neutral when the position is checked.
        if board.is_check():
            return 0.0

        if board.turn == chess.WHITE:
            white_mobility = board.legal_moves.count()
            board.push(chess.Move.null())
            black_mobility = board.legal_moves.count()
            board.pop()
        else:
            black_mobility = board.legal_moves.count()
            board.push(chess.Move.null())
            white_mobility = board.legal_moves.count()
            board.pop()

        return float(white_mobility - black_mobility)
