import sys
import time

import chess

from Fushi.evaluate import PIECE_VALUES, Evaluator

from . import InfoCallback, SearchInfo, SearchResult, Searcher, StopCondition
from .tt import NodeType, TranspositionTable, zobrist_hash

# this score indicates a forced mate
_MATE_THRESHOLD = 50_000


class AlphaBetaSearcher(Searcher):
    def __init__(
        self,
        evaluator: Evaluator,
        depth: int = 15,
        tt: TranspositionTable | None = None,
        max_extensions: int = 5,
    ):
        super().__init__(tt=tt)
        self._evaluator = evaluator
        self._depth = depth
        self._max_extensions = max_extensions
        self.nodes = 0
        self._stop_condition: StopCondition | None = None

    def _should_stop(self) -> bool:
        return self._stop_condition is not None and self._stop_condition()

    def _relative_score(self, board: chess.Board) -> int:
        """absolute score (White+, Black-) to relative score (SideToMove+)"""
        score = self._evaluator.evaluate(board)
        return score if board.turn == chess.WHITE else -score

    def _score_move(
        self, board: chess.Board, move: chess.Move, hash_move: chess.Move | None
    ) -> int:
        """score a move for order moves sorting"""
        if move == hash_move:
            return 1_000_000

        score = 0

        if board.is_capture(move):
            attacker_piece = board.piece_type_at(move.from_square)

            if board.is_en_passant(move):
                victim_piece = chess.PAWN
            else:
                victim_piece = board.piece_type_at(move.to_square)

            if victim_piece is not None and attacker_piece is not None:
                score += (
                    10000
                    + (PIECE_VALUES[victim_piece] * 10)
                    - PIECE_VALUES[attacker_piece]
                )

        if move.promotion:
            score += PIECE_VALUES[move.promotion] * 1000

        return score

    def _order_moves(self, board: chess.Board, tt_entry: object) -> list[chess.Move]:
        """Return legal moves with the TT hash move sorted first, if available."""
        moves = list(board.legal_moves)

        hash_move: chess.Move | None = None
        if tt_entry is not None and tt_entry.best_move is not None:  # type: ignore[union-attr]
            if tt_entry.best_move in board.legal_moves:  # type: ignore[union-attr]
                hash_move = tt_entry.best_move  # type: ignore[union-attr]

        moves.sort(key=lambda m: self._score_move(board, m, hash_move), reverse=True)

        return moves

    def _dfs(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        extensions: int = 0,
    ) -> tuple[int, list[chess.Move]]:
        original_alpha = alpha

        # probe TT
        key = zobrist_hash(board)
        tt_entry = self._tt.probe(key) if self._tt is not None else None

        if tt_entry is not None and tt_entry.depth >= depth:
            if tt_entry.node_type == NodeType.EXACT:
                pv = [tt_entry.best_move] if tt_entry.best_move is not None else []
                return tt_entry.score, pv
            elif tt_entry.node_type == NodeType.LOWERBOUND:
                alpha = max(alpha, tt_entry.score)
            elif tt_entry.node_type == NodeType.UPPERBOUND:
                beta = min(beta, tt_entry.score)

            if alpha >= beta:
                pv = [tt_entry.best_move] if tt_entry.best_move is not None else []
                return tt_entry.score, pv

        # leaf node
        if depth == 0 or board.is_game_over():
            score = self._relative_score(board)
            if self._tt is not None:
                self._tt.store(key, depth, score, NodeType.EXACT)
            return score, []

        # recurse
        best_score = -sys.maxsize
        best_pv: list[chess.Move] = []

        for move in self._order_moves(board, tt_entry):
            if self._should_stop():
                break

            self.nodes += 1

            board.push(move)

            # check extension
            is_check = board.is_check()
            ext = 1 if is_check and extensions < self._max_extensions else 0

            score, pv = self._dfs(
                board, depth - 1 + ext, -beta, -alpha, extensions + ext
            )
            score = -score  # reverse opponent's relative score

            board.pop()

            if score > best_score or not best_pv:
                best_score = score
                best_pv = [move] + pv

            if best_score > alpha:
                alpha = best_score

            # beta cutoff
            if alpha >= beta:
                break

        # store in TT
        if self._tt is not None and not self._should_stop():
            if best_score <= original_alpha:
                node_type = NodeType.UPPERBOUND
            elif best_score >= beta:
                node_type = NodeType.LOWERBOUND
            else:
                node_type = NodeType.EXACT

            self._tt.store(
                key,
                depth,
                best_score,
                node_type,
                best_pv[0] if best_pv else None,
            )

        return best_score, best_pv

    def search(
        self,
        board: chess.Board,
        *,
        on_info: InfoCallback | None = None,
        stop_condition: StopCondition | None = None,
    ) -> SearchResult:
        start = time.monotonic_ns()
        self.nodes = 0
        self._stop_condition = stop_condition

        if self._tt is not None:
            self._tt.new_search()

        board = board.copy()

        last_result: SearchResult | None = None

        # only one move available
        if board.legal_moves.count() == 1:
            only_move = next(iter(board.legal_moves))
            return SearchResult(
                best_move=only_move, score=0, depth=0, nodes=0, pv=[only_move]
            )

        for depth in range(1, self._depth + 1):
            if self._should_stop():
                break

            score, pv = self._dfs(board, depth, -sys.maxsize, sys.maxsize)

            if self._should_stop():
                break

            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000

            info = SearchInfo(
                depth=depth,
                score=score,
                nodes=self.nodes,
                time_ms=elapsed_ms,
                pv=pv,
            )
            if on_info:
                on_info(info)

            last_result = SearchResult(
                best_move=pv[0] if pv else None,
                score=score,
                depth=depth,
                nodes=self.nodes,
                pv=pv,
            )

            # forced mate confirmed
            if abs(score) >= _MATE_THRESHOLD:
                break

        if last_result is None:
            last_result = SearchResult(best_move=None, score=0, depth=0, nodes=0, pv=[])

        return last_result
