import sys
import time

import chess

from Fushi.evaluate import Evaluator

from . import InfoCallback, SearchInfo, SearchResult, Searcher, StopCondition
from .tt import NodeType, TranspositionTable, zobrist_hash


class MinMaxSearcher(Searcher):
    def __init__(
        self,
        evaluator: Evaluator,
        depth: int = 15,
        tt: TranspositionTable | None = None,
    ):
        super().__init__(tt=tt)
        self._evaluator = evaluator
        self._depth = depth
        self.nodes = 0
        self._stop_condition: StopCondition | None = None

    def _should_stop(self) -> bool:
        return self._stop_condition is not None and self._stop_condition()

    def _relative_score(self, board: chess.Board) -> int:
        """absolute score (White+, Black-) to relative score (SideToMove+)"""
        score = self._evaluator.evaluate(board)
        return score if board.turn == chess.WHITE else -score

    def _order_moves(self, board: chess.Board, tt_entry: object) -> list[chess.Move]:
        """Return legal moves with the TT hash move sorted first, if available."""
        moves = list(board.legal_moves)

        hash_move: chess.Move | None = None
        if tt_entry is not None and tt_entry.best_move is not None:  # type: ignore[union-attr]
            if tt_entry.best_move in board.legal_moves:  # type: ignore[union-attr]
                hash_move = tt_entry.best_move  # type: ignore[union-attr]

        if hash_move is not None:
            moves.remove(hash_move)
            moves.insert(0, hash_move)

        return moves

    def _dfs(self, board: chess.Board, depth: int) -> tuple[int, list[chess.Move]]:
        # probe
        key = zobrist_hash(board)
        tt_entry = self._tt.probe(key) if self._tt is not None else None

        if tt_entry is not None:
            if tt_entry.depth >= depth and tt_entry.node_type == NodeType.EXACT:
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

            score, pv = self._dfs(board, depth - 1)
            score = -score  # reverse opponent's score

            board.pop()

            if score > best_score or not best_pv:
                best_score = score
                best_pv = [move] + pv

        # store
        if self._tt is not None and not self._should_stop():
            self._tt.store(
                key,
                depth,
                best_score,
                NodeType.EXACT,
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

        for depth in range(1, self._depth + 1):
            if self._should_stop():
                break

            score, pv = self._dfs(board, depth)

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

        if last_result is None:
            last_result = SearchResult(best_move=None, score=0, depth=0, nodes=0, pv=[])

        return last_result
