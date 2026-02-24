import sys
import time

import chess

from Fushi.evaluate import Evaluator

from . import InfoCallback, SearchInfo, SearchResult, Searcher


class BruteForceSearcher(Searcher):
    def __init__(self, evaluator: Evaluator, depth: int = 3):
        super().__init__()
        self._evaluator = evaluator
        self._depth = depth
        self.nodes = 0

    def _dfs(self, board: chess.Board, depth: int) -> tuple[int, list[chess.Move]]:
        if depth == 0 or board.is_game_over():
            score = self._evaluator.evaluate(board)
            # If it's Black's turn, negate the absolute score (White+, Black-)
            # to get the relative score (SideToMove+)
            if board.turn == chess.BLACK:
                score = -score
            return score, []

        best_score = -sys.maxsize
        best_pv: list[chess.Move] = []

        for move in board.legal_moves:
            self.nodes += 1

            board.push(move)

            score, pv = self._dfs(board, depth - 1)
            score = -score  # reverse opponent's score

            board.pop()

            if score > best_score or not best_pv:
                best_score = score
                best_pv = [move] + pv

        return best_score, best_pv

    def search(
        self,
        board: chess.Board,
        *,
        on_info: InfoCallback | None = None,
    ) -> SearchResult:
        start = time.monotonic_ns()
        self.nodes = 0

        board = board.copy()

        best_score, best_pv = self._dfs(board, self._depth)
        best_move = best_pv[0] if best_pv else None

        elapsed_ms = (time.monotonic_ns() - start) // 1_000_000

        info = SearchInfo(
            depth=self._depth,
            score=best_score,
            nodes=self.nodes,
            time_ms=elapsed_ms,
            pv=best_pv,
        )

        if on_info:
            on_info(info)

        return SearchResult(
            best_move=best_move,
            score=best_score,
            depth=self._depth,
            nodes=self.nodes,
            pv=best_pv,
        )
