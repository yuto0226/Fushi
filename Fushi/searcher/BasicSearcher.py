import sys
import time

import chess

from Fushi.evaluator import Evaluator

from . import InfoCallback, SearchInfo, SearchResult, Searcher


class BasicSearcher(Searcher):
    def __init__(self, evaluator: Evaluator):
        super().__init__()
        self._evaluator = evaluator

    def search(
        self,
        board: chess.Board,
        *,
        on_info: InfoCallback | None = None,
    ) -> SearchResult:
        start = time.monotonic_ns()
        nodes = 0

        best_move: chess.Move | None = None
        best_score = -sys.maxsize

        board = board.copy()

        for move in board.legal_moves:
            nodes += 1
            board.push(move)
            score = -self._evaluator.evaluate(board)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        elapsed_ms = (time.monotonic_ns() - start) // 1_000_000

        info = SearchInfo(
            depth=1,
            score=best_score,
            nodes=nodes,
            time_ms=elapsed_ms,
            pv=[best_move] if best_move else [],
        )

        if on_info:
            on_info(info)

        return SearchResult(
            best_move=best_move,
            score=best_score,
            depth=1,
            nodes=nodes,
            pv=[best_move] if best_move else [],
        )
