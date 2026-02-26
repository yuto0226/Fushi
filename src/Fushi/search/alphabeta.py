from __future__ import annotations

import sys
import time

import chess

from Fushi.evaluate import Evaluator

from . import InfoCallback, SearchInfo, SearchResult, Searcher, StopCondition
from .tt import NodeType, TranspositionTable, zobrist_hash

# this score indicates a forced mate
_MATE_THRESHOLD = 50_000

# Piece values used for MVV-LVA (Most Valuable Victim - Least Valuable Attacker) ordering
_MVV_LVA_VALUE: dict[int, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}

# Internal PV cons-type: (move, tail) or None
PV = tuple[chess.Move, "PV"] | None


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

    def _mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        """Return MVV-LVA score for a capture move (higher = search earlier)."""
        # Use piece_type_at to avoid allocating Piece objects repeatedly.
        victim_type = board.piece_type_at(move.to_square)
        if victim_type is None:
            # En passant: pawn captures pawn
            return _MVV_LVA_VALUE[chess.PAWN] * 10 - _MVV_LVA_VALUE[chess.PAWN]
        attacker_type = board.piece_type_at(move.from_square)
        attacker_value = _MVV_LVA_VALUE.get(attacker_type, 0) if attacker_type else 0
        return _MVV_LVA_VALUE[victim_type] * 10 - attacker_value

    def _order_moves(self, board: chess.Board, tt_entry: object) -> list[chess.Move]:
        """Return legal moves ordered: TT hash move first, captures by MVV-LVA, then quiet moves."""
        moves = list(board.legal_moves)

        hash_move: chess.Move | None = None
        if tt_entry is not None and tt_entry.best_move is not None:  # type: ignore[union-attr]
            hm = tt_entry.best_move  # type: ignore[union-attr]
            if hm in moves:
                hash_move = hm

        # Single-pass sort avoids allocating separate captures/quiets/result
        # lists and an MVV dict.  The sort key is called exactly once per move.
        def _key(move: chess.Move) -> int:
            if hash_move is not None and move == hash_move:
                return 30_000_000
            if board.is_capture(move):
                return 10_000_000 + self._mvv_lva_score(board, move)
            return 0

        moves.sort(key=_key, reverse=True)
        return moves

    def _qsearch(
        self,
        board: chess.Board,
        alpha: int,
        beta: int,
    ) -> int:
        """Quiescence search: keep searching captures until the position is quiet."""
        # Stand-pat: the side to move can choose not to capture
        stand_pat = self._relative_score(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Only search captures, ordered by MVV-LVA.
        # is_game_over() is deferred until we know there are no captures:
        # it is expensive (checks checkmate, stalemate, 50-move, repetition)
        # and calling it on every node is wasteful.
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        if not captures:
            # Terminal position (checkmate/stalemate) → return stand-pat.
            # Quiet position with no captures → stand-pat is already correct.
            if board.is_game_over():
                return stand_pat
            return alpha

        mvv = {m: self._mvv_lva_score(board, m) for m in captures}
        captures.sort(key=lambda m: mvv[m], reverse=True)

        for move in captures:
            # TODO: Delta Pruning — if stand_pat + captured_piece_value + margin < alpha, skip
            self.nodes += 1
            board.push(move)
            score = -self._qsearch(board, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    @staticmethod
    def _pv_to_list(pv_cons: PV) -> list[chess.Move]:
        """Flatten a cons-style PV chain into a plain list."""
        out: list[chess.Move] = []
        cur = pv_cons
        while cur is not None:
            out.append(cur[0])
            cur = cur[1]
        return out

    def _dfs(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        extensions: int = 0,
    ) -> tuple[int, PV]:
        original_alpha = alpha

        # Cache frequently accessed instance attributes as locals to avoid
        # repeated attribute lookups on every node in the hot recursive path.
        tt = self._tt
        stop_cond = self._stop_condition

        # probe TT
        key = zobrist_hash(board)
        tt_entry = tt.probe(key) if tt is not None else None

        if tt_entry is not None and tt_entry.depth >= depth:
            if tt_entry.node_type == NodeType.EXACT:
                pv_cons = (
                    (tt_entry.best_move, None)
                    if tt_entry.best_move is not None
                    else None
                )
                return tt_entry.score, pv_cons
            elif tt_entry.node_type == NodeType.LOWERBOUND:
                alpha = max(alpha, tt_entry.score)
            elif tt_entry.node_type == NodeType.UPPERBOUND:
                beta = min(beta, tt_entry.score)

            if alpha >= beta:
                pv_cons = (
                    (tt_entry.best_move, None)
                    if tt_entry.best_move is not None
                    else None
                )
                return tt_entry.score, pv_cons

        # leaf node: run quiescence search instead of bare static eval
        if depth == 0 or board.is_game_over():
            score = self._qsearch(board, alpha, beta)
            return score, None

        # TODO: Razoring — if depth == 1 and static_eval + RAZOR_MARGIN < alpha, return qsearch directly
        # TODO: Futility Pruning — if depth <= 2 and static_eval + FUTILITY_MARGIN[depth] <= alpha, skip quiet moves
        # TODO: Null Move Pruning — if not in_check and has non-pawn material, try null move at depth - R - 1

        # recurse
        best_score = -sys.maxsize
        best_pv = None

        # LMR: track whether we are currently in check before iterating moves.
        # Avoid reducing when the side to move must escape check.
        in_check = board.is_check()

        for move_idx, move in enumerate(self._order_moves(board, tt_entry)):
            if stop_cond is not None and stop_cond():
                break

            self.nodes += 1

            is_capture = board.is_capture(move)
            board.push(move)

            # check extension
            is_check_after = board.is_check()
            ext = 1 if is_check_after and extensions < self._max_extensions else 0

            # TODO: Futility Pruning (per-move) — skip quiet late moves when margin cannot raise alpha

            # Late Move Reductions (LMR)
            # Reduce depth for moves searched late in the list that are quiet,
            # non-checking, and made from a non-check position.
            # If the reduced search still beats alpha, re-search at full depth.
            reduction = 0
            if (
                move_idx >= 3
                and depth >= 3
                and not is_capture
                and not is_check_after  # don't reduce moves that give check
                and not in_check  # don't reduce when escaping check
                and ext == 0
            ):
                reduction = max(1, depth // 3)
                score, pv = self._dfs(
                    board,
                    depth - 1 + ext - reduction,
                    -alpha - 1,
                    -alpha,
                    extensions + ext,
                )
                score = -score
                # Reduced search improved alpha — verify with full-depth re-search
                if score > alpha:
                    score, pv = self._dfs(
                        board, depth - 1 + ext, -beta, -alpha, extensions + ext
                    )
                    score = -score
            else:
                score, pv = self._dfs(
                    board, depth - 1 + ext, -beta, -alpha, extensions + ext
                )
                score = -score  # reverse opponent's relative score

            board.pop()

            if score > best_score or not best_pv:
                best_score = score
                best_pv = (move, pv)

            if best_score > alpha:
                alpha = best_score

            # beta cutoff
            if alpha >= beta:
                break

        # store in TT
        if tt is not None and not (stop_cond is not None and stop_cond()):
            if best_score <= original_alpha:
                node_type = NodeType.UPPERBOUND
            elif best_score >= beta:
                node_type = NodeType.LOWERBOUND
            else:
                node_type = NodeType.EXACT

            tt.store(
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

        # Do not copy root board here; use push()/pop() for in-place search.

        last_result: SearchResult | None = None

        # only one move available — avoid traversing legal_moves twice
        root_moves = list(board.legal_moves)
        if len(root_moves) == 1:
            return SearchResult(
                best_move=root_moves[0], score=0, depth=0, nodes=0, pv=[root_moves[0]]
            )

        for depth in range(1, self._depth + 1):
            if self._should_stop():
                break

            # TODO: Aspiration Windows — search with a narrow window around prev_score;
            #       re-search with wider window on fail-low / fail-high
            score, pv = self._dfs(board, depth, -sys.maxsize, sys.maxsize)

            pv_list = self._pv_to_list(pv)

            if self._should_stop():
                break

            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000

            info = SearchInfo(
                depth=depth,
                score=score,
                nodes=self.nodes,
                time_ms=elapsed_ms,
                pv=pv_list,
            )
            if on_info:
                on_info(info)

            last_result = SearchResult(
                best_move=pv[0] if pv else None,
                score=score,
                depth=depth,
                nodes=self.nodes,
                pv=pv_list,
            )

            # forced mate confirmed
            if abs(score) >= _MATE_THRESHOLD:
                break

        if last_result is None:
            last_result = SearchResult(best_move=None, score=0, depth=0, nodes=0, pv=[])

        return last_result
