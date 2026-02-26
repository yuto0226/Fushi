import sys
import threading
import time
from typing import Callable

import chess

from .search import Searcher

ENGINE_NAME = "Fushi"
ENGINE_AUTHOR = "Yuto"
VERSION = "0.0.1"

BestMoveCallback = Callable[[chess.Move | None, chess.Move | None], None]
"""
Arguments: (best_move, ponder_move)
  best_move   — move the engine wants to play
  ponder_move — expected opponent reply (pv[1]), or None
"""


class Engine:
    def __init__(self, searcher: Searcher, debug=False):
        self._name = ENGINE_NAME
        self._author = ENGINE_AUTHOR
        self._is_ready = True
        self._stop_event = threading.Event()
        self.debug = debug

        self.board = chess.Board()
        self.searcher = searcher
        self._search_thread: threading.Thread | None = None

        # pondering state, only written inside go() and _run()'s finally
        self._is_pondering: bool = False
        self._timer: threading.Timer | None = None

        # time-management state (written by go()/_run(), read by ponderhit/on_info)
        self._soft_ms: int | None = None
        self._hard_ms: int | None = None
        self._search_start_ns: int = 0

        # search option
        self.ponder = False
        self.wtime: int | None = None  # ms
        self.btime: int | None = None  # ms
        self.winc: int = 0  # ms
        self.binc: int = 0  # ms
        self.movestogo: int | None = None
        self.movetime: int | None = None  # ms, fixed time for this move
        self.depth: int = sys.maxsize
        self.nodes: int = sys.maxsize
        self.mate: int = sys.maxsize
        self.infinite: bool = False

    def name(self):
        return self._name

    def author(self):
        return self._author

    def is_debug(self):
        return self.debug

    def is_ready(self):
        return self._is_ready

    def stop(self):
        """Stop searching asap.

        Works identically whether we are in SEARCHING or PONDERING:
        _stop_event fires → search returns → _run()'s finally calls on_bestmove.
        """
        self._stop_event.set()

    def stopping(self) -> bool:
        return self._stop_event.is_set()

    def ponderhit(self) -> None:
        """Opponent played the predicted move.

        Transitions PONDERING → SEARCHING by arming the time limit.
        The search thread keeps running — the TT is already warm.
        The clock reference is reset to now so the allocated soft/hard
        budget is measured from the moment we start thinking for real.
        """
        if not self._is_pondering:
            return
        self._is_pondering = False
        self._search_start_ns = time.monotonic_ns()
        self._arm_timer(self._hard_ms)

    def reset(self):
        self._stop_event.set()
        if self._search_thread and self._search_thread.is_alive():
            self._search_thread.join()
        self._is_pondering = False
        self.board.reset()

    def set_position(self, fen: str = "", moves: list[str] = []) -> None:
        if fen == "startpos":
            self.board.reset()
        elif fen:
            self.board.set_fen(fen)

        if moves:
            for move_uci in moves:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                except ValueError:
                    continue

    def _compute_time_budget(self) -> tuple[int | None, int | None]:
        """Return (soft_ms, hard_ms).

        soft_ms — optimal time; the engine stops starting a new iteration once
                  this is exceeded (checked at iteration boundaries).
        hard_ms — absolute cap; enforced via a threading.Timer that fires
                  _stop_event, so the search is aborted mid-iteration if
                  necessary.

        Formula
        -------
        Allocate  base / moves_left + inc * 0.8  as the soft target, where
        moves_left shrinks as the game progresses (so later moves get a bit
        more time, which mirrors human behaviour in the endgame).
        The hard limit is min(soft * 3, base * 0.5) so a single move can never
        consume more than half the remaining clock.
        """
        if self.infinite:
            return None, None

        if self.movetime is not None:
            return self.movetime, self.movetime

        base = self.wtime if self.board.turn == chess.WHITE else self.btime
        inc = self.winc if self.board.turn == chess.WHITE else self.binc

        if base is None:
            return None, None

        # Estimate remaining moves in the game.
        if self.movestogo is not None and self.movestogo > 0:
            # Cyclic time control: we know exactly how many moves are left.
            # Add a small buffer so we don't empty the clock on the last move.
            moves_left = self.movestogo + 2
        else:
            # Sudden-death heuristic: assumption is the game lasts ~80 plies
            # total, so remaining plies ≈ 80 - current_ply, clamped to [10, 50].
            ply = (self.board.fullmove_number - 1) * 2 + (
                0 if self.board.turn == chess.WHITE else 1
            )
            moves_left = max(10, min(50, 50 - ply // 3))

        alloc = base / moves_left + inc * 0.8
        soft_ms = int(alloc)
        hard_ms = int(min(soft_ms * 3.0, base * 0.5))
        hard_ms = max(hard_ms, soft_ms)  # hard must be >= soft

        return soft_ms, hard_ms

    def go(self, on_bestmove: BestMoveCallback) -> None:
        """
        Start searching in a background thread.
        Calls on_bestmove(best_move, ponder_move) when done.

        If self.ponder is True, searches indefinitely until ponderhit() or stop()
        """
        # abort any previous search
        if self._search_thread and self._search_thread.is_alive():
            self._stop_event.set()
            self._search_thread.join()

        self._stop_event.clear()
        self._timer = None
        self._is_pondering = self.ponder  # PONDERING if ponder flag set

        def _run():
            soft_ms, hard_ms = self._compute_time_budget()
            self._soft_ms = soft_ms
            self._hard_ms = hard_ms
            self._search_start_ns = time.monotonic_ns()

            # Arm the hard timer now for normal search; pondering waits for ponderhit().
            if not self._is_pondering:
                self._arm_timer(hard_ms)

            # ----------------------------------------------------------------
            # Soft-bound logic is wired through the on_info callback, which
            # fires after every completed depth in the iterative deepening loop.
            # This way we never cut a search mid-iteration.
            # ----------------------------------------------------------------
            _soft_ns = soft_ms * 1_000_000 if soft_ms is not None else None

            # Per-iteration history for easy-move detection:
            #   list of (best_move, score) for the last few completed depths.
            _history: list[tuple[object, int]] = []

            def on_info(info):
                print(info.to_uci(), flush=True)

                if _soft_ns is None or self._is_pondering:
                    return  # no time limit or still pondering — keep going

                best_move = info.pv[0] if info.pv else None
                _history.append((best_move, info.score))

                elapsed_ns = time.monotonic_ns() - self._search_start_ns

                # Easy-move detection:
                # If the best move hasn't changed for the last STABLE_DEPTHS
                # iterations AND the score has been stable (< SCORE_WINDOW cp),
                # accept the result early at EASY_SCALE of the soft bound.
                _STABLE_DEPTHS = 3
                _SCORE_WINDOW = 20  # centipawns
                _EASY_SCALE = 0.5

                if len(_history) >= _STABLE_DEPTHS:
                    recent = _history[-_STABLE_DEPTHS:]
                    moves_stable = len({m for m, _ in recent}) == 1
                    score_range = max(s for _, s in recent) - min(s for _, s in recent)
                    if moves_stable and score_range <= _SCORE_WINDOW:
                        if elapsed_ns >= _soft_ns * _EASY_SCALE:
                            self._stop_event.set()
                            return

                # Normal soft bound: stop after current iteration if time is up.
                if elapsed_ns >= _soft_ns:
                    self._stop_event.set()

            try:
                result = self.searcher.search(
                    self.board,
                    on_info=on_info,
                    stop_condition=self._stop_event.is_set,
                )
                ponder_move = result.pv[1] if len(result.pv) >= 2 else None
                on_bestmove(result.best_move, ponder_move)
            finally:
                if self._timer is not None:
                    self._timer.cancel()
                self._is_pondering = False

        self._search_thread = threading.Thread(target=_run, daemon=True)
        self._search_thread.start()

    def _arm_timer(self, ms: int | None) -> None:
        """Start a one-shot timer that fires _stop_event after *ms* milliseconds."""
        if ms is None:
            return
        self._timer = threading.Timer(ms / 1000.0, self._stop_event.set)
        self._timer.daemon = True
        self._timer.start()
