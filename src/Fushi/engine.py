import sys
import threading
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
        """
        if not self._is_pondering:
            return
        self._is_pondering = False
        self._arm_timer()

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

    def _compute_move_time(self) -> int | None:
        """base / 20 + increment / 2"""
        if self.infinite:
            return None

        if self.movetime is not None:
            return self.movetime

        base = self.wtime if self.board.turn == chess.WHITE else self.btime
        inc = self.winc if self.board.turn == chess.WHITE else self.binc

        if base is None:
            return None

        return base // 20 + inc // 2

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
            # Arm the timer now for normal search; pondering waits for ponderhit().
            if not self._is_pondering:
                self._arm_timer()

            def on_info(info):
                print(info.to_uci(), flush=True)

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

    def _arm_timer(self) -> None:
        """Start a one-shot timer that fires _stop_event after the computed budget."""
        ms = self._compute_move_time()
        if ms is None:
            return
        self._timer = threading.Timer(ms / 1000.0, self._stop_event.set)
        self._timer.daemon = True
        self._timer.start()
