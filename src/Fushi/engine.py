import sys
import threading
from typing import Callable

import chess

from .search import Searcher

ENGINE_NAME = "Fushi"
ENGINE_AUTHOR = "Yuto"
VERSION = "0.0.1"

BestMoveCallback = Callable[[chess.Move | None], None]


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
        """stop searching asap"""
        self._stop_event.set()

    def stopping(self) -> bool:
        return self._stop_event.is_set()

    def reset(self):
        pass

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

    def go(self, on_bestmove: BestMoveCallback) -> None:
        """
        Start searching in a background thread.
        Calls on_bestmove(move) when done.
        """
        # abort any previous search
        if self._search_thread and self._search_thread.is_alive():
            self._stop_event.set()
            self._search_thread.join()

        self._stop_event.clear()

        def _run():
            timer: threading.Timer | None = None

            if self.movetime is not None:
                # UCI movetime is MS, Timer needs SECONDS
                timer = threading.Timer(self.movetime / 1000.0, self._stop_event.set)
                timer.daemon = True
                timer.start()

            def on_info(info):
                print(info.to_uci(), flush=True)

            try:
                result = self.searcher.search(
                    self.board,
                    on_info=on_info,
                    stop_condition=self._stop_event.is_set,
                )
                on_bestmove(result.best_move)
            finally:
                if timer is not None:
                    timer.cancel()

        self._search_thread = threading.Thread(target=_run, daemon=True)
        self._search_thread.start()
