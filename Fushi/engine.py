import chess

from .search import Searcher

ENGINE_NAME = "Fushi"
ENGINE_AUTHOR = "Yuto"
VERSION = "0.0.1"


class Engine:
    def __init__(self, searcher: Searcher, debug=False):
        self._name = ENGINE_NAME
        self._author = ENGINE_AUTHOR
        self._is_ready = True
        self._stop = False
        self.debug = debug

        self.board = chess.Board()
        self.searcher = searcher

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
        self._stop = True

    def stopping(self) -> bool:
        return self._stop

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

    def search(self) -> chess.Move | None:
        self._stop = False

        def on_info(info):
            print(info.to_uci())

        result = self.searcher.search(self.board, on_info=on_info)
        return result.best_move
