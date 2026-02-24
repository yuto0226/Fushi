import sys

from Fushi.engine import Engine


class UCIHandler:
    # public
    def __init__(self, engine: Engine):
        self._engine = engine
        self._commands = {
            "uci": self._cmd_uci,
            "debug": self._cmd_debug,
            "isready": self._cmd_isready,
            "setoption": self._cmd_setoption,
            "register": self._cmd_register,
            "ucinewgame": self._cmd_ucinewgame,
            "position": self._cmd_position,
            "go": self._cmd_go,
            "stop": self._cmd_stop,
            "ponderhit": self._cmd_ponderhit,
            "quit": self._cmd_quit,
        }

    def run(self) -> None:
        for line in sys.stdin:
            parts = line.strip().split()
            if not parts:
                continue

            cmd, *args = parts
            if handler := self._commands.get(cmd):
                handler(args)

    # protected
    def _cmd_uci(self, _) -> None:
        print(f"id name {self._engine.name()}")
        print(f"id author {self._engine.author()}")

        print("uciok")
        pass

    def _cmd_debug(self, args: list[str]) -> None:
        """debug [on|off]"""
        if args[0] == "on":
            self._engine.debug = True
        elif args[1] == "off":
            self._engine.debug = False

    def _cmd_isready(self, _) -> None:
        """
        some operation may consume time to finish
        e.g. tables constructing
        """
        if self._engine.is_ready():
            print("readyok")

    def _cmd_setoption(self, args: list[str]) -> None:
        pass

    def _cmd_register(self, args: list[str]) -> None:
        """
        register [later|name <username> code <serial number>]

        for licensed engine, unsupport
        """
        print("uciok")
        pass

    def _cmd_ucinewgame(self, _) -> None:
        """create new game"""
        pass

    def _cmd_position(self, args: list[str]) -> None:
        """position [fen <fenstring>|starpos] moves <move_0> .... <move_i>"""
        fen = "startpos"
        moves = []
        if "fen" in args:
            idx = args.index("fen")
            moves_idx = args.index("moves") if "moves" in args else len(args)
            fen = " ".join(args[idx + 1 : moves_idx])
            moves = args[moves_idx + 1 :]
        elif "moves" in args:
            moves = args[args.index("moves") + 1 :]
        self._engine.set_position(fen, moves)
        pass

    def _cmd_go(self, args: list[str]) -> None:
        """search for best move"""
        move = self._engine.search()
        print(f"bestmove {move}")

    def _cmd_stop(self, _) -> None:
        """finish search asap"""
        self._engine.stop()
        pass

    def _cmd_ponderhit(self, _) -> None:
        pass

    def _cmd_quit(self, _) -> None:
        """quit asap"""
        raise SystemExit
