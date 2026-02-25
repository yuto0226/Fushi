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
        print("option name UCI_Ponder type check default false")
        print("uciok")

    def _cmd_debug(self, args: list[str]) -> None:
        """debug [on|off]"""
        if args[0] == "on":
            self._engine.debug = True
        elif args[0] == "off":
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
        self._engine.reset()

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
        """
        go [searchmoves <move>...] [ponder] [wtime <ms>] [btime <ms>]
           [winc <ms>] [binc <ms>] [movestogo <n>] [depth <n>] [nodes <n>]
           [mate <n>] [movetime <ms>] [infinite]
        """
        # reset search options to defaults
        import sys as _sys

        self._engine.ponder = False
        self._engine.wtime = None
        self._engine.btime = None
        self._engine.winc = 0
        self._engine.binc = 0
        self._engine.movestogo = None
        self._engine.movetime = None
        self._engine.depth = _sys.maxsize
        self._engine.nodes = _sys.maxsize
        self._engine.mate = _sys.maxsize
        self._engine.infinite = False

        i = 0
        while i < len(args):
            token = args[i]
            if token == "ponder":
                self._engine.ponder = True
            elif token == "infinite":
                self._engine.infinite = True
            elif token in (
                "wtime",
                "btime",
                "winc",
                "binc",
                "movestogo",
                "movetime",
                "depth",
                "nodes",
                "mate",
            ):
                if i + 1 < len(args):
                    try:
                        setattr(self._engine, token, int(args[i + 1]))
                    except ValueError:
                        pass
                    i += 1
            i += 1

        def on_bestmove(best_move, ponder_move):
            tokens = [f"bestmove {best_move.uci() if best_move else '(none)'}"]
            if ponder_move is not None:
                tokens.append(f"ponder {ponder_move.uci()}")
            print(" ".join(tokens), flush=True)

        self._engine.go(on_bestmove=on_bestmove)

    def _cmd_stop(self, _) -> None:
        """finish search asap"""
        self._engine.stop()
        pass

    def _cmd_ponderhit(self, _) -> None:
        """ppponent played the predicted move"""
        self._engine.ponderhit()

    def _cmd_quit(self, _) -> None:
        """quit asap"""
        raise SystemExit
