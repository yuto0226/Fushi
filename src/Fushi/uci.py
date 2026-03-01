import sys

from Fushi.engine import Engine


_BENCH_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
    "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
    "rq3rk1/ppp2ppp/1bnpb3/3N2B1/3NP3/7P/PPPQ1PP1/2KR3R w - - 7 14 moves d4e6",
    "r1bq1r1k/1pp1n1pp/1p1p4/4p2Q/4Pp2/1BNP4/PPP2PPP/3R1RK1 w - - 2 14 moves g2g4",
    "r3r1k1/2p2ppp/p1p1bn2/8/1q2P3/2NPQN2/PPP3PP/R4RK1 b - - 2 15",
    "r1bbk1nr/pp3p1p/2n5/1N4p1/2Np1B2/8/PPP2PPP/2KR1B1R w kq - 0 13",
    "r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 1 16",
    "4r1k1/r1q2ppp/ppp2n2/4P3/5Rb1/1N1BQ3/PPP3PP/R5K1 w - - 1 17",
    "2rqkb1r/ppp2p2/2npb1p1/1N1Nn2p/2P1PP2/8/PP2B1PP/R1BQK2R b KQ - 0 11",
    "r1bq1r1k/b1p1npp1/p2p3p/1p6/3PP3/1B2NN2/PP3PPP/R2Q1RK1 w - - 1 16",
    "3r1rk1/p5pp/bpp1pp2/8/q1PP1P2/b3P3/P2NQRPP/1R2B1K1 b - - 6 22",
    "r1q2rk1/2p1bppp/2Pp4/p6b/Q1PNp3/4B3/PP1R1PPP/2K4R w - - 2 18",
    "4k2r/1pb2ppp/1p2p3/1R1p4/3P4/2r1PN2/P4PPP/1R4K1 b - - 3 22",
    "3q2k1/pb3p1p/4pbp1/2r5/PpN2N2/1P2P2P/5PP1/Q2R2K1 b - - 4 26",
    "6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1",
    "3b4/5kp1/1p1p1p1p/pP1PpP1P/P1P1P3/3KN3/8/8 w - - 0 1",
    "2K5/p7/7P/5pR1/8/5k2/r7/8 w - - 0 1 moves g5g6 f3e3 g6g5 e3f3",
    "8/6pk/1p6/8/PP3p1p/5P2/4KP1q/3Q4 w - - 0 1",
    "7k/3p2pp/4q3/8/4Q3/5Kp1/P6b/8 w - - 0 1",
    "8/2p5/8/2kPKp1p/2p4P/2P5/3P4/8 w - - 0 1",
    "8/1p3pp1/7p/5P1P/2k3P1/8/2K2P2/8 w - - 0 1",
    "8/pp2r1k1/2p1p3/3pP2p/1P1P1P1P/P5KR/8/8 w - - 0 1",
    "8/3p4/p1bk3p/Pp6/1Kp1PpPp/2P2P1P/2P5/5B2 b - - 0 1",
    "5k2/7R/4P2p/5K2/p1r2P1p/8/8/8 b - - 0 1",
    "6k1/6p1/P6p/r1N5/5p2/7P/1b3PP1/4R1K1 w - - 0 1",
    "1r3k2/4q3/2Pp3b/3Bp3/2Q2p2/1p1P2P1/1P2KP2/3N4 w - - 0 1",
    "6k1/4pp1p/3p2p1/P1pPb3/R7/1r2P1PP/3B1P2/6K1 w - - 0 1",
    "8/3p3B/5p2/5P2/p7/PP5b/k7/6K1 w - - 0 1",
    "5rk1/q6p/2p3bR/1pPp1rP1/1P1Pp3/P3B1Q1/1K3P2/R7 w - - 93 90",
    "4rrk1/1p1nq3/p7/2p1P1pp/3P2bp/3Q1Bn1/PPPB4/1K2R1NR w - - 40 21",
    "r3k2r/3nnpbp/q2pp1p1/p7/Pp1PPPP1/4BNN1/1P5P/R2Q1RK1 w kq - 0 16",
    "3Qb1k1/1r2ppb1/pN1n2q1/Pp1Pp1Pr/4P2p/4BP2/4B1R1/1R5K b - - 11 40",
    "4k3/3q1r2/1N2r1b1/3ppN2/2nPP3/1B1R2n1/2R1Q3/3K4 w - - 5 1",
    "k7/2n1n3/1nbNbn2/2NbRBn1/1nbRQR2/2NBRBN1/3N1N2/7K w - - 0 1",
    "K7/8/8/BNQNQNB1/N5N1/R1Q1q2r/n5n1/bnqnqnbk w - - 0 1",
    "8/8/8/8/5kp1/P7/8/1K1N4 w - - 0 1",
    "8/8/8/5N2/8/p7/8/2NK3k w - - 0 1",
    "8/3k4/8/8/8/4B3/4KB2/2B5 w - - 0 1",
    "8/8/1P6/5pr1/8/4R3/7k/2K5 w - - 0 1",
    "8/2p4P/8/kr6/6R1/8/8/1K6 w - - 0 1",
    "8/8/3P3k/8/1p6/8/1P6/1K3n2 b - - 0 1",
    "8/R7/2q5/8/6k1/8/1P5p/K6R w - - 0 124",
    "6k1/3b3r/1p1p4/p1n2p2/1PPNpP1q/P3Q1p1/1R1RB1P1/5K2 b - - 0 1",
    "r2r1n2/pp2bk2/2p1p2p/3q4/3PN1QP/2P3R1/P4PP1/5RK1 w - - 0 1",
    "8/8/8/8/8/6k1/6p1/6K1 w - -",
    "7k/7P/6K1/8/3B4/8/8/8 b - -",
]


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
            "bench": self._cmd_bench,
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

    def _cmd_bench(self, args: list[str]) -> None:
        """bench [depth]"""
        import sys as _sys
        import time as _time

        try:
            depth = int(args[0])
        except IndexError, ValueError:
            depth = 5

        # we recursively find the first searcher that has '_depth' and set it
        def set_searcher_depth(searcher, d: int) -> int | None:
            if hasattr(searcher, "_depth"):
                old = getattr(searcher, "_depth")
                setattr(searcher, "_depth", d)
                return old
            if hasattr(searcher, "_inner"):
                return set_searcher_depth(getattr(searcher, "_inner"), d)
            return None

        old_depth = set_searcher_depth(self._engine.searcher, depth)

        # clear TT
        def clear_searcher_tt(searcher) -> None:
            if hasattr(searcher, "_tt") and searcher._tt is not None:
                searcher._tt.clear()
            elif hasattr(searcher, "_inner"):
                clear_searcher_tt(getattr(searcher, "_inner"))

        clear_searcher_tt(self._engine.searcher)

        start = _time.monotonic_ns()
        nodes = 0
        num = len(_BENCH_FENS)

        for cnt, bench_entry in enumerate(_BENCH_FENS, 1):
            if " moves " in bench_entry:
                fen, moves_str = bench_entry.split(" moves ", 1)
                moves = moves_str.split()
            else:
                fen = bench_entry
                moves = []

            print(f"position: {cnt}/{num} ({fen})", file=_sys.stderr)

            self._engine.set_position(fen, moves)
            self._engine._stop_event.clear()

            res = self._engine.searcher.search(
                self._engine.board,
                stop_condition=lambda: False,
            )
            nodes += res.nodes

        elapsed_ns = max(_time.monotonic_ns() - start, 1)
        elapsed_ms = elapsed_ns // 1_000_000
        nps = (nodes * 1_000) // (elapsed_ms if elapsed_ms > 0 else 1)

        print("\n===========================", file=_sys.stderr)
        print(f"Total time (ms) : {elapsed_ms}", file=_sys.stderr)
        print(f"Nodes searched  : {nodes}", file=_sys.stderr)
        print(f"Nodes/second    : {nps}", file=_sys.stderr)

        # restore depth
        if old_depth is not None:
            set_searcher_depth(self._engine.searcher, old_depth)
