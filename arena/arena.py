import concurrent.futures
import io
import os
import random
import sys
import math
import chess
import chess.engine
import chess.pgn
import requests
import zstandard as zstd
from tqdm import tqdm

LICHESS_DB_URL = (
    "https://database.lichess.org/standard/lichess_db_standard_rated_2016-03.pgn.zst"
)
LICHESS_DB_LOCAL = "lichess_db.pgn.zst"
OPENINGS_FILE = "openings.fen"


def download_lichess_db(url, output_path):
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping download.")
        return

    print("Downloading Lichess DB (this might take a while)...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with (
        open(output_path, "wb") as f,
        tqdm(
            desc=output_path,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in response.iter_content(chunk_size=1024 * 1024):
            size = f.write(data)
            bar.update(size)


def extract_openings(zst_path, fen_path, count=500):
    if os.path.exists(fen_path):
        print(f"File {fen_path} already exists. Skipping extraction.")
        return

    print("Extracting opening FENs from ZST...")
    fens = {}
    seen_openings = set()

    with open(zst_path, "rb") as zst_file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(zst_file) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")

            with tqdm(total=count, desc="Extracting FENs") as pbar:
                while len(fens) < count:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break  # EOF

                    white_elo = game.headers.get("WhiteElo", "0")
                    black_elo = game.headers.get("BlackElo", "0")
                    try:
                        if int(white_elo) <= 1500 or int(black_elo) <= 1500:
                            continue
                    except ValueError:
                        continue

                    board = game.board()
                    ply_to_play = math.floor(16 + 20 * random.random()) & ~1
                    num_ply_played = 0
                    fen = None

                    for move in game.mainline_moves():
                        board.push(move)
                        num_ply_played += 1
                        if num_ply_played == ply_to_play:
                            fen = board.fen()

                    if fen is not None:
                        num_pieces = sum(fen.lower().count(char) for char in "rnbq")
                        if num_ply_played > ply_to_play + 40 and num_pieces >= 10:
                            opening = game.headers.get("Opening", "Unknown")
                            if opening != "Unknown" and opening not in seen_openings:
                                if fen not in fens:
                                    fens[fen] = opening
                                    seen_openings.add(opening)
                                    pbar.update(1)

    with open(fen_path, "w") as f:
        for fen, opening in fens.items():
            f.write(f"{opening}\n{fen}\n")
    print(f"Extracted {len(fens)} FENs to {fen_path}")


def play_game(fen, old_engine_path, new_engine_path, time_limit, depth_limit):
    old_engine_abs = os.path.abspath(old_engine_path)
    new_engine_abs = os.path.abspath(new_engine_path)

    # We use chess.engine.SimpleEngine for synchronous operations
    old_engine = chess.engine.SimpleEngine.popen_uci(
        ["uv", "run", "python", old_engine_abs]
    )
    new_engine = chess.engine.SimpleEngine.popen_uci(
        ["uv", "run", "python", new_engine_abs]
    )

    # 50% chance for new engine to be white
    if random.choice([True, False]):
        white = new_engine
        white_name = "NewEngine"
        black = old_engine
        black_name = "OldEngine"
    else:
        white = old_engine
        white_name = "OldEngine"
        black = new_engine
        black_name = "NewEngine"

    board = chess.Board(fen)
    moves = []

    # Determine limits
    if depth_limit is not None:
        limit = chess.engine.Limit(depth=depth_limit)
    else:
        limit = chess.engine.Limit(time=time_limit or 0.1)

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            result = white.play(board, limit)
        else:
            result = black.play(board, limit)

        if result.move is None:
            break

        board.push(result.move)
        moves.append(result.move.uci())

    outcome = board.outcome(claim_draw=True)
    winner = outcome.winner if outcome else None

    if winner == chess.WHITE:
        res = "1-0" if white_name == "NewEngine" else "0-1"
    elif winner == chess.BLACK:
        res = "0-1" if black_name == "NewEngine" else "1-0"
    else:
        res = "1/2-1/2"

    old_engine.quit()
    new_engine.quit()

    return res


def run_tournament(
    fen_path,
    old_engine_path,
    new_engine_path,
    match_count=500,
    concurrency=20,
    time_limit=None,
    depth_limit=None,
):
    from collections import defaultdict

    print(f"Loading FENs from {fen_path}...")
    with open(fen_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    openings_dict = defaultdict(list)
    for i in range(0, len(lines) - 1, 2):
        opening = lines[i]
        fen = lines[i + 1]
        openings_dict[opening].append(fen)

    selected_fens = []

    for op in openings_dict:
        random.shuffle(openings_dict[op])

    opening_keys = list(openings_dict.keys())
    random.shuffle(opening_keys)

    while len(selected_fens) < match_count and opening_keys:
        for op in list(opening_keys):
            if len(selected_fens) >= match_count:
                break
            selected_fens.append(openings_dict[op].pop())
            if not openings_dict[op]:
                opening_keys.remove(op)

    fens = selected_fens

    print(f"Starting tournament: {match_count} games with {concurrency} concurrency.")

    wins = 0
    losses = 0
    draws = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                play_game,
                fen,
                old_engine_path,
                new_engine_path,
                time_limit,
                depth_limit,
            ): fen
            for fen in fens
        }

        with tqdm(total=match_count, desc="Matches") as pbar:
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res == "1-0":
                    wins += 1
                elif res == "0-1":
                    losses += 1
                else:
                    draws += 1

                pbar.set_postfix({"W": wins, "L": losses, "D": draws})
                pbar.update(1)

    print("\n--- Tournament Results (New vs Old) ---")
    print(f"Wins:  {wins}")
    print(f"Draws: {draws}")
    print(f"Losses: {losses}")

    if wins + losses > 0:
        win_rate = (wins + (draws / 2)) / match_count
        print(f"Score: {(wins + draws / 2)} / {match_count} ({win_rate * 100:.1f}%)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Chess Engine Tournament Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Download command
    parser_download = subparsers.add_parser(
        "download", help="Download DB and extract FENs"
    )
    parser_download.add_argument(
        "--fens", type=int, default=500, help="Number of FENs to extract"
    )

    # Run command
    parser_run = subparsers.add_parser("run", help="Run tournament")
    parser_run.add_argument(
        "--games", type=int, default=500, help="Number of games to play"
    )
    parser_run.add_argument("--workers", type=int, default=20, help="Concurrency level")
    parser_run.add_argument(
        "--time", type=float, default=1, help="Time limit per move (seconds)"
    )
    parser_run.add_argument(
        "--depth", type=int, default=None, help="Fixed search depth per move"
    )
    parser_run.add_argument(
        "--old", default="../old_engine/Fushi/fushi", help="Path to old engine"
    )
    parser_run.add_argument("--new", default="../fushi", help="Path to new engine")

    args = parser.parse_args()

    if args.command == "download":
        download_lichess_db(LICHESS_DB_URL, LICHESS_DB_LOCAL)
        extract_openings(LICHESS_DB_LOCAL, OPENINGS_FILE, count=args.fens)
    elif args.command == "run":
        if not os.path.exists(OPENINGS_FILE):
            print(f"Error: {OPENINGS_FILE} not found. Run 'download' command first.")
            sys.exit(1)

        if args.time is None and args.depth is None:
            args.time = 0.1

        run_tournament(
            OPENINGS_FILE,
            args.old,
            args.new,
            args.games,
            args.workers,
            args.time,
            args.depth,
        )


if __name__ == "__main__":
    main()
