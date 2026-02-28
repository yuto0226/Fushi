import io
import chess.pgn
import zstandard as zstd
from collections import Counter
from tqdm import tqdm
import math

LICHESS_DB_LOCAL = "lichess_db.pgn.zst"


def analyze_elo_distribution(zst_path, max_games=None):
    total_games = 0
    elo_counter = Counter()

    print(f"Analyzing {zst_path}...")

    with open(zst_path, "rb") as zst_file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(zst_file) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")

            with tqdm(desc="Analyzing Games") as pbar:
                while True:
                    if max_games and total_games >= max_games:
                        break

                    try:
                        # Only parse headers to be fast
                        headers = chess.pgn.read_headers(text_stream)
                        if headers is None:
                            break  # EOF

                        total_games += 1

                        # Get White and Black ELO
                        white_elo = headers.get("WhiteElo", "?")
                        black_elo = headers.get("BlackElo", "?")

                        for elo_str in (white_elo, black_elo):
                            if elo_str and elo_str != "?":
                                try:
                                    elo = int(elo_str)
                                    # Group by hundreds (e.g., 1542 -> 1500)
                                    bucket = (math.floor(elo / 100)) * 100
                                    elo_counter[bucket] += 1
                                except ValueError:
                                    pass

                        pbar.update(1)
                    except Exception:
                        # Ignore parsing errors for malformed games
                        pass

    print("\n--- Analysis Results ---")
    print(f"Total Games Parsed: {total_games}")

    print("\nELO Distribution (by hundreds):")
    total_players = sum(elo_counter.values())

    for elo_bucket in sorted(elo_counter.keys()):
        count = elo_counter[elo_bucket]
        pct = (count / total_players) * 100 if total_players > 0 else 0

        # Simple ASCII bar
        bar_len = int(pct / 2)
        bar = "█" * max(1, bar_len)

        if elo_bucket >= 3000:
            label = "3000+"
        else:
            label = f"{elo_bucket}-{elo_bucket + 99}"

        print(f"{label:>9}: {count:>8} ({pct:>5.2f}%) | {bar}")


if __name__ == "__main__":
    analyze_elo_distribution(
        LICHESS_DB_LOCAL, max_games=2000000
    )  # Analyze first 200k games for speed
