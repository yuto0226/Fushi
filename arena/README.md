# Tournament Manager

This manager is used to run tournaments between two chess engines. Quick test for new engine improvements.

https://database.lichess.org/

## Usage

```bash
uv run arena.py download --fens 500
uv run arena.py run --games 500 --workers 20 --time 1 --old ../old_engine/Fushi/fushi --new ../fushi
```
