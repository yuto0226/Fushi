<div align="center">

# Fushi

**A UCI-compatible chess engine written in Python**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.14%2B-blue)](https://www.python.org/)

</div>

## Overview

Fushi is a chess engine that communicates via the [UCI protocol](https://www.shredderchess.com/chess-features/uci-universal-chess-interface.html), making it compatible with most chess GUIs (e.g. [En Croissant](https://encroissant.org/), [Lichess](https://lichess.org/)).

## Quickstart

### Clone & Install

Clone the repository and install all dependencies using [uv](https://github.com/astral-sh/uv):

```sh
git clone https://github.com/yuto0226/fushi.git && cd fushi
uv sync
```

### Usage

Fushi communicates over stdin/stdout using the UCI protocol. You can interact with it directly in the terminal, or connect it to any UCI-compatible chess GUI.

```sh
$ ./fushi
> uci
id name Fushi
id author Yuto
uciok
> isready
readyok
> go
info depth 3 score cp 2 nodes 9322 time 527 pv e2e3 g7g6 f1a6
bestmove e2e3
> quit
```

## Development

```sh
uv sync --dev

# Run tests
uv run pytest

# Lint & format
uv run ruff check .
uv run ruff format .

# Type check
uv run pyright
```

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

This project uses [python-chess](https://github.com/niklasf/python-chess), which is also licensed under GPLv3.

## Star History

<a href="https://www.star-history.com/#yuto0226/fushi&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=yuto0226/fushi&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=yuto0226/fushi&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=yuto0226/fushi&type=date&legend=top-left" />
 </picture>
</a>

---

<div align="center">

Made with ❤️ by [Yuto](https://blog.yuto0226.dev/)

</div>
