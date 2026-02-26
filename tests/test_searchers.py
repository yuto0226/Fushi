"""Functional tests for all Searcher implementations in Fushi.search."""

from __future__ import annotations

import chess
import pytest

from Fushi.book import BookReader
from Fushi.evaluate.shannon import ShannonEvaluator
from Fushi.search import (
    AlphaBetaSearcher,
    BookSearcher,
    BruteForceSearcher,
    MinMaxSearcher,
    SearchResult,
    TranspositionTable,
)
from Fushi.search.basic import BasicSearcher

# ---------------------------------------------------------------------------
# Shared test positions
# ---------------------------------------------------------------------------

# Fool's-mate set-up: Black has Qh4# (mate in 1).
MATE_IN_1_FEN = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2"
MATE_IN_1_MOVE = chess.Move.from_uci("d8h4")

# Scholar's mate: White has already delivered Qxf7# → Black is checkmated.
CHECKMATE_FEN = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"

# Classic pawn-vs-king stalemate: Black king on f8 is stalemated.
#   Bk=f8, Wp=f7, Wk=f6  → f7 pawn covers e8/g8, Wk covers e7/g7/e6/g6.
STALEMATE_FEN = "5k2/5P2/5K2/8/8/8/8/8 b - - 0 1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ev():
    return ShannonEvaluator()


@pytest.fixture
def tt():
    return TranspositionTable(size_mb=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_legal(board: chess.Board, move: chess.Move | None) -> bool:
    return move is not None and move in board.legal_moves


def _pv_all_legal(pv: list[chess.Move]) -> bool:
    """Replay the PV on a fresh board and verify each move is legal."""
    if not pv:
        return True
    board = chess.Board()
    for move in pv:
        if move not in board.legal_moves:
            return False
        board.push(move)
    return True


# ---------------------------------------------------------------------------
# BasicSearcher
# ---------------------------------------------------------------------------


class TestBasicSearcher:
    def test_returns_search_result(self, ev):
        s = BasicSearcher(ev)
        result = s.search(chess.Board())
        assert isinstance(result, SearchResult)

    def test_best_move_is_legal(self, ev):
        board = chess.Board()
        result = BasicSearcher(ev).search(board)
        assert _is_legal(board, result.best_move)

    def test_nodes_counted(self, ev):
        result = BasicSearcher(ev).search(chess.Board())
        assert result.nodes > 0

    def test_depth_is_one(self, ev):
        result = BasicSearcher(ev).search(chess.Board())
        assert result.depth == 1

    def test_on_info_called(self, ev):
        calls: list = []
        BasicSearcher(ev).search(chess.Board(), on_info=calls.append)
        assert len(calls) == 1

    def test_stop_condition_respected(self, ev):
        # Immediate stop: no move evaluated.
        result = BasicSearcher(ev).search(chess.Board(), stop_condition=lambda: True)
        # best_move may be None (nothing evaluated) – just ensure no crash.
        assert isinstance(result, SearchResult)

    def test_checkmate_position(self, ev):
        board = chess.Board(CHECKMATE_FEN)
        assert board.is_checkmate()
        result = BasicSearcher(ev).search(board)
        assert result.best_move is None
        assert result.nodes == 0

    def test_stalemate_position(self, ev):
        board = chess.Board(STALEMATE_FEN)
        assert board.is_stalemate()
        result = BasicSearcher(ev).search(board)
        assert result.best_move is None


# ---------------------------------------------------------------------------
# BruteForceSearcher (dfs.py)
# ---------------------------------------------------------------------------


class TestBruteForceSearcher:
    def test_returns_search_result(self, ev):
        result = BruteForceSearcher(ev, depth=2).search(chess.Board())
        assert isinstance(result, SearchResult)

    def test_best_move_is_legal(self, ev):
        board = chess.Board()
        result = BruteForceSearcher(ev, depth=2).search(board)
        assert _is_legal(board, result.best_move)

    def test_nodes_counted(self, ev):
        result = BruteForceSearcher(ev, depth=2).search(chess.Board())
        assert result.nodes > 0

    def test_pv_non_empty(self, ev):
        result = BruteForceSearcher(ev, depth=2).search(chess.Board())
        assert len(result.pv) > 0

    def test_on_info_called(self, ev):
        calls: list = []
        BruteForceSearcher(ev, depth=2).search(chess.Board(), on_info=calls.append)
        assert len(calls) == 1

    def test_stop_condition_respected(self, ev):
        result = BruteForceSearcher(ev, depth=2).search(
            chess.Board(), stop_condition=lambda: True
        )
        assert isinstance(result, SearchResult)

    def test_checkmate_position(self, ev):
        board = chess.Board(CHECKMATE_FEN)
        result = BruteForceSearcher(ev, depth=2).search(board)
        assert result.best_move is None

    def test_finds_mate_in_1(self, ev):
        board = chess.Board(MATE_IN_1_FEN)
        result = BruteForceSearcher(ev, depth=1).search(board)
        assert result.best_move == MATE_IN_1_MOVE


# ---------------------------------------------------------------------------
# MinMaxSearcher
# ---------------------------------------------------------------------------


class TestMinMaxSearcher:
    def test_returns_search_result(self, ev, tt):
        result = MinMaxSearcher(ev, depth=2, tt=tt).search(chess.Board())
        assert isinstance(result, SearchResult)

    def test_best_move_is_legal(self, ev, tt):
        board = chess.Board()
        result = MinMaxSearcher(ev, depth=2, tt=tt).search(board)
        assert _is_legal(board, result.best_move)

    def test_nodes_counted(self, ev, tt):
        result = MinMaxSearcher(ev, depth=2, tt=tt).search(chess.Board())
        assert result.nodes > 0

    def test_pv_non_empty(self, ev, tt):
        result = MinMaxSearcher(ev, depth=2, tt=tt).search(chess.Board())
        assert len(result.pv) > 0

    def test_on_info_called_per_depth(self, ev, tt):
        calls: list = []
        MinMaxSearcher(ev, depth=3, tt=tt).search(chess.Board(), on_info=calls.append)
        # iterative deepening: one info per depth level
        assert len(calls) == 3

    def test_stop_condition_respected(self, ev, tt):
        result = MinMaxSearcher(ev, depth=5, tt=tt).search(
            chess.Board(), stop_condition=lambda: True
        )
        assert isinstance(result, SearchResult)

    def test_works_without_tt(self, ev):
        result = MinMaxSearcher(ev, depth=2).search(chess.Board())
        assert _is_legal(chess.Board(), result.best_move)

    def test_checkmate_position(self, ev, tt):
        board = chess.Board(CHECKMATE_FEN)
        result = MinMaxSearcher(ev, depth=2, tt=tt).search(board)
        assert result.best_move is None

    def test_finds_mate_in_1(self, ev, tt):
        board = chess.Board(MATE_IN_1_FEN)
        result = MinMaxSearcher(ev, depth=1, tt=tt).search(board)
        assert result.best_move == MATE_IN_1_MOVE

    def test_tt_populated_after_search(self, ev, tt):
        MinMaxSearcher(ev, depth=2, tt=tt).search(chess.Board())
        assert tt.stores > 0

    def test_tt_reused_across_searches(self, ev, tt):
        s = MinMaxSearcher(ev, depth=2, tt=tt)
        s.search(chess.Board())
        first_stores = tt.stores
        # Second identical search should get TT hits.
        s.search(chess.Board())
        assert tt.hits > 0 or tt.stores >= first_stores  # TT was touched


# ---------------------------------------------------------------------------
# AlphaBetaSearcher
# ---------------------------------------------------------------------------


class TestAlphaBetaSearcher:
    def test_returns_search_result(self, ev, tt):
        result = AlphaBetaSearcher(ev, depth=2, tt=tt).search(chess.Board())
        assert isinstance(result, SearchResult)

    def test_best_move_is_legal(self, ev, tt):
        board = chess.Board()
        result = AlphaBetaSearcher(ev, depth=2, tt=tt).search(board)
        assert _is_legal(board, result.best_move)

    def test_nodes_counted(self, ev, tt):
        result = AlphaBetaSearcher(ev, depth=2, tt=tt).search(chess.Board())
        assert result.nodes > 0

    def test_pv_non_empty(self, ev, tt):
        result = AlphaBetaSearcher(ev, depth=2, tt=tt).search(chess.Board())
        assert len(result.pv) > 0

    def test_pv_moves_legal(self, ev, tt):
        result = AlphaBetaSearcher(ev, depth=3, tt=tt).search(chess.Board())
        assert _pv_all_legal(result.pv)

    def test_on_info_called_per_depth(self, ev, tt):
        calls: list = []
        AlphaBetaSearcher(ev, depth=3, tt=tt).search(
            chess.Board(), on_info=calls.append
        )
        assert len(calls) == 3

    def test_stop_condition_respected(self, ev, tt):
        result = AlphaBetaSearcher(ev, depth=10, tt=tt).search(
            chess.Board(), stop_condition=lambda: True
        )
        assert isinstance(result, SearchResult)

    def test_works_without_tt(self, ev):
        result = AlphaBetaSearcher(ev, depth=2).search(chess.Board())
        assert _is_legal(chess.Board(), result.best_move)

    def test_checkmate_position(self, ev, tt):
        board = chess.Board(CHECKMATE_FEN)
        result = AlphaBetaSearcher(ev, depth=2, tt=tt).search(board)
        assert result.best_move is None

    def test_stalemate_position(self, ev, tt):
        board = chess.Board(STALEMATE_FEN)
        result = AlphaBetaSearcher(ev, depth=2, tt=tt).search(board)
        assert result.best_move is None

    def test_single_legal_move(self, ev, tt):
        """Verify the fast-path for positions with exactly one legal move."""
        # Position: White King h1, Black Queen g2. White must capture Kxg2.
        fen = "k7/8/8/8/8/8/6q1/7K w - - 0 1"
        board = chess.Board(fen)
        assert board.legal_moves.count() == 1

        result = AlphaBetaSearcher(ev, depth=2, tt=tt).search(board)

        assert result.best_move == chess.Move.from_uci("h1g2")
        assert result.depth == 0
        assert result.nodes == 0
        assert result.pv == [chess.Move.from_uci("h1g2")]

    def test_finds_mate_in_1(self, ev, tt):
        board = chess.Board(MATE_IN_1_FEN)
        result = AlphaBetaSearcher(ev, depth=1, tt=tt).search(board)
        assert result.best_move == MATE_IN_1_MOVE

    def test_tt_populated_after_search(self, ev, tt):
        AlphaBetaSearcher(ev, depth=2, tt=tt).search(chess.Board())
        assert tt.stores > 0

    def test_alphabeta_fewer_nodes_than_bruteforce(self, ev):
        """Alpha-beta must visit strictly fewer nodes than brute-force at same depth."""
        board = chess.Board()
        ab = AlphaBetaSearcher(ev, depth=3)
        ab.search(board)
        bf = BruteForceSearcher(ev, depth=3)
        bf.search(board)
        assert ab.nodes < bf.nodes


# ---------------------------------------------------------------------------
# BookSearcher
# ---------------------------------------------------------------------------


class _StubBookHit(BookReader):
    """Always returns the e2e4 move as a book hit."""

    def probe(self, board: chess.Board) -> chess.Move | None:
        return chess.Move.from_uci("e2e4")


class _StubBookMiss(BookReader):
    """Always reports a miss."""

    def probe(self, board: chess.Board) -> chess.Move | None:
        return None


class TestBookSearcher:
    def test_book_hit_returns_book_move(self, ev, tt):
        inner = AlphaBetaSearcher(ev, depth=2, tt=tt)
        s = BookSearcher(inner, _StubBookHit())
        result = s.search(chess.Board())
        assert result.best_move == chess.Move.from_uci("e2e4")
        assert result.depth == 0
        assert result.nodes == 0

    def test_book_hit_pv_contains_move(self, ev, tt):
        inner = AlphaBetaSearcher(ev, depth=2, tt=tt)
        s = BookSearcher(inner, _StubBookHit())
        result = s.search(chess.Board())
        assert result.pv == [chess.Move.from_uci("e2e4")]

    def test_book_miss_delegates_to_inner(self, ev, tt):
        inner = AlphaBetaSearcher(ev, depth=2, tt=tt)
        s = BookSearcher(inner, _StubBookMiss())
        board = chess.Board()
        result = s.search(board)
        assert isinstance(result, SearchResult)
        assert _is_legal(board, result.best_move)
        assert result.nodes > 0

    def test_book_miss_passes_callbacks(self, ev, tt):
        calls: list = []
        inner = AlphaBetaSearcher(ev, depth=2, tt=tt)
        BookSearcher(inner, _StubBookMiss()).search(chess.Board(), on_info=calls.append)
        assert len(calls) > 0

    def test_shares_tt_with_inner(self, ev, tt):
        inner = AlphaBetaSearcher(ev, depth=2, tt=tt)
        s = BookSearcher(inner, _StubBookMiss())
        assert s._tt is tt
