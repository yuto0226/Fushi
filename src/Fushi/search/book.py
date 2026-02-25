from __future__ import annotations

import chess

from Fushi.book import BookReader

from . import InfoCallback, SearchResult, Searcher, StopCondition


class BookSearcher(Searcher):
    """
    A :class:`~Fushi.search.Searcher` decorator that consults an opening
    book before delegating to an inner searcher.

    Usage::

        from Fushi.book import PolyglotBookReader
        from Fushi.search import MinMaxSearcher, TranspositionTable
        from Fushi.search.book import BookSearcher
        from Fushi.evaluate import ShannonEvaluator

        tt   = TranspositionTable()
        inner = MinMaxSearcher(ShannonEvaluator(), tt=tt)
        book  = PolyglotBookReader("books/Titans.bin")
        searcher = BookSearcher(inner, book)

    The ``Engine`` receives ``searcher`` and does not need to be aware
    of the book at all.  Swapping the inner searcher (e.g. to an
    AlphaBeta engine later) requires only changing one line.

    Parameters
    ----------
    inner:
        The underlying searcher to fall back to when the position is
        not covered by the book.
    book:
        Any :class:`~Fushi.book.BookReader` implementation.
    """

    def __init__(self, inner: Searcher, book: BookReader) -> None:
        # Share the transposition table with the inner searcher so that
        # book moves do not leave a gap in the TT between searches.
        super().__init__(tt=inner._tt)
        self._inner = inner
        self._book = book

    def search(
        self,
        board: chess.Board,
        *,
        on_info: InfoCallback | None = None,
        stop_condition: StopCondition | None = None,
    ) -> SearchResult:
        """
        Return a book move immediately if available, otherwise delegate
        to the inner searcher for a full tree search.
        """
        book_move = self._book.probe(board)
        if book_move is not None:
            return SearchResult(
                best_move=book_move,
                score=0,  # book moves carry no centipawn estimate
                depth=0,
                nodes=0,
                pv=[book_move],
            )

        return self._inner.search(
            board,
            on_info=on_info,
            stop_condition=stop_condition,
        )
