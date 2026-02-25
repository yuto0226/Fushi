from __future__ import annotations

from abc import ABC, abstractmethod

import chess


class BookReader(ABC):
    """
    Abstract interface for opening book readers.

    Any search algorithm can be augmented with a book by wrapping it
    inside a ``BookSearcher``. The ``probe`` method is the only required
    contract: given a board position, return a book move or ``None``.
    """

    @abstractmethod
    def probe(self, board: chess.Board) -> chess.Move | None:
        """
        Look up the current position in the opening book.

        Returns a legal :class:`chess.Move` chosen according to the
        book's weighting strategy, or ``None`` if the position is not
        covered.
        """
        ...


from .polyglot import PolyglotBookReader as PolyglotBookReader  # noqa: E402
