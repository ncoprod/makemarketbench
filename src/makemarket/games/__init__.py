"""Game registry."""

from .base import Belief, Episode, MarketParams, Quote
from .cards import CardsGame
from .dice import DiceGame

GAMES = {"dice": DiceGame, "cards": CardsGame}


def make_game(name: str):
    try:
        return GAMES[name]()
    except KeyError:
        raise ValueError(f"unknown game {name!r}; choose from {sorted(GAMES)}") from None


__all__ = ["Belief", "Episode", "MarketParams", "Quote", "make_game", "GAMES"]
