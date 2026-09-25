from __future__ import annotations

from dataclasses import dataclass, replace


RANK_NAMES = {
    1: "A",
    11: "J",
    12: "Q",
    13: "K",
}
SUIT_NAMES = ("C", "D", "S", "H")


@dataclass(frozen=True)
class Card:
    """A playing card.

    Suits are integers so experiments can use smaller or unusual decks. For the
    normal four-suit deck, suit colors alternate by parity: clubs/spades black,
    diamonds/hearts red.
    """

    suit: int
    rank: int
    face_up: bool = False

    @property
    def color(self) -> int:
        return self.suit % 2

    def face_up_card(self) -> "Card":
        return replace(self, face_up=True)

    def face_down_card(self) -> "Card":
        return replace(self, face_up=False)

    def short_name(self) -> str:
        rank = RANK_NAMES.get(self.rank, str(self.rank))
        suit = SUIT_NAMES[self.suit] if self.suit < len(SUIT_NAMES) else f"S{self.suit}"
        return f"{rank}{suit}" if self.face_up else "XX"

    def __str__(self) -> str:
        return self.short_name()
