"""
Card representation and Scryfall data integration.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import hashlib
import json
import re


class CardType(Enum):
    CREATURE = auto()
    INSTANT = auto()
    SORCERY = auto()
    ARTIFACT = auto()
    ENCHANTMENT = auto()
    PLANESWALKER = auto()
    LAND = auto()
    BATTLE = auto()


class Color(Enum):
    WHITE = "W"
    BLUE = "U"
    BLACK = "B"
    RED = "R"
    GREEN = "G"
    COLORLESS = "C"


# Common keywords we care about for simulation
KEYWORDS = {
    # Evergreen
    "flying",
    "first strike",
    "double strike",
    "deathtouch",
    "haste",
    "hexproof",
    "indestructible",
    "lifelink",
    "menace",
    "reach",
    "trample",
    "vigilance",
    "ward",
    "flash",
    "defender",
    # Draw/Ramp related
    "draw",
    "scry",
    "surveil",
    # Mana related
    "add",
    "treasure",
    # ETB related
    "enters the battlefield",
    "etb",
    # Recursion
    "return",
    "graveyard",
}


@dataclass
class ManaCost:
    """Represents a mana cost like {2}{U}{U}."""

    white: int = 0
    blue: int = 0
    black: int = 0
    red: int = 0
    green: int = 0
    colorless: int = 0  # Generic mana requirement

    @classmethod
    def from_string(cls, mana_string: str) -> "ManaCost":
        """Parse a mana cost string like '{2}{U}{U}' or '2UU'."""
        if not mana_string:
            return cls()

        cost = cls()

        # Handle both {2}{U}{U} and 2UU formats
        # Extract all components
        generic_match = re.findall(r"\{(\d+)\}|^(\d+)", mana_string)
        for match in generic_match:
            num = match[0] or match[1]
            if num:
                cost.colorless += int(num)

        # Count colored mana
        mana_upper = mana_string.upper()
        cost.white = mana_upper.count("W")
        cost.blue = mana_upper.count("U")
        cost.black = mana_upper.count("B")
        cost.red = mana_upper.count("R")
        cost.green = mana_upper.count("G")

        return cost

    @property
    def cmc(self) -> int:
        """Converted mana cost / mana value."""
        return self.white + self.blue + self.black + self.red + self.green + self.colorless

    @property
    def colors(self) -> set[Color]:
        """Return set of colors in this mana cost."""
        colors = set()
        if self.white > 0:
            colors.add(Color.WHITE)
        if self.blue > 0:
            colors.add(Color.BLUE)
        if self.black > 0:
            colors.add(Color.BLACK)
        if self.red > 0:
            colors.add(Color.RED)
        if self.green > 0:
            colors.add(Color.GREEN)
        return colors

    def can_pay_with(self, mana_pool: "ManaPool") -> bool:
        """Check if a mana pool can pay this cost."""
        # First check colored requirements
        if mana_pool.white < self.white:
            return False
        if mana_pool.blue < self.blue:
            return False
        if mana_pool.black < self.black:
            return False
        if mana_pool.red < self.red:
            return False
        if mana_pool.green < self.green:
            return False

        # Check if remaining mana can cover generic cost
        remaining = (
            (mana_pool.white - self.white)
            + (mana_pool.blue - self.blue)
            + (mana_pool.black - self.black)
            + (mana_pool.red - self.red)
            + (mana_pool.green - self.green)
            + mana_pool.colorless
        )
        return remaining >= self.colorless


@dataclass
class ManaPool:
    """Tracks available mana."""

    white: int = 0
    blue: int = 0
    black: int = 0
    red: int = 0
    green: int = 0
    colorless: int = 0

    def add(self, color: Color, amount: int = 1) -> None:
        """Add mana to the pool."""
        match color:
            case Color.WHITE:
                self.white += amount
            case Color.BLUE:
                self.blue += amount
            case Color.BLACK:
                self.black += amount
            case Color.RED:
                self.red += amount
            case Color.GREEN:
                self.green += amount
            case Color.COLORLESS:
                self.colorless += amount

    def pay(self, cost: ManaCost) -> bool:
        """
        Pay a mana cost from this pool.
        Returns True if successful, False if insufficient mana.
        Modifies pool in place.
        """
        if not cost.can_pay_with(self):
            return False

        # Pay colored costs first
        self.white -= cost.white
        self.blue -= cost.blue
        self.black -= cost.black
        self.red -= cost.red
        self.green -= cost.green

        # Pay generic with whatever's left (prioritize colorless)
        remaining_generic = cost.colorless

        # Use colorless first
        colorless_to_use = min(self.colorless, remaining_generic)
        self.colorless -= colorless_to_use
        remaining_generic -= colorless_to_use

        # Then use colored mana for generic (order doesn't matter for simulation)
        for color in ["white", "blue", "black", "red", "green"]:
            if remaining_generic <= 0:
                break
            available = getattr(self, color)
            to_use = min(available, remaining_generic)
            setattr(self, color, available - to_use)
            remaining_generic -= to_use

        return True

    def clear(self) -> None:
        """Empty the mana pool (end of phase)."""
        self.white = 0
        self.blue = 0
        self.black = 0
        self.red = 0
        self.green = 0
        self.colorless = 0

    @property
    def total(self) -> int:
        """Total mana available."""
        return self.white + self.blue + self.black + self.red + self.green + self.colorless

    def copy(self) -> "ManaPool":
        """Create a copy of this mana pool."""
        return ManaPool(
            white=self.white,
            blue=self.blue,
            black=self.black,
            red=self.red,
            green=self.green,
            colorless=self.colorless,
        )


@dataclass
class Card:
    """
    Represents a Magic card with attributes relevant for simulation.
    """

    name: str
    mana_cost: ManaCost
    card_types: set[CardType]
    oracle_text: str = ""

    # Creature stats
    power: Optional[int] = None
    toughness: Optional[int] = None

    # Color identity (for Commander legality)
    color_identity: set[Color] = field(default_factory=set)

    # Extracted keywords
    keywords: set[str] = field(default_factory=set)

    # Scryfall ID for reference
    scryfall_id: Optional[str] = None

    # Is this a commander?
    is_commander: bool = False

    def __post_init__(self):
        """Extract keywords from oracle text if not provided."""
        if not self.keywords and self.oracle_text:
            self._extract_keywords()

    def _extract_keywords(self) -> None:
        """Extract relevant keywords from oracle text."""
        text_lower = self.oracle_text.lower()
        for keyword in KEYWORDS:
            if keyword in text_lower:
                self.keywords.add(keyword)

    @property
    def cmc(self) -> int:
        """Mana value of the card."""
        return self.mana_cost.cmc

    @property
    def is_creature(self) -> bool:
        return CardType.CREATURE in self.card_types

    @property
    def is_land(self) -> bool:
        return CardType.LAND in self.card_types

    @property
    def is_instant(self) -> bool:
        return CardType.INSTANT in self.card_types

    @property
    def has_flash(self) -> bool:
        return "flash" in self.keywords

    @property
    def is_castable_at_instant_speed(self) -> bool:
        return self.is_instant or self.has_flash

    def can_be_in_deck(self, commander: "Card") -> bool:
        """Check if this card's color identity fits the commander."""
        return self.color_identity.issubset(commander.color_identity)

    @classmethod
    def from_scryfall(cls, data: dict) -> "Card":
        """Create a Card from Scryfall API response."""
        # Parse card types
        type_line = data.get("type_line", "").lower()
        card_types = set()

        if "creature" in type_line:
            card_types.add(CardType.CREATURE)
        if "instant" in type_line:
            card_types.add(CardType.INSTANT)
        if "sorcery" in type_line:
            card_types.add(CardType.SORCERY)
        if "artifact" in type_line:
            card_types.add(CardType.ARTIFACT)
        if "enchantment" in type_line:
            card_types.add(CardType.ENCHANTMENT)
        if "planeswalker" in type_line:
            card_types.add(CardType.PLANESWALKER)
        if "land" in type_line:
            card_types.add(CardType.LAND)
        if "battle" in type_line:
            card_types.add(CardType.BATTLE)

        # Parse color identity
        color_map = {
            "W": Color.WHITE,
            "U": Color.BLUE,
            "B": Color.BLACK,
            "R": Color.RED,
            "G": Color.GREEN,
        }
        color_identity = {color_map[c] for c in data.get("color_identity", []) if c in color_map}

        # Parse power/toughness
        power = None
        toughness = None
        if "power" in data:
            try:
                power = int(data["power"])
            except ValueError:
                power = 0  # Handle * or X
        if "toughness" in data:
            try:
                toughness = int(data["toughness"])
            except ValueError:
                toughness = 0

        # Check if legendary creature (potential commander)
        is_commander = "legendary" in type_line and CardType.CREATURE in card_types

        return cls(
            name=data.get("name", "Unknown"),
            mana_cost=ManaCost.from_string(data.get("mana_cost", "")),
            card_types=card_types,
            oracle_text=data.get("oracle_text", ""),
            power=power,
            toughness=toughness,
            color_identity=color_identity,
            scryfall_id=data.get("id"),
            is_commander=is_commander,
        )

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return False
        return self.name == other.name

    def __repr__(self) -> str:
        if self.is_creature:
            return f"Card({self.name}, {self.cmc}mv, {self.power}/{self.toughness})"
        return f"Card({self.name}, {self.cmc}mv)"


# Common basic lands for testing
BASIC_LANDS = {
    "Plains": Card(
        name="Plains",
        mana_cost=ManaCost(),
        card_types={CardType.LAND},
        oracle_text="{T}: Add {W}.",
        color_identity={Color.WHITE},
        keywords={"add"},
    ),
    "Island": Card(
        name="Island",
        mana_cost=ManaCost(),
        card_types={CardType.LAND},
        oracle_text="{T}: Add {U}.",
        color_identity={Color.BLUE},
        keywords={"add"},
    ),
    "Swamp": Card(
        name="Swamp",
        mana_cost=ManaCost(),
        card_types={CardType.LAND},
        oracle_text="{T}: Add {B}.",
        color_identity={Color.BLACK},
        keywords={"add"},
    ),
    "Mountain": Card(
        name="Mountain",
        mana_cost=ManaCost(),
        card_types={CardType.LAND},
        oracle_text="{T}: Add {R}.",
        color_identity={Color.RED},
        keywords={"add"},
    ),
    "Forest": Card(
        name="Forest",
        mana_cost=ManaCost(),
        card_types={CardType.LAND},
        oracle_text="{T}: Add {G}.",
        color_identity={Color.GREEN},
        keywords={"add"},
    ),
}
