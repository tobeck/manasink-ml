"""
Card feature extraction for ML training.

This module provides functions to extract ML-ready features from Scryfall
card data and store them in the database for efficient retrieval during
training.

Key features extracted:
- Mana vector: (W, U, B, R, G, C) counts from mana cost
- Type flags: Boolean flags for each card type
- Keyword bitmap: 32-bit bitmap encoding common MTG keywords
- Power/toughness: For creatures
- Color identity: 5-bit WUBRG bitmap
- Role scores: Estimated role scores based on oracle text patterns
"""

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .database import DEFAULT_DB_PATH, create_features_schema

# Keywords to encode in the bitmap (up to 32)
KEYWORD_LIST = [
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
    "proliferate",
    "cascade",
    "storm",
    "affinity",
    "convoke",
    "delve",
    "dredge",
    "undying",
    "persist",
    "annihilator",
    "infect",
    "wither",
    "exalted",
    "landfall",
    "threshold",
    "kicker",
    "flashback",
]

# Patterns for role detection in oracle text (case-insensitive matching)
ROLE_PATTERNS = {
    "ramp": [
        r"add \{[wubrgcWUBRGC]\}",
        r"add \d+ mana",
        r"search your library for .* land",
        r"put .* land .* onto the battlefield",
        r"whenever .* land enters",
        r"mana of any",
        r"treasure token",
    ],
    "card_draw": [
        r"draw .* card",
        r"draw a card",
        r"draws? cards?",
        r"scry \d+",
        r"look at the top",
        r"reveal the top",
        r"put .* into your hand",
    ],
    "removal": [
        r"destroy target",
        r"exile target",
        r"deals? \d+ damage to",
        r"target creature gets -",
        r"-\d+/-\d+ until",
        r"sacrifice a creature",
        r"return target .* to its owner's hand",
        r"counter target",
    ],
    "board_wipe": [
        r"destroy all",
        r"exile all",
        r"each creature gets -",
        r"deals? \d+ damage to each",
        r"return all .* to their owners",
        r"sacrifice all",
    ],
    "protection": [
        r"hexproof",
        r"shroud",
        r"indestructible",
        r"protection from",
        r"can't be .* target",
        r"prevent .* damage",
        r"phase out",
        r"regenerate",
    ],
    "finisher": [
        r"double strike",
        r"infect",
        r"annihilator",
        r"extra combat",
        r"additional combat",
        r"you win the game",
        r"opponent loses the game",
        r"deals combat damage .* draw",
        r"whenever .* deals damage",
    ],
    "utility": [
        r"whenever",
        r"activated ability",
        r"at the beginning of",
        r"you may",
        r"choose one",
        r"copy",
    ],
}


@dataclass
class CardFeatures:
    """ML-ready features for a single card."""

    scryfall_id: str
    card_name: str

    # Mana vector (WUBRGC)
    mana_w: int = 0
    mana_u: int = 0
    mana_b: int = 0
    mana_r: int = 0
    mana_g: int = 0
    mana_c: int = 0  # Generic/colorless
    cmc: float = 0.0

    # Type flags
    is_creature: bool = False
    is_instant: bool = False
    is_sorcery: bool = False
    is_artifact: bool = False
    is_enchantment: bool = False
    is_planeswalker: bool = False
    is_land: bool = False
    is_legendary: bool = False

    # Keyword bitmap
    keyword_bitmap: int = 0

    # Creature stats
    power: Optional[int] = None
    toughness: Optional[int] = None

    # Color identity (5-bit WUBRG)
    color_identity_bitmap: int = 0

    # Role scores (0.0 to 1.0)
    role_ramp: float = 0.0
    role_card_draw: float = 0.0
    role_removal: float = 0.0
    role_board_wipe: float = 0.0
    role_protection: float = 0.0
    role_finisher: float = 0.0
    role_utility: float = 0.0

    def to_vector(self) -> list[float]:
        """Convert features to a flat vector for ML input."""
        return [
            # Mana (6 + 1 cmc = 7)
            float(self.mana_w),
            float(self.mana_u),
            float(self.mana_b),
            float(self.mana_r),
            float(self.mana_g),
            float(self.mana_c),
            float(self.cmc),
            # Types (8)
            float(self.is_creature),
            float(self.is_instant),
            float(self.is_sorcery),
            float(self.is_artifact),
            float(self.is_enchantment),
            float(self.is_planeswalker),
            float(self.is_land),
            float(self.is_legendary),
            # Stats (2)
            float(self.power or 0),
            float(self.toughness or 0),
            # Color identity (5 bits expanded)
            float((self.color_identity_bitmap >> 0) & 1),  # W
            float((self.color_identity_bitmap >> 1) & 1),  # U
            float((self.color_identity_bitmap >> 2) & 1),  # B
            float((self.color_identity_bitmap >> 3) & 1),  # R
            float((self.color_identity_bitmap >> 4) & 1),  # G
            # Roles (7)
            float(self.role_ramp),
            float(self.role_card_draw),
            float(self.role_removal),
            float(self.role_board_wipe),
            float(self.role_protection),
            float(self.role_finisher),
            float(self.role_utility),
        ]

    @staticmethod
    def vector_size() -> int:
        """Return the size of the feature vector."""
        return 7 + 8 + 2 + 5 + 7  # 29 features


def _parse_mana_cost(mana_cost: str) -> tuple[int, int, int, int, int, int]:
    """Parse mana cost string into (W, U, B, R, G, C) tuple."""
    if not mana_cost:
        return (0, 0, 0, 0, 0, 0)

    mana_cost = mana_cost.upper()

    w = mana_cost.count("W")
    u = mana_cost.count("U")
    b = mana_cost.count("B")
    r = mana_cost.count("R")
    g = mana_cost.count("G")

    # Parse generic mana (colorless cost)
    c = 0
    generic_match = re.findall(r"\{(\d+)\}", mana_cost)
    for match in generic_match:
        c += int(match)

    # Also handle X costs
    x_count = mana_cost.count("X")
    # X is treated as 0 for feature purposes

    return (w, u, b, r, g, c)


def _encode_keywords(oracle_text: str, keywords: list[str]) -> int:
    """Encode keywords as a 32-bit bitmap."""
    if not oracle_text:
        return 0

    text_lower = oracle_text.lower()
    all_keywords = keywords if keywords else []

    # Combine explicit keywords with text-detected keywords
    keyword_set = set(k.lower() for k in all_keywords)

    for kw in KEYWORD_LIST:
        if kw in text_lower:
            keyword_set.add(kw)

    bitmap = 0
    for i, kw in enumerate(KEYWORD_LIST):
        if kw in keyword_set:
            bitmap |= 1 << i

    return bitmap


def _encode_color_identity(color_identity: list[str]) -> int:
    """Encode color identity as 5-bit WUBRG bitmap."""
    if not color_identity:
        return 0

    bitmap = 0
    color_map = {"W": 0, "U": 1, "B": 2, "R": 3, "G": 4}

    for color in color_identity:
        if color in color_map:
            bitmap |= 1 << color_map[color]

    return bitmap


def _compute_role_scores(oracle_text: str, type_line: str) -> dict[str, float]:
    """Compute role scores based on oracle text patterns."""
    if not oracle_text:
        return {role: 0.0 for role in ROLE_PATTERNS}

    text_lower = oracle_text.lower()
    type_lower = type_line.lower() if type_line else ""

    scores = {}
    for role, patterns in ROLE_PATTERNS.items():
        matches = 0
        for pattern in patterns:
            if re.search(pattern, text_lower):
                matches += 1

        # Normalize to 0-1 range (cap at 3 matches = 1.0)
        scores[role] = min(matches / 3.0, 1.0)

    # Boost scores based on card type
    if "land" in type_lower:
        scores["ramp"] = max(scores["ramp"], 0.5)

    return scores


def extract_features_from_scryfall(card_data: dict) -> CardFeatures:
    """
    Extract ML features from a Scryfall card JSON.

    Args:
        card_data: Scryfall card JSON data

    Returns:
        CardFeatures object with extracted features
    """
    scryfall_id = card_data.get("id", "")
    name = card_data.get("name", "Unknown")
    mana_cost = card_data.get("mana_cost", "")
    type_line = card_data.get("type_line", "").lower()
    oracle_text = card_data.get("oracle_text", "")
    color_identity = card_data.get("color_identity", [])
    keywords = card_data.get("keywords", [])

    # Parse mana cost
    w, u, b, r, g, c = _parse_mana_cost(mana_cost)
    cmc = card_data.get("cmc", 0) or 0

    # Parse type line
    is_creature = "creature" in type_line
    is_instant = "instant" in type_line
    is_sorcery = "sorcery" in type_line
    is_artifact = "artifact" in type_line
    is_enchantment = "enchantment" in type_line
    is_planeswalker = "planeswalker" in type_line
    is_land = "land" in type_line
    is_legendary = "legendary" in type_line

    # Parse power/toughness
    power = None
    toughness = None
    if "power" in card_data:
        try:
            power = int(card_data["power"])
        except (ValueError, TypeError):
            power = 0  # Handle * or X
    if "toughness" in card_data:
        try:
            toughness = int(card_data["toughness"])
        except (ValueError, TypeError):
            toughness = 0

    # Encode keywords and color identity
    keyword_bitmap = _encode_keywords(oracle_text, keywords)
    color_identity_bitmap = _encode_color_identity(color_identity)

    # Compute role scores
    role_scores = _compute_role_scores(oracle_text, type_line)

    return CardFeatures(
        scryfall_id=scryfall_id,
        card_name=name,
        mana_w=w,
        mana_u=u,
        mana_b=b,
        mana_r=r,
        mana_g=g,
        mana_c=c,
        cmc=cmc,
        is_creature=is_creature,
        is_instant=is_instant,
        is_sorcery=is_sorcery,
        is_artifact=is_artifact,
        is_enchantment=is_enchantment,
        is_planeswalker=is_planeswalker,
        is_land=is_land,
        is_legendary=is_legendary,
        keyword_bitmap=keyword_bitmap,
        power=power,
        toughness=toughness,
        color_identity_bitmap=color_identity_bitmap,
        role_ramp=role_scores.get("ramp", 0),
        role_card_draw=role_scores.get("card_draw", 0),
        role_removal=role_scores.get("removal", 0),
        role_board_wipe=role_scores.get("board_wipe", 0),
        role_protection=role_scores.get("protection", 0),
        role_finisher=role_scores.get("finisher", 0),
        role_utility=role_scores.get("utility", 0),
    )


def populate_card_features(
    db_path: Optional[Path] = None,
    batch_size: int = 1000,
    show_progress: bool = True,
) -> int:
    """
    Populate the card_features table from Scryfall data in the cards table.

    Args:
        db_path: Path to SQLite database
        batch_size: Number of cards to process per batch
        show_progress: Show progress bar

    Returns:
        Number of cards processed
    """
    db_path = db_path or DEFAULT_DB_PATH

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    # Create features table if needed
    conn = create_features_schema(db_path)
    conn.row_factory = sqlite3.Row

    # Get total count
    cursor = conn.execute("SELECT COUNT(*) as count FROM cards")
    total = cursor.fetchone()["count"]

    print(f"Extracting features for {total:,} cards...")

    # Process in batches
    processed = 0
    offset = 0

    if show_progress and HAS_TQDM:
        pbar = tqdm(total=total, desc="Extracting features")
    else:
        pbar = None

    while offset < total:
        cursor = conn.execute(
            "SELECT scryfall_id, scryfall_json FROM cards LIMIT ? OFFSET ?",
            (batch_size, offset),
        )

        batch = []
        for row in cursor:
            try:
                card_data = json.loads(row["scryfall_json"])
                features = extract_features_from_scryfall(card_data)
                batch.append(features)
            except (json.JSONDecodeError, Exception) as e:
                continue

        # Insert batch
        for f in batch:
            conn.execute(
                """
                INSERT OR REPLACE INTO card_features
                (scryfall_id, card_name, mana_w, mana_u, mana_b, mana_r, mana_g, mana_c,
                 cmc, is_creature, is_instant, is_sorcery, is_artifact, is_enchantment,
                 is_planeswalker, is_land, is_legendary, keyword_bitmap, power, toughness,
                 color_identity_bitmap, role_ramp, role_card_draw, role_removal,
                 role_board_wipe, role_protection, role_finisher, role_utility)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f.scryfall_id,
                    f.card_name,
                    f.mana_w,
                    f.mana_u,
                    f.mana_b,
                    f.mana_r,
                    f.mana_g,
                    f.mana_c,
                    f.cmc,
                    int(f.is_creature),
                    int(f.is_instant),
                    int(f.is_sorcery),
                    int(f.is_artifact),
                    int(f.is_enchantment),
                    int(f.is_planeswalker),
                    int(f.is_land),
                    int(f.is_legendary),
                    f.keyword_bitmap,
                    f.power,
                    f.toughness,
                    f.color_identity_bitmap,
                    f.role_ramp,
                    f.role_card_draw,
                    f.role_removal,
                    f.role_board_wipe,
                    f.role_protection,
                    f.role_finisher,
                    f.role_utility,
                ),
            )

        conn.commit()
        processed += len(batch)
        offset += batch_size

        if pbar:
            pbar.update(len(batch))

    if pbar:
        pbar.close()

    conn.close()
    print(f"Extracted features for {processed:,} cards")
    return processed


def get_feature_vector(
    card_name: str,
    db_path: Optional[Path] = None,
) -> Optional[CardFeatures]:
    """
    Get features for a single card by name.

    Args:
        card_name: Exact card name
        db_path: Path to SQLite database

    Returns:
        CardFeatures object or None if not found
    """
    db_path = db_path or DEFAULT_DB_PATH

    if not db_path.exists():
        return None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.execute(
        "SELECT * FROM card_features WHERE card_name = ?",
        (card_name,),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return CardFeatures(
        scryfall_id=row["scryfall_id"],
        card_name=row["card_name"],
        mana_w=row["mana_w"],
        mana_u=row["mana_u"],
        mana_b=row["mana_b"],
        mana_r=row["mana_r"],
        mana_g=row["mana_g"],
        mana_c=row["mana_c"],
        cmc=row["cmc"],
        is_creature=bool(row["is_creature"]),
        is_instant=bool(row["is_instant"]),
        is_sorcery=bool(row["is_sorcery"]),
        is_artifact=bool(row["is_artifact"]),
        is_enchantment=bool(row["is_enchantment"]),
        is_planeswalker=bool(row["is_planeswalker"]),
        is_land=bool(row["is_land"]),
        is_legendary=bool(row["is_legendary"]),
        keyword_bitmap=row["keyword_bitmap"],
        power=row["power"],
        toughness=row["toughness"],
        color_identity_bitmap=row["color_identity_bitmap"],
        role_ramp=row["role_ramp"],
        role_card_draw=row["role_card_draw"],
        role_removal=row["role_removal"],
        role_board_wipe=row["role_board_wipe"],
        role_protection=row["role_protection"],
        role_finisher=row["role_finisher"],
        role_utility=row["role_utility"],
    )


def get_batch_features(
    card_names: list[str],
    db_path: Optional[Path] = None,
) -> dict[str, CardFeatures]:
    """
    Get features for multiple cards by name.

    Args:
        card_names: List of card names
        db_path: Path to SQLite database

    Returns:
        Dict mapping card names to CardFeatures
    """
    db_path = db_path or DEFAULT_DB_PATH

    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Use parameterized query with placeholders
    placeholders = ",".join("?" * len(card_names))
    cursor = conn.execute(
        f"SELECT * FROM card_features WHERE card_name IN ({placeholders})",
        card_names,
    )

    result = {}
    for row in cursor:
        features = CardFeatures(
            scryfall_id=row["scryfall_id"],
            card_name=row["card_name"],
            mana_w=row["mana_w"],
            mana_u=row["mana_u"],
            mana_b=row["mana_b"],
            mana_r=row["mana_r"],
            mana_g=row["mana_g"],
            mana_c=row["mana_c"],
            cmc=row["cmc"],
            is_creature=bool(row["is_creature"]),
            is_instant=bool(row["is_instant"]),
            is_sorcery=bool(row["is_sorcery"]),
            is_artifact=bool(row["is_artifact"]),
            is_enchantment=bool(row["is_enchantment"]),
            is_planeswalker=bool(row["is_planeswalker"]),
            is_land=bool(row["is_land"]),
            is_legendary=bool(row["is_legendary"]),
            keyword_bitmap=row["keyword_bitmap"],
            power=row["power"],
            toughness=row["toughness"],
            color_identity_bitmap=row["color_identity_bitmap"],
            role_ramp=row["role_ramp"],
            role_card_draw=row["role_card_draw"],
            role_removal=row["role_removal"],
            role_board_wipe=row["role_board_wipe"],
            role_protection=row["role_protection"],
            role_finisher=row["role_finisher"],
            role_utility=row["role_utility"],
        )
        result[row["card_name"]] = features

    conn.close()
    return result


def get_features_stats(db_path: Optional[Path] = None) -> dict:
    """Get statistics about the card_features table."""
    db_path = db_path or DEFAULT_DB_PATH

    if not db_path.exists():
        return {"exists": False}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Check if table exists
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='card_features'"
    )
    if not cursor.fetchone():
        conn.close()
        return {"exists": True, "features_initialized": False}

    stats = {"exists": True, "features_initialized": True}

    # Total count
    cursor = conn.execute("SELECT COUNT(*) as count FROM card_features")
    stats["total_features"] = cursor.fetchone()["count"]

    # Count by type
    cursor = conn.execute("SELECT SUM(is_creature) as count FROM card_features")
    stats["creatures"] = cursor.fetchone()["count"] or 0

    cursor = conn.execute("SELECT SUM(is_land) as count FROM card_features")
    stats["lands"] = cursor.fetchone()["count"] or 0

    # Average CMC
    cursor = conn.execute("SELECT AVG(cmc) as avg_cmc FROM card_features WHERE is_land = 0")
    stats["avg_cmc"] = round(cursor.fetchone()["avg_cmc"] or 0, 2)

    conn.close()
    return stats
