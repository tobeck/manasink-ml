"""
Scryfall API client for fetching card data.
"""

import json
import time
from pathlib import Path
from typing import Optional
import hashlib

# Note: requests is imported conditionally to allow the module to load without it
try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from src.game.card import Card

SCRYFALL_API_BASE = "https://api.scryfall.com"
CACHE_DIR = Path("data/raw/scryfall_cache")


class ScryfallClient:
    """
    Client for the Scryfall API with caching support.
    """

    def __init__(self, cache_dir: Optional[Path] = None, rate_limit_ms: int = 100):
        if not HAS_REQUESTS:
            raise ImportError("requests library required for ScryfallClient")

        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_ms = rate_limit_ms
        self._last_request_time = 0

    def _rate_limit(self) -> None:
        """Respect Scryfall's rate limits."""
        now = time.time() * 1000
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_ms:
            time.sleep((self.rate_limit_ms - elapsed) / 1000)
        self._last_request_time = time.time() * 1000

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.json"

    def _get_cached(self, key: str) -> Optional[dict]:
        """Get cached response if available."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        return None

    def _set_cached(self, key: str, data: dict) -> None:
        """Cache a response."""
        cache_path = self._get_cache_path(key)
        with open(cache_path, "w") as f:
            json.dump(data, f)

    def get_card_by_name(self, name: str, fuzzy: bool = False) -> Optional[Card]:
        """
        Fetch a card by name.

        Args:
            name: Card name to search for
            fuzzy: If True, use fuzzy matching

        Returns:
            Card object or None if not found
        """
        cache_key = f"card_name:{name}:{fuzzy}"
        cached = self._get_cached(cache_key)

        if cached:
            return Card.from_scryfall(cached)

        self._rate_limit()

        endpoint = "named"
        params = {"fuzzy" if fuzzy else "exact": name}

        response = requests.get(
            f"{SCRYFALL_API_BASE}/cards/{endpoint}",
            params=params,
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()
        data = response.json()

        self._set_cached(cache_key, data)
        return Card.from_scryfall(data)

    def get_card_by_id(self, scryfall_id: str) -> Optional[Card]:
        """Fetch a card by Scryfall ID."""
        cache_key = f"card_id:{scryfall_id}"
        cached = self._get_cached(cache_key)

        if cached:
            return Card.from_scryfall(cached)

        self._rate_limit()

        response = requests.get(f"{SCRYFALL_API_BASE}/cards/{scryfall_id}")

        if response.status_code == 404:
            return None

        response.raise_for_status()
        data = response.json()

        self._set_cached(cache_key, data)
        return Card.from_scryfall(data)

    def search_cards(
        self,
        query: str,
        unique: str = "cards",
        order: str = "name",
        max_results: int = 100,
    ) -> list[Card]:
        """
        Search for cards using Scryfall's search syntax.

        Args:
            query: Scryfall search query (e.g., "c:green t:creature cmc<=3")
            unique: How to handle reprints ("cards", "art", "prints")
            order: Sort order
            max_results: Maximum number of results to return

        Returns:
            List of Card objects
        """
        cache_key = f"search:{query}:{unique}:{order}:{max_results}"
        cached = self._get_cached(cache_key)

        if cached:
            return [Card.from_scryfall(c) for c in cached]

        cards = []
        url = f"{SCRYFALL_API_BASE}/cards/search"
        params = {"q": query, "unique": unique, "order": order}

        while url and len(cards) < max_results:
            self._rate_limit()

            response = requests.get(url, params=params)

            if response.status_code == 404:
                break  # No results

            response.raise_for_status()
            data = response.json()

            for card_data in data.get("data", []):
                if len(cards) >= max_results:
                    break
                cards.append(card_data)

            # Get next page if available
            url = data.get("next_page")
            params = {}  # Params are included in next_page URL

        self._set_cached(cache_key, cards)
        return [Card.from_scryfall(c) for c in cards]

    def get_commanders(
        self,
        colors: Optional[str] = None,
        max_results: int = 100,
    ) -> list[Card]:
        """
        Get legendary creatures that can be commanders.

        Args:
            colors: Color filter (e.g., "GU" for green-blue)
            max_results: Maximum results

        Returns:
            List of commander-legal legendary creatures
        """
        query = "is:commander"
        if colors:
            query += f" id<={colors}"

        return self.search_cards(query, max_results=max_results)

    def get_cards_for_commander(
        self,
        commander: Card,
        card_type: Optional[str] = None,
        max_cmc: Optional[int] = None,
        max_results: int = 100,
    ) -> list[Card]:
        """
        Get cards legal in a commander's color identity.

        Args:
            commander: The commander card
            card_type: Filter by type (e.g., "creature", "instant")
            max_cmc: Maximum mana value
            max_results: Maximum results

        Returns:
            List of cards legal for the commander
        """
        # Build color identity string
        color_map = {"WHITE": "W", "BLUE": "U", "BLACK": "B", "RED": "R", "GREEN": "G"}
        colors = "".join(color_map.get(c.name, "") for c in commander.color_identity)

        query = f"id<={colors}" if colors else "id:c"  # colorless if no colors
        query += " legal:commander"

        if card_type:
            query += f" t:{card_type}"

        if max_cmc is not None:
            query += f" cmc<={max_cmc}"

        return self.search_cards(query, max_results=max_results)


def download_bulk_data(output_path: Optional[Path] = None) -> Path:
    """
    Download Scryfall's bulk data file.
    This is more efficient for large-scale processing.

    Returns:
        Path to the downloaded file
    """
    if not HAS_REQUESTS:
        raise ImportError("requests library required")

    output_path = output_path or Path("data/raw/scryfall_bulk.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get bulk data info
    response = requests.get(f"{SCRYFALL_API_BASE}/bulk-data")
    response.raise_for_status()

    bulk_data = response.json()

    # Find the "oracle_cards" bulk data (one entry per card name)
    oracle_data = None
    for item in bulk_data["data"]:
        if item["type"] == "oracle_cards":
            oracle_data = item
            break

    if not oracle_data:
        raise ValueError("Could not find oracle_cards bulk data")

    # Download the bulk file
    print(f"Downloading {oracle_data['download_uri']}...")
    response = requests.get(oracle_data["download_uri"], stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded to {output_path}")
    return output_path


def load_bulk_data(path: Optional[Path] = None) -> dict[str, Card]:
    """
    Load cards from bulk data file.

    Returns:
        Dict mapping card names to Card objects
    """
    path = path or Path("data/raw/scryfall_bulk.json")

    if not path.exists():
        raise FileNotFoundError(f"Bulk data not found at {path}. Run download_bulk_data() first.")

    with open(path) as f:
        data = json.load(f)

    cards = {}
    for card_data in data:
        try:
            card = Card.from_scryfall(card_data)
            cards[card.name] = card
        except Exception as e:
            # Skip cards that fail to parse
            pass

    return cards


# Convenience functions


def fetch_card(name: str) -> Optional[Card]:
    """Quick helper to fetch a single card by name."""
    client = ScryfallClient()
    return client.get_card_by_name(name, fuzzy=True)


def fetch_commander(name: str) -> Optional[Card]:
    """Fetch a commander by name and validate it's legal as a commander."""
    card = fetch_card(name)
    if card and card.is_commander:
        return card
    return None
