"""
EDHREC API client for fetching commander recommendations and synergy data.

EDHREC provides:
- Top commanders with popularity rankings
- Card recommendations per commander (with inclusion rates and synergy scores)
- Average decklists (consensus 99)
- Salt scores for power level estimation
"""

import hashlib
import json
import re
import time
from pathlib import Path

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


EDHREC_JSON_BASE = "https://json.edhrec.com/pages"
DEFAULT_CACHE_DIR = Path("data/raw/edhrec_cache")

# Headers to mimic browser requests (EDHREC blocks bare requests)
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://edhrec.com/",
}

# Color combinations to query for commanders (most popular combinations)
# Note: /commanders.json is blocked, so we aggregate from color pages
COLOR_COMBINATIONS = [
    "w",
    "u",
    "b",
    "r",
    "g",  # mono
    "wu",
    "wb",
    "wr",
    "wg",  # white pairs
    "ub",
    "ur",
    "ug",  # blue pairs
    "br",
    "bg",  # black pairs
    "rg",  # red-green
    "wub",
    "wur",
    "wug",
    "wbr",
    "wbg",
    "wrg",  # 3-color with white
    "ubr",
    "ubg",
    "urg",
    "brg",  # 3-color without white
    "wubr",
    "wubg",
    "wurg",
    "wbrg",
    "ubrg",  # 4-color
    "wubrg",  # 5-color
    "colorless",  # colorless
]


class EDHRecClient:
    """
    Client for EDHREC's JSON API with caching support.

    EDHREC doesn't have a documented API, but exposes JSON endpoints
    that power their website. This client accesses those endpoints
    with appropriate rate limiting and caching.

    Example:
        client = EDHRecClient()
        commanders = client.get_top_commanders(limit=100)
        data = client.get_commander_data("atraxa-praetors-voice")
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        rate_limit_ms: int = 150,
    ):
        if not HAS_REQUESTS:
            raise ImportError("requests library required for EDHRecClient")

        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_ms = rate_limit_ms
        self._last_request_time = 0

    def _rate_limit(self) -> None:
        """Respect rate limits between requests."""
        now = time.time() * 1000
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_ms:
            time.sleep((self.rate_limit_ms - elapsed) / 1000)
        self._last_request_time = time.time() * 1000

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.json"

    def _get_cached(self, key: str) -> dict | None:
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

    def _clear_cache(self, key: str) -> None:
        """Clear a specific cache entry."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()

    def _fetch(self, url: str, cache_key: str, use_cache: bool = True) -> dict | None:
        """
        Fetch JSON from URL with caching and rate limiting.

        Args:
            url: Full URL to fetch
            cache_key: Key for caching the response
            use_cache: Whether to use cached data

        Returns:
            Parsed JSON response or None if not found
        """
        if use_cache:
            cached = self._get_cached(cache_key)
            if cached:
                return cached

        self._rate_limit()

        try:
            response = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)

            if response.status_code == 404:
                return None

            if response.status_code == 403:
                # EDHREC may block requests; return None gracefully
                return None

            response.raise_for_status()
            data = response.json()

            self._set_cached(cache_key, data)
            return data

        except requests.RequestException:
            return None

    @staticmethod
    def name_to_slug(name: str) -> str:
        """
        Convert a commander name to EDHREC's slug format.

        Examples:
            "Atraxa, Praetors' Voice" -> "atraxa-praetors-voice"
            "Urza, Lord High Artificer" -> "urza-lord-high-artificer"
        """
        # Lowercase
        slug = name.lower()
        # Remove apostrophes and commas
        slug = slug.replace("'", "").replace(",", "")
        # Replace spaces and special chars with hyphens
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        # Remove leading/trailing hyphens
        slug = slug.strip("-")
        # Collapse multiple hyphens
        slug = re.sub(r"-+", "-", slug)
        return slug

    def get_top_commanders(
        self,
        limit: int = 100,
        use_cache: bool = True,
        colors: list[str] | None = None,
    ) -> list[dict]:
        """
        Get the top commanders by popularity.

        Fetches commanders from color-specific pages and aggregates results.
        The main /commanders.json endpoint is blocked, so we use color pages.

        Args:
            limit: Maximum number of commanders to return
            use_cache: Use cached data if available
            colors: Specific color combinations to fetch (default: all major combinations)

        Returns:
            List of commander dicts with keys: name, slug, num_decks, etc.
        """
        colors_to_fetch = colors or COLOR_COMBINATIONS
        all_commanders = {}  # Use dict to dedupe by name

        for color in colors_to_fetch:
            url = f"{EDHREC_JSON_BASE}/commanders/{color}.json"
            cache_key = f"commanders_color:{color}"

            data = self._fetch(url, cache_key, use_cache)
            if not data:
                continue

            # Extract commanders from cardlists
            container = data.get("container", {})
            json_dict = container.get("json_dict", {})
            cardlists = json_dict.get("cardlists", [])

            for cardlist in cardlists:
                for card in cardlist.get("cardviews", []):
                    name = card.get("name")
                    if name and name not in all_commanders:
                        # Normalize the card data
                        all_commanders[name] = {
                            "name": name,
                            "slug": card.get("sanitized", self.name_to_slug(name)),
                            "num_decks": card.get("num_decks", 0),
                            "color_identity": color.upper() if color != "colorless" else "C",
                        }

            # Early exit if we have enough
            if len(all_commanders) >= limit * 2:
                break

        # Sort by number of decks and return top N
        sorted_commanders = sorted(
            all_commanders.values(),
            key=lambda c: c.get("num_decks", 0),
            reverse=True,
        )

        return sorted_commanders[:limit]

    def get_commander_data(
        self,
        name: str,
        use_cache: bool = True,
    ) -> dict | None:
        """
        Get full commander data including card recommendations.

        Args:
            name: Commander name (will be converted to slug)
            use_cache: Use cached data if available

        Returns:
            Dict with commander info and recommendations, or None if not found
        """
        slug = self.name_to_slug(name)
        url = f"{EDHREC_JSON_BASE}/commanders/{slug}.json"
        cache_key = f"commander:{slug}"

        data = self._fetch(url, cache_key, use_cache)
        if not data:
            return None

        # Extract relevant data
        container = data.get("container", {})
        json_dict = container.get("json_dict", {})

        result = {
            "name": name,
            "slug": slug,
            "raw_data": data,
        }

        # Get card recommendations from cardlists
        cardlists = json_dict.get("cardlists", [])
        recommendations = []

        # Get total decks for this commander to calculate inclusion rate
        card_info = json_dict.get("card", {})
        total_decks = card_info.get("num_decks", 1)  # Avoid div by zero

        for cardlist in cardlists:
            category = cardlist.get("header", "Unknown")
            cards = cardlist.get("cardviews", [])

            for card in cards:
                inclusion = card.get("inclusion", 0)
                potential = card.get("potential_decks", total_decks)
                # Calculate inclusion rate as percentage (0-100)
                inclusion_rate = (inclusion / potential * 100) if potential > 0 else 0

                rec = {
                    "card_name": card.get("name", ""),
                    "inclusion_rate": round(inclusion_rate, 1),
                    "synergy_score": card.get("synergy", 0),
                    "num_decks": card.get("num_decks", 0),
                    "category": category,
                }
                recommendations.append(rec)

        result["recommendations"] = recommendations

        # Get commander metadata
        card_info = json_dict.get("card", {})
        result["edhrec_rank"] = card_info.get("rank")
        result["num_decks"] = card_info.get("num_decks", 0)
        result["salt_score"] = card_info.get("salt")
        result["color_identity"] = card_info.get("color_identity", [])

        return result

    def get_average_deck(
        self,
        name: str,
        use_cache: bool = True,
    ) -> list[str]:
        """
        Get the average decklist for a commander (consensus 99).

        Args:
            name: Commander name (will be converted to slug)
            use_cache: Use cached data if available

        Returns:
            List of card names in the average deck
        """
        slug = self.name_to_slug(name)
        url = f"{EDHREC_JSON_BASE}/average-decks/{slug}.json"
        cache_key = f"avgdeck:{slug}"

        data = self._fetch(url, cache_key, use_cache)
        if not data:
            return []

        # Extract card names from the deck data
        container = data.get("container", {})
        json_dict = container.get("json_dict", {})

        cards = []

        # Get cards from various sections
        cardlists = json_dict.get("cardlists", [])
        for cardlist in cardlists:
            for card in cardlist.get("cardviews", []):
                card_name = card.get("name")
                if card_name:
                    cards.append(card_name)

        return cards

    def get_salt_scores(
        self,
        use_cache: bool = True,
    ) -> dict[str, float]:
        """
        Get global salt scores for cards.

        Salt score indicates how "annoying" a card is to play against,
        on a scale of 0-4. Higher salt = more powerful/frustrating.

        Returns:
            Dict mapping card names to salt scores
        """
        url = f"{EDHREC_JSON_BASE}/top/salt.json"
        cache_key = "salt_scores"

        data = self._fetch(url, cache_key, use_cache)
        if not data:
            return {}

        container = data.get("container", {})
        json_dict = container.get("json_dict", {})
        cardlists = json_dict.get("cardlists", [])

        salt_scores = {}

        for cardlist in cardlists:
            for card in cardlist.get("cardviews", []):
                name = card.get("name")
                if not name:
                    continue

                # Salt may be in 'salt' field or parsed from 'label'
                salt = card.get("salt")
                if salt is None:
                    # Parse from label: "Salt Score: 3.06\n14814 decks"
                    label = card.get("label", "")
                    match = re.search(r"Salt Score:\s*([\d.]+)", label)
                    if match:
                        salt = float(match.group(1))

                if salt is not None:
                    salt_scores[name] = float(salt)

        return salt_scores


# Convenience functions


def fetch_commander_recommendations(name: str) -> dict | None:
    """Quick helper to fetch recommendations for a commander."""
    client = EDHRecClient()
    return client.get_commander_data(name)


def fetch_salt_scores() -> dict[str, float]:
    """Quick helper to fetch global salt scores."""
    client = EDHRecClient()
    return client.get_salt_scores()
