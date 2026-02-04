"""
CLI for the data service.

Provides commands for managing the card database:
- sync: Download card data and populate database
- stats: Show database statistics
- search: Search for cards
- card: Get details for a specific card

Usage:
    manasink-data sync          # Download and populate database
    manasink-data sync --force  # Force re-download
    manasink-data stats         # Show database statistics
    manasink-data search "Sol Ring"  # Search by name
    manasink-data card "Sol Ring"    # Get card details
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .database import CardDatabase, DEFAULT_DB_PATH
from .ingest import sync_database, get_database_stats, DEFAULT_BULK_PATH
from .edhrec_ingest import (
    sync_edhrec_data,
    get_edhrec_stats,
    get_commander_recommendations,
    get_salt_scores_from_db,
    estimate_deck_power,
)
from .features import populate_card_features, get_features_stats
from .categories import populate_card_categories, get_categories_stats


def cmd_sync(args: argparse.Namespace) -> int:
    """Sync command: download and populate database."""
    print("Manasink Data Service - Sync")
    print("=" * 40)

    bulk_path = Path(args.bulk_path) if args.bulk_path else DEFAULT_BULK_PATH
    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB_PATH

    try:
        result = sync_database(
            bulk_path=bulk_path,
            db_path=db_path,
            force=args.force,
            show_progress=not args.quiet,
        )

        if result["skipped"]:
            return 0

        print(f"\nSync completed successfully!")
        return 0

    except Exception as e:
        print(f"\nError during sync: {e}", file=sys.stderr)
        return 1


def cmd_stats(args: argparse.Namespace) -> int:
    """Stats command: show database statistics."""
    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB_PATH

    stats = get_database_stats(db_path)

    if not stats.get("exists"):
        print(f"Database not found at {db_path}")
        print("Run 'manasink-data sync' to download card data.")
        return 1

    print("Manasink Card Database Statistics")
    print("=" * 40)
    print(f"Database: {stats['path']}")
    print(f"Size: {stats['file_size_mb']:.1f} MB")
    print()
    print(f"Total cards: {stats['total_cards']:,}")
    print(f"Commander-legal: {stats['commander_legal']:,}")
    print(f"Potential commanders: {stats['commanders']:,}")
    print()
    print("By type:")
    print(f"  Creatures: {stats['type_creature']:,}")
    print(f"  Instants: {stats['type_instant']:,}")
    print(f"  Sorceries: {stats['type_sorcery']:,}")
    print(f"  Artifacts: {stats['type_artifact']:,}")
    print(f"  Enchantments: {stats['type_enchantment']:,}")
    print(f"  Lands: {stats['type_land']:,}")
    print(f"  Planeswalkers: {stats['type_planeswalker']:,}")
    print()
    if stats.get("last_updated"):
        print(f"Last synced: {stats['last_updated']}")
    if stats.get("scryfall_updated_at"):
        print(f"Scryfall data: {stats['scryfall_updated_at']}")

    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Search command: search for cards."""
    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB_PATH

    try:
        db = CardDatabase(db_path)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    # Build search kwargs
    kwargs = {"limit": args.limit}

    if args.query:
        kwargs["name_contains"] = args.query

    if args.type:
        kwargs["card_types"] = [args.type]

    if args.colors:
        kwargs["color_identity"] = args.colors

    if args.max_cmc is not None:
        kwargs["max_cmc"] = args.max_cmc

    if args.commander:
        kwargs["is_commander"] = True
        kwargs["is_legal_commander"] = True

    if args.text:
        kwargs["text_contains"] = args.text

    cards = db.search(**kwargs)
    db.close()

    if not cards:
        print("No cards found matching criteria.")
        return 0

    print(f"Found {len(cards)} cards:")
    print()

    for card in cards:
        if card.is_creature:
            print(f"  {card.name} ({card.cmc}mv) - {card.power}/{card.toughness}")
        else:
            types = ", ".join(t.name.lower() for t in card.card_types)
            print(f"  {card.name} ({card.cmc}mv) - {types}")

    return 0


def cmd_card(args: argparse.Namespace) -> int:
    """Card command: get details for a specific card."""
    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB_PATH

    try:
        db = CardDatabase(db_path)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    card = db.get_card(args.name)

    if not card:
        # Try partial match
        cards = db.search(name_contains=args.name, limit=5)
        db.close()

        if cards:
            print(f"Card '{args.name}' not found. Did you mean:")
            for c in cards:
                print(f"  - {c.name}")
        else:
            print(f"Card '{args.name}' not found.")
        return 1

    if args.json:
        # Output raw Scryfall JSON
        raw = db.get_scryfall_json(args.name)
        db.close()
        print(json.dumps(raw, indent=2))
        return 0

    db.close()

    # Pretty print card details
    print()
    print(card.name)
    print("=" * len(card.name))
    print(f"Mana Cost: {card.mana_cost.cmc}mv")
    print(f"Types: {', '.join(t.name.lower() for t in card.card_types)}")

    if card.is_creature:
        print(f"Power/Toughness: {card.power}/{card.toughness}")

    if card.color_identity:
        colors = ", ".join(c.name.lower() for c in card.color_identity)
        print(f"Color Identity: {colors}")

    if card.oracle_text:
        print()
        print("Oracle Text:")
        for line in card.oracle_text.split("\n"):
            print(f"  {line}")

    if card.keywords:
        print()
        print(f"Keywords: {', '.join(sorted(card.keywords))}")

    if card.is_commander:
        print()
        print("* Can be your commander")

    return 0


def cmd_commanders(args: argparse.Namespace) -> int:
    """List commanders, optionally filtered by color."""
    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB_PATH

    try:
        db = CardDatabase(db_path)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    commanders = db.get_commanders(colors=args.colors, limit=args.limit)
    db.close()

    if not commanders:
        print("No commanders found.")
        return 0

    color_desc = f" ({args.colors})" if args.colors else ""
    print(f"Found {len(commanders)} commanders{color_desc}:")
    print()

    for card in commanders:
        identity = "".join(c.value for c in card.color_identity if c.value != "C")
        identity_str = f"[{identity}]" if identity else "[C]"
        print(f"  {identity_str:8} {card.name} ({card.cmc}mv) - {card.power}/{card.toughness}")

    return 0


def cmd_edhrec_sync(args: argparse.Namespace) -> int:
    """Sync EDHREC data to the database."""
    print("Manasink Data Service - EDHREC Sync")
    print("=" * 40)

    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB_PATH

    try:
        result = sync_edhrec_data(
            db_path=db_path,
            limit=args.limit,
            force=args.force,
            show_progress=not args.quiet,
        )

        if result.get("skipped"):
            return 0

        print(f"\nEDHREC sync completed successfully!")
        return 0

    except Exception as e:
        print(f"\nError during EDHREC sync: {e}", file=sys.stderr)
        return 1


def cmd_edhrec_stats(args: argparse.Namespace) -> int:
    """Show EDHREC data statistics."""
    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB_PATH

    stats = get_edhrec_stats(db_path)

    if not stats.get("exists"):
        print(f"Database not found at {db_path}")
        print("Run 'manasink-data sync' to download card data.")
        return 1

    if not stats.get("edhrec_initialized"):
        print("EDHREC data not yet synced.")
        print("Run 'manasink-data edhrec-sync' to fetch EDHREC data.")
        return 1

    print("Manasink EDHREC Data Statistics")
    print("=" * 40)
    print(f"Database: {stats['path']}")
    print()
    print(f"Commanders synced: {stats['commanders']:,}")
    print(f"Card recommendations: {stats['recommendations']:,}")
    print(f"Average deck cards: {stats['average_deck_cards']:,}")
    print(f"Salt scores: {stats['salt_scores']:,}")
    print()

    if stats.get("top_commanders"):
        print("Top commanders by popularity:")
        for cmd in stats["top_commanders"]:
            print(f"  #{cmd['rank'] or '?':3} {cmd['name']} ({cmd['num_decks']:,} decks)")
        print()

    if stats.get("highest_salt"):
        print("Highest salt cards:")
        for card in stats["highest_salt"]:
            print(f"  #{card['rank']:3} {card['name']} (salt: {card['salt']:.2f})")
        print()

    if stats.get("last_updated"):
        print(f"Last synced: {stats['last_updated']}")

    return 0


def cmd_edhrec_recs(args: argparse.Namespace) -> int:
    """Get EDHREC recommendations for a commander."""
    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB_PATH

    recs = get_commander_recommendations(
        commander_name=args.name,
        db_path=db_path,
        limit=args.limit,
        min_synergy=args.min_synergy,
        category=args.category,
    )

    if not recs:
        print(f"No recommendations found for '{args.name}'")
        print("Make sure EDHREC data is synced: manasink-data edhrec-sync")
        return 1

    print(f"Recommendations for {args.name}")
    print("=" * 50)
    print()

    if args.json:
        print(json.dumps(recs, indent=2))
        return 0

    # Group by category if not filtered
    if args.category:
        for rec in recs:
            synergy = rec["synergy_score"]
            inclusion = rec["inclusion_rate"]
            print(f"  {rec['card_name']}")
            print(f"    Synergy: {synergy:+.0%}  Inclusion: {inclusion:.0f}%")
    else:
        # Group by category
        categories = {}
        for rec in recs:
            cat = rec["category"] or "Other"
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(rec)

        for cat, cat_recs in categories.items():
            print(f"{cat}:")
            for rec in cat_recs[:5]:  # Top 5 per category
                synergy = rec["synergy_score"]
                inclusion = rec["inclusion_rate"]
                print(f"  {rec['card_name']:40} syn:{synergy:+.0%} inc:{inclusion:.0f}%")
            print()

    return 0


def cmd_extract_features(args: argparse.Namespace) -> int:
    """Extract ML features from card data."""
    print("Manasink Data Service - Extract Features")
    print("=" * 40)

    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB_PATH

    try:
        count = populate_card_features(
            db_path=db_path,
            batch_size=args.batch_size,
            show_progress=not args.quiet,
        )

        print(f"\nExtracted features for {count:,} cards")
        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        print("Run 'manasink-data sync' first to download card data.")
        return 1
    except Exception as e:
        print(f"\nError during feature extraction: {e}", file=sys.stderr)
        return 1


def cmd_aggregate_categories(args: argparse.Namespace) -> int:
    """Aggregate card categories from EDHREC data."""
    print("Manasink Data Service - Aggregate Categories")
    print("=" * 40)

    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB_PATH

    try:
        count = populate_card_categories(
            db_path=db_path,
            show_progress=not args.quiet,
        )

        if count == 0:
            print("\nNo categories to aggregate.")
            print("Run 'manasink-data edhrec-sync' first to fetch EDHREC data.")
            return 1

        print(f"\nAggregated {count:,} category entries")
        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nError during category aggregation: {e}", file=sys.stderr)
        return 1


def cmd_features_stats(args: argparse.Namespace) -> int:
    """Show feature extraction statistics."""
    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB_PATH

    stats = get_features_stats(db_path)

    if not stats.get("exists"):
        print(f"Database not found at {db_path}")
        return 1

    if not stats.get("features_initialized"):
        print("Card features not yet extracted.")
        print("Run 'manasink-data extract-features' first.")
        return 1

    print("Manasink Card Features Statistics")
    print("=" * 40)
    print(f"Total cards with features: {stats['total_features']:,}")
    print(f"Creatures: {stats['creatures']:,}")
    print(f"Lands: {stats['lands']:,}")
    print(f"Average CMC (non-land): {stats['avg_cmc']:.2f}")

    return 0


def cmd_categories_stats(args: argparse.Namespace) -> int:
    """Show category aggregation statistics."""
    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB_PATH

    stats = get_categories_stats(db_path)

    if not stats.get("exists"):
        print(f"Database not found at {db_path}")
        return 1

    if not stats.get("categories_initialized"):
        print("Card categories not yet aggregated.")
        print("Run 'manasink-data aggregate-categories' first.")
        return 1

    print("Manasink Card Categories Statistics")
    print("=" * 40)
    print(f"Total category entries: {stats['total_entries']:,}")
    print(f"Unique cards: {stats['unique_cards']:,}")
    print(f"Unique categories: {stats['unique_categories']:,}")
    print()

    if stats.get("top_categories"):
        print("Top categories by card count:")
        for cat in stats["top_categories"]:
            print(f"  {cat['category']:20} {cat['card_count']:,} cards")

    return 0


def cmd_edhrec_power(args: argparse.Namespace) -> int:
    """Estimate power level for a deck based on salt scores."""
    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB_PATH

    # Read deck list from file or stdin
    if args.deck_file:
        deck_path = Path(args.deck_file)
        if not deck_path.exists():
            print(f"Deck file not found: {args.deck_file}", file=sys.stderr)
            return 1
        with open(deck_path) as f:
            lines = f.readlines()
    else:
        print("Enter card names (one per line, Ctrl+D when done):")
        lines = sys.stdin.readlines()

    # Parse card names (handle "1x Card Name" format)
    cards = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Remove quantity prefix if present
        parts = line.split(" ", 1)
        if parts[0].rstrip("x").isdigit() and len(parts) > 1:
            cards.append(parts[1])
        else:
            cards.append(line)

    if not cards:
        print("No cards provided.", file=sys.stderr)
        return 1

    # Get salt scores
    salt_scores = get_salt_scores_from_db(db_path)
    if not salt_scores:
        print("Salt scores not found. Run 'manasink-data edhrec-sync' first.")
        return 1

    # Estimate power level
    estimate = estimate_deck_power(cards, salt_scores)

    print(f"\nPower Level Estimate for {len(cards)} cards")
    print("=" * 40)
    print(f"Salt Sum: {estimate.salt_sum:.1f}")
    print(f"Bracket: {estimate.bracket} ({estimate.description})")
    print(f"Power Score: {estimate.power_score}/10")
    print()

    # Show high-salt cards in the deck
    deck_salt = [(card, salt_scores.get(card, 0)) for card in cards]
    deck_salt.sort(key=lambda x: x[1], reverse=True)

    high_salt = [(c, s) for c, s in deck_salt if s > 1.5]
    if high_salt:
        print("High-salt cards in deck:")
        for card, salt in high_salt[:10]:
            print(f"  {card}: {salt:.2f}")

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="manasink-data",
        description="Manasink data service for managing MTG card database",
    )
    parser.add_argument(
        "--db-path",
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Download and populate card database")
    sync_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if up to date",
    )
    sync_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress bars",
    )
    sync_parser.add_argument(
        "--bulk-path",
        help=f"Path for bulk JSON file (default: {DEFAULT_BULK_PATH})",
    )

    # Stats command
    subparsers.add_parser("stats", help="Show database statistics")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for cards")
    search_parser.add_argument("query", nargs="?", help="Card name to search for")
    search_parser.add_argument("--type", "-t", help="Filter by card type")
    search_parser.add_argument("--colors", "-c", help="Filter by color identity (e.g., UG)")
    search_parser.add_argument("--max-cmc", type=int, help="Maximum mana value")
    search_parser.add_argument("--commander", action="store_true", help="Only show commanders")
    search_parser.add_argument("--text", help="Search oracle text")
    search_parser.add_argument("--limit", "-l", type=int, default=20, help="Max results")

    # Card command
    card_parser = subparsers.add_parser("card", help="Get details for a specific card")
    card_parser.add_argument("name", help="Card name")
    card_parser.add_argument("--json", "-j", action="store_true", help="Output raw JSON")

    # Commanders command
    commanders_parser = subparsers.add_parser("commanders", help="List commanders")
    commanders_parser.add_argument("--colors", "-c", help="Filter by color identity")
    commanders_parser.add_argument("--limit", "-l", type=int, default=50, help="Max results")

    # EDHREC sync command
    edhrec_sync_parser = subparsers.add_parser(
        "edhrec-sync", help="Sync EDHREC commander data"
    )
    edhrec_sync_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=100,
        help="Number of top commanders to sync (default: 100)",
    )
    edhrec_sync_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-sync even if recently updated",
    )
    edhrec_sync_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress bars",
    )

    # EDHREC stats command
    subparsers.add_parser("edhrec-stats", help="Show EDHREC data statistics")

    # EDHREC recommendations command
    edhrec_recs_parser = subparsers.add_parser(
        "edhrec-recs", help="Get EDHREC recommendations for a commander"
    )
    edhrec_recs_parser.add_argument("name", help="Commander name")
    edhrec_recs_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=50,
        help="Max recommendations to show",
    )
    edhrec_recs_parser.add_argument(
        "--min-synergy",
        type=float,
        help="Minimum synergy score (e.g., 0.1 for 10%%)",
    )
    edhrec_recs_parser.add_argument(
        "--category", "-c",
        help="Filter by category (e.g., Creatures, Ramp)",
    )
    edhrec_recs_parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON",
    )

    # EDHREC power level command
    edhrec_power_parser = subparsers.add_parser(
        "edhrec-power", help="Estimate deck power level from salt scores"
    )
    edhrec_power_parser.add_argument(
        "deck_file",
        nargs="?",
        help="Path to deck list file (or stdin if not provided)",
    )

    # Extract features command
    features_parser = subparsers.add_parser(
        "extract-features", help="Extract ML features from card data"
    )
    features_parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of cards to process per batch (default: 1000)",
    )
    features_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress bars",
    )

    # Features stats command
    subparsers.add_parser("features-stats", help="Show feature extraction statistics")

    # Aggregate categories command
    categories_parser = subparsers.add_parser(
        "aggregate-categories", help="Aggregate card categories from EDHREC data"
    )
    categories_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress bars",
    )

    # Categories stats command
    subparsers.add_parser("categories-stats", help="Show category aggregation statistics")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "sync": cmd_sync,
        "stats": cmd_stats,
        "search": cmd_search,
        "card": cmd_card,
        "commanders": cmd_commanders,
        "edhrec-sync": cmd_edhrec_sync,
        "edhrec-stats": cmd_edhrec_stats,
        "edhrec-recs": cmd_edhrec_recs,
        "edhrec-power": cmd_edhrec_power,
        "extract-features": cmd_extract_features,
        "features-stats": cmd_features_stats,
        "aggregate-categories": cmd_aggregate_categories,
        "categories-stats": cmd_categories_stats,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
