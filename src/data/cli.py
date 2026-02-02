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
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
