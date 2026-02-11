#!/usr/bin/env python3
"""Quick sync script that creates tables and adds sample data for testing."""
import json
import os

os.environ.setdefault(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/manasink"
)

from src.data.db_config import DatabaseManager
from src.data.db_models import AverageDeckCard, CardModel, CommanderModel

# Create tables
db = DatabaseManager()
db.create_tables()
print("Tables created!")

# Add a sample commander
session = db.session()

# Check if we already have data
if session.query(CardModel).count() > 0:
    print(f"Already have {session.query(CardModel).count()} cards")
    session.close()
    exit()

# Add sample cards - need scryfall_json (NOT NULL)
sample_cards = [
    {
        "scryfall_id": "9939f21a-2a31-4f91-8c42-f4c01dcf32d1",
        "name": "Sol Ring",
        "name_lower": "sol ring",
        "mana_cost": "{1}",
        "cmc": 1,
        "type_line": "Artifact",
        "oracle_text": "{T}: Add {C}{C}.",
        "legal_commander": True,
        "scryfall_json": json.dumps({"name": "Sol Ring", "cmc": 1}),
    },
    {
        "scryfall_id": "74c89ce5-cbb4-4667-8f26-0d52234c6c5f",
        "name": "Command Tower",
        "name_lower": "command tower",
        "mana_cost": "",
        "cmc": 0,
        "type_line": "Land",
        "oracle_text": "{T}: Add one mana of any color in your commander's color identity.",
        "legal_commander": True,
        "scryfall_json": json.dumps({"name": "Command Tower", "cmc": 0}),
    },
    {
        "scryfall_id": "cab5d1e6-f3f4-4f52-8ab9-a27c9c9b75d1",
        "name": "Arcane Signet",
        "name_lower": "arcane signet",
        "mana_cost": "{2}",
        "cmc": 2,
        "type_line": "Artifact",
        "oracle_text": "{T}: Add one mana of any color in your commander's color identity.",
        "legal_commander": True,
        "scryfall_json": json.dumps({"name": "Arcane Signet", "cmc": 2}),
    },
    {
        "scryfall_id": "f8bfaf27-5ee4-4f6c-a64d-2e69c6dc1f10",
        "name": "Forest",
        "name_lower": "forest",
        "mana_cost": "",
        "cmc": 0,
        "type_line": "Basic Land — Forest",
        "oracle_text": "({T}: Add {G}.)",
        "legal_commander": True,
        "scryfall_json": json.dumps({"name": "Forest", "cmc": 0}),
    },
    {
        "scryfall_id": "4b9b3f33-3a9c-4e4c-8c71-6a50d4b2da04",
        "name": "Llanowar Elves",
        "name_lower": "llanowar elves",
        "mana_cost": "{G}",
        "cmc": 1,
        "type_line": "Creature — Elf Druid",
        "oracle_text": "{T}: Add {G}.",
        "power": "1",
        "toughness": "1",
        "legal_commander": True,
        "scryfall_json": json.dumps({"name": "Llanowar Elves", "cmc": 1}),
    },
    {
        "scryfall_id": "a1f95b7a-4b5c-4c6d-9a1e-2b3c4d5e6f7a",
        "name": "Birds of Paradise",
        "name_lower": "birds of paradise",
        "mana_cost": "{G}",
        "cmc": 1,
        "type_line": "Creature — Bird",
        "oracle_text": "Flying\n{T}: Add one mana of any color.",
        "power": "0",
        "toughness": "1",
        "legal_commander": True,
        "scryfall_json": json.dumps({"name": "Birds of Paradise", "cmc": 1}),
    },
    {
        "scryfall_id": "b2c3d4e5-f6a7-8b9c-0d1e-2f3a4b5c6d7e",
        "name": "Cultivate",
        "name_lower": "cultivate",
        "mana_cost": "{2}{G}",
        "cmc": 3,
        "type_line": "Sorcery",
        "oracle_text": "Search your library for up to two basic land cards, reveal those cards, and put one onto the battlefield tapped and the other into your hand, then shuffle.",
        "legal_commander": True,
        "scryfall_json": json.dumps({"name": "Cultivate", "cmc": 3}),
    },
    {
        "scryfall_id": "c3d4e5f6-a7b8-9c0d-1e2f-3a4b5c6d7e8f",
        "name": "Kodama's Reach",
        "name_lower": "kodama's reach",
        "mana_cost": "{2}{G}",
        "cmc": 3,
        "type_line": "Sorcery — Arcane",
        "oracle_text": "Search your library for up to two basic land cards, reveal those cards, and put one onto the battlefield tapped and the other into your hand, then shuffle.",
        "legal_commander": True,
        "scryfall_json": json.dumps({"name": "Kodama's Reach", "cmc": 3}),
    },
    {
        "scryfall_id": "d4e5f6a7-b8c9-0d1e-2f3a-4b5c6d7e8f9a",
        "name": "Rampant Growth",
        "name_lower": "rampant growth",
        "mana_cost": "{1}{G}",
        "cmc": 2,
        "type_line": "Sorcery",
        "oracle_text": "Search your library for a basic land card, put that card onto the battlefield tapped, then shuffle.",
        "legal_commander": True,
        "scryfall_json": json.dumps({"name": "Rampant Growth", "cmc": 2}),
    },
    {
        "scryfall_id": "e5f6a7b8-c9d0-1e2f-3a4b-5c6d7e8f9a0b",
        "name": "Nature's Lore",
        "name_lower": "nature's lore",
        "mana_cost": "{1}{G}",
        "cmc": 2,
        "type_line": "Sorcery",
        "oracle_text": "Search your library for a Forest card, put that card onto the battlefield, then shuffle.",
        "legal_commander": True,
        "scryfall_json": json.dumps({"name": "Nature's Lore", "cmc": 2}),
    },
]

# Add more lands to fill out a minimal deck
for i in range(20):
    sample_cards.append({
        "scryfall_id": f"forest-{i:03d}",
        "name": f"Forest",
        "name_lower": "forest",
        "mana_cost": "",
        "cmc": 0,
        "type_line": "Basic Land — Forest",
        "oracle_text": "({T}: Add {G}.)",
        "legal_commander": True,
        "scryfall_json": json.dumps({"name": "Forest", "cmc": 0}),
    })

for card_data in sample_cards:
    card = CardModel(**card_data)
    session.add(card)

# Add sample commander - needs name_slug (NOT NULL)
commander = CommanderModel(
    name="Test Commander",
    name_slug="test-commander",
    color_identity="G",
    num_decks=1000,
)
session.add(commander)
session.commit()

# Add sample deck (use unique card names only)
commander_id = commander.id
seen_names = set()
slot = 0
for card_data in sample_cards:
    if card_data["name"] not in seen_names:
        seen_names.add(card_data["name"])
        deck_card = AverageDeckCard(
            commander_id=commander_id,
            card_name=card_data["name"],
            slot_number=slot,
        )
        session.add(deck_card)
        slot += 1

session.commit()
print(f"Added {len(sample_cards)} sample cards and 1 commander with {slot} deck cards")
session.close()
