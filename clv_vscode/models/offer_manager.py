import json
import os
import uuid

OFFERS_PATH = os.path.join("data", "offers.json")

def load_offers():
    if not os.path.exists(OFFERS_PATH):
        return []
    return json.load(open(OFFERS_PATH))

def save_offers(offers):
    with open(OFFERS_PATH, "w") as f:
        json.dump(offers, f, indent=2)

def add_offer(segment, title, description, category):
    offers = load_offers()
    offers.append({
        "id": str(uuid.uuid4()),
        "segment": segment,
        "title": title,
        "description": description,
        "category": category,
        "active": True
    })
    save_offers(offers)

def update_offer(offer_id, updated_fields):
    offers = load_offers()
    for o in offers:
        if o["id"] == offer_id:
            o.update(updated_fields)
    save_offers(offers)

def delete_offer(offer_id):
    offers = [o for o in load_offers() if o["id"] != offer_id]
    save_offers(offers)