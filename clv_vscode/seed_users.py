"""
seed_users.py
-------------
Run once to seed your MongoDB with default users.
Usage: python seed_users.py
"""

import hashlib, os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.environ.get("MONGO_URI", "")

if not MONGO_URI:
    print("ERROR: MONGO_URI not found in .env file")
    print("Make sure your .env file contains:")
    print("  MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/clv_db")
    exit(1)

def h(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

try:
    from pymongo import MongoClient
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client["clv_db"]
    print("✅ Connected to MongoDB!")
except Exception as e:
    print(f"❌ Could not connect to MongoDB: {e}")
    print("\nCommon fixes:")
    print("  1. Check your MONGO_URI in .env is correct")
    print("  2. Go to MongoDB Atlas → Network Access → Add your current IP")
    print("  3. Make sure your cluster is running (not paused)")
    exit(1)

users = [
    {"username": "admin",   "name": "Admin User", "password": h("admin123"),   "role": "admin"},
    {"username": "aditi",   "name": "Aditi",       "password": h("aditi123"),   "role": "admin"},
    {"username": "shreyas", "name": "Shreyas",     "password": h("shreyas123"), "role": "admin"},
    {"username": "viewer",  "name": "Viewer",      "password": h("view123"),    "role": "viewer"},
]

for u in users:
    db["users"].update_one(
        {"username": u["username"]},
        {"$set": u},
        upsert=True
    )

print(f"\n✅ {len(users)} users seeded into MongoDB")
print("\nUsers in database:")
for u in db["users"].find({}, {"password": 0, "_id": 0}):
    print(f"  {u['username']:<12}  role={u['role']:<8}  name={u['name']}")

print("\nDefault passwords:")
print("  admin    → admin123")
print("  aditi    → aditi123")
print("  shreyas  → shreyas123")
print("  viewer   → view123")
print("\nChange these from the User Management page after logging in.")
