import json
import hashlib
import os

CREDENTIALS_PATH = os.path.join("ui", "credentials.json")

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def load_credentials():
    """
    Safely load credentials.json.
    If file is missing or empty/invalid, initialize a fresh structure.
    """
    if not os.path.exists(CREDENTIALS_PATH):
        return {"users": {}}

    try:
        with open(CREDENTIALS_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {"users": {}}
            return json.loads(content)
    except json.JSONDecodeError:
        print("⚠ credentials.json is invalid or empty. Reinitializing it.")
        return {"users": {}}

def save_credentials(data):
    with open(CREDENTIALS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def main():
    users = {
        "admin": {
            "name": "Admin User",
            "password": hash_password("admin123"),
            "role": "admin"
        },
        "marketing1": {
            "name": "Marketing Manager",
            "password": hash_password("marketing123"),
            "role": "marketing"
        },
        "product1": {
            "name": "Product Manager",
            "password": hash_password("product123"),
            "role": "product"
        },
        "analyst1": {
            "name": "Data Analyst",
            "password": hash_password("analyst123"),
            "role": "analyst"
        }
    }

    data = load_credentials()
    data["users"].update(users)
    save_credentials(data)

    print("✅ Users added successfully to ui/credentials.json\n")
    print("Login credentials:")
    print("Admin        → admin / admin123")
    print("Marketing    → marketing1 / marketing123")
    print("Product      → product1 / product123")
    print("Analyst      → analyst1 / analyst123")

if __name__ == "__main__":
    main()