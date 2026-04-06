import json
import os
from datetime import datetime

MEMORY_PATH = os.path.join(os.path.dirname(__file__), "qa_memory.json")

def normalize_question(question: str) -> str:
    return " ".join(question.lower().strip().split())

def _load_memory():
    if not os.path.exists(MEMORY_PATH):
        return {}
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_memory(memory: dict):
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

def get_cached_answer(question: str):
    """
    Returns cached QA dict if exists, else None
    """
    memory = _load_memory()
    key = normalize_question(question)
    return memory.get(key)

def store_answer(question: str, answer: str):
    """
    Stores question + answer in memory
    """
    memory = _load_memory()
    key = normalize_question(question)

    memory[key] = {
        "question": question,
        "answer": answer,
        "saved_at": datetime.now().isoformat()
    }

    _save_memory(memory)