import json
from difflib import SequenceMatcher

# Load prepared evaluation data
with open("rag_evaluation/eval_questions.json") as f:
    eval_data = json.load(f)

from datasets import load_from_disk
dataset = load_from_disk("rag_evaluation/ragas_dataset")

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

precision_scores = []
recall_scores = []

for i in range(len(dataset)):
    gt = dataset[i]["ground_truth"]
    contexts = dataset[i]["contexts"]

    # Count relevant contexts
    relevant = sum(similarity(ctx, gt) > 0.3 for ctx in contexts)

    # Precision = relevant retrieved / total retrieved
    precision = relevant / len(contexts) if contexts else 0
    precision_scores.append(precision)

    # Recall = did we retrieve ANY relevant context?
    recall_scores.append(1 if relevant > 0 else 0)

print("✅ Manual Retrieval Evaluation")
print("Context Precision:", round(sum(precision_scores)/len(precision_scores), 3))
print("Context Recall:", round(sum(recall_scores)/len(recall_scores), 3))