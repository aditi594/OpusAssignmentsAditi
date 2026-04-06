from datasets import load_from_disk
from difflib import SequenceMatcher

dataset = load_from_disk("rag_evaluation/ragas_dataset")

def is_relevant(context, ground_truth, threshold=0.25):
    return SequenceMatcher(
        None,
        context.lower(),
        ground_truth.lower()
    ).ratio() >= threshold

hits = 0

for i in range(len(dataset)):
    gt = dataset[i]["ground_truth"]
    contexts = dataset[i]["contexts"]

    if any(is_relevant(ctx, gt) for ctx in contexts):
        hits += 1

accuracy = hits / len(dataset)

print("✅ Retrieval Accuracy@K:", round(accuracy * 100, 2), "%")