import csv

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


model = SentenceTransformer("all-MiniLM-L6-v2")


bugs = []

with open("data/bugs.csv", "r") as file:
    reader = csv.DictReader(file)

    for bug in reader:
        bugs.append(bug)


test_pairs = [
    # Similar pairs — expected 1
    ("BUG-003", "BUG-004", 1),
    ("BUG-005", "BUG-006", 1),
    ("BUG-007", "BUG-008", 1),
    ("BUG-001", "BUG-002", 1),
    ("BUG-009", "BUG-010", 1),
    ("BUG-001", "BUG-009", 0),

    # Different pairs — expected 0
    ("BUG-001", "BUG-005", 0),
    ("BUG-003", "BUG-007", 0),
    ("BUG-005", "BUG-009", 0),
    ("BUG-002", "BUG-008", 0),
    ("BUG-004", "BUG-006", 0),
]


bug_lookup = {bug["id"]: bug for bug in bugs}


correct = 0
total = len(test_pairs)

true_positive = 0
false_positive = 0
false_negative = 0


for bug_a_id, bug_b_id, expected in test_pairs:

    bug_a = bug_lookup[bug_a_id]
    bug_b = bug_lookup[bug_b_id]

    embedding_a = model.encode(bug_a["description"])
    embedding_b = model.encode(bug_b["description"])

    similarity = cos_sim(embedding_a, embedding_b).item()

    predicted = 1 if similarity >= 0.65 else 0

    if predicted == expected:
        correct += 1

    if predicted == 1 and expected == 1:
        true_positive += 1
    elif predicted == 1 and expected == 0:
        false_positive += 1
    elif predicted == 0 and expected == 1:
        false_negative += 1

    print(
        bug_a_id,
        "<->",
        bug_b_id,
        "| Similarity:",
        round(similarity, 3),
        "| Expected:",
        expected,
        "| Predicted:",
        predicted
    )


accuracy = correct / total

precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

print()
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")