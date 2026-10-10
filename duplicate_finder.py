import csv

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


model = SentenceTransformer("all-MiniLM-L6-v2")


bugs = []

with open("data/bugs.csv", "r") as file:
    reader = csv.DictReader(file)

    for bug in reader:
        bugs.append(bug)


texts = [bug["description"] for bug in bugs]

embeddings = model.encode(texts)


for i in range(len(bugs)):
    for j in range(i + 1, len(bugs)):

        similarity = cos_sim(
            embeddings[i],
            embeddings[j]
        ).item()

        if similarity >= 0.70:
            print(
                bugs[i]["id"],
                "<->",
                bugs[j]["id"],
                "Similarity:",
                round(similarity, 3)
            )