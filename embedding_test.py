from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Load the AI model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Two bug reports
bug_1 = "Game crashes when loading Level 3"
bug_2 = "The game audio stops after the player pauses the game"

# Convert both bug reports into embeddings
embedding_1 = model.encode(bug_1)
embedding_2 = model.encode(bug_2)

# Compare the two embeddings
similarity = cos_sim(embedding_1, embedding_2)

print("Bug 1:", bug_1)
print("Bug 2:", bug_2)
print("Similarity:", similarity.item())