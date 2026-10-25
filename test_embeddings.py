
from sentence_transformers import SentenceTransformer

# Load a pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Two example sentences
sentences = [
    "My technical interview is scheduled for October 28.",
    "When is my interview?"
]

# Convert sentences into numerical vectors
embeddings = model.encode(sentences)

print("Number of sentences:", len(embeddings))
print("Vector dimensions:", embeddings.shape)
print("First vector (first 5 numbers):", embeddings[0][:5])
