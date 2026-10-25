
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# 1. Load email data
df = pd.read_csv("sample_emails.csv")

# 2. Prepare searchable email text
email_texts = (
    df["subject"].fillna("") + ". " +
    df["body"].fillna("")
).tolist()

# 3. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Convert emails into vectors
email_vectors = model.encode(
    email_texts,
    normalize_embeddings=True
)

# 5. Create FAISS search index
dimension = email_vectors.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(email_vectors)

# 6. Ask a question
question = input("Ask a question about your emails: ")

# 7. Convert question into a vector
question_vector = model.encode(
    [question],
    normalize_embeddings=True
)

# 8. Find the most relevant email
scores, indices = index.search(question_vector, k=2)

# 9. Display search results
print("\nRelevant emails:\n")

for score, i in zip(scores[0], indices[0]):
    print("Subject:", df.iloc[i]["subject"])
    print("Sender:", df.iloc[i]["sender"])
    print("Email:", df.iloc[i]["body"])
    print("Similarity score:", round(float(score), 3))
    print("-" * 40)
