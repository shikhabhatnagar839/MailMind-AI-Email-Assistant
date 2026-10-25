
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load emails
df = pd.read_csv("sample_emails.csv")

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Combine each email's subject and body
email_texts = (
    df["subject"].fillna("") + ". " +
    df["body"].fillna("")
).tolist()

# Convert emails into vectors
embeddings = model.encode(email_texts)

print("Total emails:", len(email_texts))
print("Embedding shape:", embeddings.shape)
print("First email:", df.iloc[0]["subject"])
print("First vector dimensions:", len(embeddings[0]))

