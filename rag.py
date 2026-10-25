import os
import pandas as pd
import faiss
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# 1. Load API key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY .env file mein nahi mili.")

# 2. Connect to Groq
client = Groq(api_key=api_key)

# 3. Load emails
df = pd.read_csv("sample_emails.csv")

email_texts = (
    df["subject"].fillna("") + ". " +
    df["body"].fillna("")
).tolist()

# 4. Create email embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
email_vectors = model.encode(
    email_texts,
    normalize_embeddings=True
)

# 5. Create FAISS index
index = faiss.IndexFlatIP(email_vectors.shape[1])
index.add(email_vectors)

# 6. Ask the user a question
question = input("Ask MailMind a question: ")

# 7. Search for relevant emails
question_vector = model.encode(
    [question],
    normalize_embeddings=True
)

scores, indices = index.search(question_vector, k=2)

# 8. Build context from retrieved emails
context_parts = []

for i in indices[0]:
    email = df.iloc[int(i)]
    context_parts.append(
        f"Subject: {email['subject']}\n"
        f"Sender: {email['sender']}\n"
        f"Email: {email['body']}"
    )

context = "\n\n".join(context_parts)

# 9. Ask the AI to answer using the email context
response = client.chat.completions.create(
model="openai/gpt-oss-20b",    messages=[
        {
            "role": "system",
            "content": (
                "You are MailMind, an AI email assistant. "
                "Answer using only the provided email context. "
                "If the information is missing, say you could not "
                "find it in the emails. Do not invent details."
            )
        },
        {
            "role": "user",
            "content": (
                f"Email context:\n{context}\n\n"
                f"Question: {question}"
            )
        }
    ],
    temperature=0.2
)

print("\nMailMind's Answer:")
print(response.choices[0].message.content)