
import os
import pandas as pd
import streamlit as st
import faiss

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="MailMind",
    page_icon="📧",
    layout="wide"
)

st.title("📧 MailMind — AI Email Assistant")
st.write("Search your emails and ask questions using AI.")


# ---------------- LOAD EMAILS ----------------
try:
    df = pd.read_csv("sample_emails.csv")
except FileNotFoundError:
    st.error("sample_emails.csv file nahi mili. Project folder check karo.")
    st.stop()

required_columns = ["subject", "sender", "date", "body"]

if not all(column in df.columns for column in required_columns):
    st.error("CSV file mein required columns missing hain.")
    st.stop()

if df.empty:
    st.warning("CSV file mein koi email nahi hai.")
    st.stop()


# ---------------- EMAIL PRIORITY ----------------
def detect_priority(row):
    text = (
        str(row["subject"]) + " " +
        str(row["body"])
    ).lower()

    high_keywords = [
        "urgent", "immediately", "interview",
        "deadline", "submit", "required"
    ]

    medium_keywords = [
        "meeting", "internship", "reminder",
        "opportunity", "schedule"
    ]

    if any(word in text for word in high_keywords):
        return "High"
    elif any(word in text for word in medium_keywords):
        return "Medium"
    else:
        return "Low"


df["priority"] = df.apply(detect_priority, axis=1)


# ---------------- DASHBOARD STATISTICS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Emails", len(df))

with col2:
    st.metric("Unique Senders", df["sender"].nunique())

with col3:
    st.metric("Email Categories", df["subject"].nunique())


# ---------------- SEARCH EMAILS ----------------
st.subheader("📩 Your Emails")

search_query = st.text_input(
    "🔍 Search by subject or sender",
    placeholder="e.g. Interview or hr@techcorp.example"
)

if search_query.strip():
    filtered_df = df[
        df["subject"].str.contains(
            search_query, case=False, na=False, regex=False
        )
        |
        df["sender"].str.contains(
            search_query, case=False, na=False, regex=False
        )
    ]
else:
    filtered_df = df

st.dataframe(
    filtered_df[
        ["subject", "sender", "date", "priority", "body"]
    ],
    use_container_width=True
)


# ---------------- LOAD API KEY ----------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error(
        "GROQ_API_KEY nahi mili. Apni .env file check karo."
    )
    st.stop()


# ---------------- LOAD EMBEDDING MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# ---------------- PREPARE EMAIL EMBEDDINGS ----------------
email_texts = (
    df["subject"].fillna("").astype(str)
    + ". "
    + df["body"].fillna("").astype(str)
).tolist()


@st.cache_resource
def create_search_index(texts):
    vectors = model.encode(
        texts,
        normalize_embeddings=True
    )

    search_index = faiss.IndexFlatIP(vectors.shape[1])
    search_index.add(vectors)

    return search_index


index = create_search_index(email_texts)


# ---------------- ASK MAILMIND ----------------
st.subheader("💬 Ask MailMind")

question = st.text_input(
    "Enter your question",
    placeholder="When is my interview?"
)

if st.button("Get Answer", type="primary"):

    if not question.strip():
        st.warning("Please enter a question first.")

    else:
        try:
            with st.spinner(
                "Searching emails and generating your answer..."
            ):
                # 1. Convert question into an embedding
                question_vector = model.encode(
                    [question],
                    normalize_embeddings=True
                )

                # 2. Retrieve the two most relevant emails
                scores, indices = index.search(
                    question_vector, k=min(2, len(df))
                )

                context_parts = []
                source_indices = []

                for i in indices[0]:
                    if i >= 0:
                        email = df.iloc[int(i)]
                        source_indices.append(int(i))

                        context_parts.append(
                            f"Subject: {email['subject']}\n"
                            f"Sender: {email['sender']}\n"
                            f"Date: {email['date']}\n"
                            f"Email: {email['body']}"
                        )

                context = "\n\n".join(context_parts)

                # 3. Send retrieved emails and question to the LLM
                client = Groq(api_key=api_key)

                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are MailMind, an AI email assistant. "
                                "Answer the user's question using only "
                                "the provided email context. "
                                "If the answer is not present, say you "
                                "could not find it in the emails. "
                                "Do not invent details. "
                                "Keep the answer clear and concise."
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
                    temperature=0.2,
                    max_completion_tokens=1024
                )

                answer = response.choices[0].message.content

            # 4. Display the answer
            st.subheader("🤖 MailMind's Answer")
            st.write(answer or "The AI did not return an answer.")

            # 5. Display source emails
            st.subheader("📚 Source Emails")

            for i in source_indices:
                email = df.iloc[i]

                with st.expander(f"📧 {email['subject']}"):
                    st.write("**Sender:**", email["sender"])
                    st.write("**Date:**", email["date"])
                    st.write("**Priority:**", email["priority"])
                    st.write("**Email Content:**", email["body"])

        except Exception as error:
            st.error(
                "Answer generate karte waqt error aaya. "
                "Neeche error details dekho."
            )
            st.code(str(error))
