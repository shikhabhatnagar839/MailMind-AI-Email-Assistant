
import pandas as pd

# Load our sample email dataset
df = pd.read_csv("sample_emails.csv")

# Display basic information
print("Total emails:", len(df))
print("\nEmail columns:")
print(df.columns.tolist())

print("\nFirst two emails:")
print(df[["subject", "sender", "body"]].head(2).to_string(index=False))
