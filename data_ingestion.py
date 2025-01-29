import sqlite3
from datasets import load_dataset
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

print("Step 1: Data Collection")
print("Loading IMDB dataset from Hugging Face...")
dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]

print(f"Train data size: {len(train_data)} rows")
print(f"Test data size: {len(test_data)} rows")

print("\nStep 2: Setting up SQLite database...")
conn = sqlite3.connect("imdb_reviews.db")
cursor = conn.cursor()

print("Creating table 'imdb_reviews'...")
cursor.execute('''
    CREATE TABLE IF NOT EXISTS imdb_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        review_text TEXT,
        sentiment TEXT
    )
''')

print("Inserting data into the table...")
for review, sentiment in zip(train_data["text"], train_data["label"]):
    sentiment_label = "positive" if sentiment == 1 else "negative"
    cursor.execute("INSERT INTO imdb_reviews (review_text, sentiment) VALUES (?, ?)", 
                   (review, sentiment_label))

conn.commit()
conn.close()
print("Data inserted into SQLite database successfully!")

print("\nStep 3: Data Cleaning & EDA")
print("Cleaning text data...")

def clean_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text

train_data = train_data.map(lambda x: {"cleaned_text": clean_text(x["text"])})
test_data = test_data.map(lambda x: {"cleaned_text": clean_text(x["text"])});

print("Text cleaning completed!")

# Sentiment distribution
print("Performing Exploratory Data Analysis...")
sentiment_counts = pd.Series(train_data["label"]).value_counts()
print("Sentiment Distribution:\n", sentiment_counts)

plt.figure(figsize=(8, 6))
plt.bar(["Negative", "Positive"], sentiment_counts.values, color=["red", "green"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.show()

# Average review length
train_data = train_data.map(lambda x: {"review_length": len(x["cleaned_text"])})

positive_reviews = [x["review_length"] for x in train_data if x["label"] == 1]
negative_reviews = [x["review_length"] for x in train_data if x["label"] == 0]

avg_length_positive = sum(positive_reviews) / len(positive_reviews)
avg_length_negative = sum(negative_reviews) / len(negative_reviews)

print(f"Average review length (Positive): {avg_length_positive}")
print(f"Average review length (Negative): {avg_length_negative}")

# Word cloud
print("Generating Word Clouds...")
positive_text = " ".join([x["cleaned_text"] for x in train_data if x["label"] == 1])
negative_text = " ".join([x["cleaned_text"] for x in train_data if x["label"] == 0])

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
wordcloud_positive = WordCloud(width=800, height=400, background_color="white").generate(positive_text)
plt.imshow(wordcloud_positive, interpolation="bilinear")
plt.title("Word Cloud: Positive Reviews")
plt.axis("off")

plt.subplot(1, 2, 2)
wordcloud_negative = WordCloud(width=800, height=400, background_color="white").generate(negative_text)
plt.imshow(wordcloud_negative, interpolation="bilinear")
plt.title("Word Cloud: Negative Reviews")
plt.axis("off")

plt.show()

# Optional: Save cleaned dataset as CSV
print("Saving cleaned dataset as CSV for future reference...")
cleaned_train_data = pd.DataFrame({
    "review_text": train_data["text"],
    "cleaned_text": train_data["cleaned_text"],
    "sentiment": ["positive" if label == 1 else "negative" for label in train_data["label"]]
})

cleaned_test_data = pd.DataFrame({
    "review_text": test_data["text"],
    "cleaned_text": test_data["cleaned_text"],
    "sentiment": ["positive" if label == 1 else "negative" for label in test_data["label"]]
})

cleaned_train_data.to_csv("cleaned_train_data.csv", index=False)
cleaned_test_data.to_csv("cleaned_test_data.csv", index=False)
print("Cleaned datasets saved as 'cleaned_train_data.csv' and 'cleaned_test_data.csv'!")