# model_training.py
"""
This script includes training two models:
1. Logistic Regression (TF-IDF vectorization).
2. DistilBERT (advanced model using Hugging Face).

Note:
- Logistic Regression model is designed to run on both Colab and local environments.
- DistilBERT model was trained on Google Colab due to hardware limitations.
  After training, the models and vectorizers were saved locally for deployment.

Both models can be used interchangeably in the application.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split
import pickle
from datasets import load_dataset
import tensorflow as tf
from datasets import Dataset
import pandas as pd

# ----------------- Logistic Regression Model -----------------

print("Step 1: Logistic Regression Model Training")

# Load full IMDB dataset
print("Loading the full IMDB dataset...")
dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]

# Extract text and labels
print("Preparing training and test datasets for Logistic Regression...")
X_train_text = train_data["text"]
y_train = train_data["label"]
X_test_text = test_data["text"]
y_test = test_data["label"]

# TF-IDF Vectorization
print("Vectorizing text data using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# Train Logistic Regression Model
print("Training Logistic Regression model on the complete dataset...")
model_lr = LogisticRegression(max_iter=500)  # Increased iterations for convergence
model_lr.fit(X_train, y_train)

# Evaluate Model
print("Evaluating the Logistic Regression model on the test dataset...")
y_pred = model_lr.predict(X_test)
print("Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and vectorizer
print("Saving the Logistic Regression model and vectorizer...")
with open("model_lr.pkl", "wb") as f:
    pickle.dump(model_lr, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Logistic Regression model and vectorizer saved successfully!")
"""
# ----------------- DistilBERT Model (Advanced) -----------------

print("\nStep 2: DistilBERT Model Training")

# Tokenizer and Model Initialization
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = TFDistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# Tokenize the data
print("Tokenizing data for DistilBERT...")
train_data_bert = Dataset.from_pandas(pd.DataFrame({"text": X_train_text, "label": y_train}))
test_data_bert = Dataset.from_pandas(pd.DataFrame({"text": X_test_text, "label": y_test}))

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

train_data_bert = train_data_bert.map(tokenize_function, batched=True)
test_data_bert = test_data_bert.map(tokenize_function, batched=True)

# Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Convert to TensorFlow Dataset
train_dataset = train_data_bert.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols="label",
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

test_dataset = test_data_bert.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols="label",
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

# Compile the model
bert_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the model
print("Training DistilBERT model...")
bert_model.fit(train_dataset, epochs=3)

# Evaluate the model
print("Evaluating DistilBERT model...")
bert_model.evaluate(test_dataset)

# Save DistilBERT Model
print("Saving the DistilBERT model and tokenizer...")
bert_model.save_pretrained("./distilbert_model")
tokenizer.save_pretrained("./distilbert_model")
print("DistilBERT model and tokenizer saved successfully!")

# ----------------- Summary -----------------

print("\nModel training complete.")
print("1. Logistic Regression model: model_lr.pkl and vectorizer.pkl")
print("2. DistilBERT model: distilbert_model directory (saved tokenizer and weights)")

"""