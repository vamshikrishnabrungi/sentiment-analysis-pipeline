from flask import Flask, request, render_template_string
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import pickle
import os
import torch

# Disable GPU for safety
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)

# Ensure model files exist
if not os.path.exists("model_lr.pkl") or not os.path.exists("vectorizer.pkl"):
    raise FileNotFoundError("Logistic Regression model or vectorizer not found!")

if not os.path.exists("distilbert_model"):
    raise FileNotFoundError("DistilBERT model folder not found!")

# Load the Logistic Regression model and vectorizer
with open("model_lr.pkl", "rb") as lr_file, open("vectorizer.pkl", "rb") as vec_file:
    model_lr = pickle.load(lr_file)
    vectorizer = pickle.load(vec_file)

# Load the DistilBERT model
device = torch.device("cpu")
distilbert_model = DistilBertForSequenceClassification.from_pretrained("distilbert_model", from_tf=True)
distilbert_model.to(device)
distilbert_model.eval()  # Set model to evaluation mode

tokenizer = DistilBertTokenizer.from_pretrained("distilbert_model")


@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = ""
    review_text = ""

    if request.method == "POST":
        selected_model = request.form.get("model_select")
        review_text = request.form.get("review_text")

        if selected_model == "logistic_regression" and review_text:
            vectorized_text = vectorizer.transform([review_text])
            prediction = model_lr.predict(vectorized_text)
            sentiment = "Positive" if prediction[0] == 1 else "Negative"

        elif selected_model == "distilbert" and review_text:
            inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            with torch.no_grad():  # Avoid computation graph tracking
                outputs = distilbert_model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1).item()
            sentiment = "Positive" if prediction == 1 else "Negative"

        return render_template_string(HTML_TEMPLATE, sentiment=sentiment, review_text=review_text)

    return render_template_string(HTML_TEMPLATE, sentiment="", review_text="")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis</title>
</head>
<body>
    <h1>Sentiment Analysis</h1>
    <form method="POST">
        <label for="review_text">Enter Review:</label><br>
        <textarea id="review_text" name="review_text" rows="5" cols="40">{{ review_text }}</textarea><br><br>

        <label for="model_select">Select Model:</label><br>
        <select id="model_select" name="model_select">
            <option value="logistic_regression">Logistic Regression</option>
            <option value="distilbert">DistilBERT</option>
        </select><br><br>
        
        <button type="submit">Submit</button>
    </form>
    <h2>Predicted Sentiment: {{ sentiment }}</h2>
</body>
</html>
"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
