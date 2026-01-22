from flask import Flask, render_template, request
import pickle
import pandas as pd
from preprocess import clean_text
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load model
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    sentiment_class = None
    df_results = None

    # -------- Single Message --------
    if request.method == "POST" and "text" in request.form:
        text = request.form["text"]

        if text.strip() != "":
            clean = clean_text(text)
            vec = vectorizer.transform([clean])
            probs = model.predict_proba(vec)[0]
            pred = probs.argmax()
            confidence = round(probs[pred] * 100, 2)

            if pred == 2:
                sentiment_class = "Positive"
                sentiment = f"😊 Positive ({confidence}%)"
            elif pred == 1:
                sentiment_class = "Neutral"
                sentiment = f"😐 Neutral ({confidence}%)"
            else:
                sentiment_class = "Negative"
                sentiment = f"😠 Negative ({confidence}%)"

    # -------- CSV Upload --------
    if request.method == "POST" and "chatfile" in request.files:
        file = request.files["chatfile"]

        if file.filename.endswith(".csv"):
            df = pd.read_csv(file)

            results = []

            for _, row in df.iterrows():
                msg = row["message"]
                clean = clean_text(msg)
                vec = vectorizer.transform([clean])
                probs = model.predict_proba(vec)[0]
                pred = probs.argmax()
                confidence = round(probs[pred] * 100, 2)

                if pred == 2:
                    label, emoji = "Positive", "😊"
                elif pred == 1:
                    label, emoji = "Neutral", "😐"
                else:
                    label, emoji = "Negative", "😠"

                results.append({
                    "sender": row["sender"],
                    "message": msg,
                    "sentiment": label,
                    "confidence": confidence,
                    "emoji": emoji
                })

            df_results = pd.DataFrame(results)
            df_results.to_csv("static/chat_sentiment_results.csv", index=False)

            # Chart
            counts = df_results["sentiment"].value_counts()
            plt.figure(figsize=(5,4))
            counts.plot(kind="bar")
            plt.title("Sentiment Distribution")
            plt.ylabel("Messages")
            plt.tight_layout()
            plt.savefig("static/sentiment_chart.png")
            plt.close()

    return render_template(
        "index.html",
        sentiment=sentiment,
        sentiment_class=sentiment_class,
        df_results=df_results
    )

if __name__ == "__main__":
    app.run(debug=True)
