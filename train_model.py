import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocess import clean_text

# Dataset
data = {
    "text": [
        "I love this product",
        "This is the worst experience",
        "It is okay nothing special",
        "Amazing quality and service",
        "I hate this so much",
        "The product is average",
        "Very happy with the result",
        "Terrible and disappointing",
        "Not bad could be better",
        "Best purchase ever",
        "Worst product ever",
        "Its fine works as expected"
    ],
    # 0 = Negative, 1 = Neutral, 2 = Positive
    "label": [2,0,1,2,0,1,2,0,1,2,0,1]
}

df = pd.DataFrame(data)
df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['label']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved")
