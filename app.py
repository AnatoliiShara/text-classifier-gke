import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

# Синтезуємо дані
def generate_dataset():
    data = {
        "text": [
            "I love programming in Python", "Python is great for data science", 
            "Data analysis is fun", "Football is exciting", "Soccer is popular worldwide",
            "I enjoy playing basketball", "Music makes me happy", 
            "Classical music is relaxing", "Rock music energizes me",
            "I love coding", "Coding challenges are fun", "Programming is creative"
        ],
        "label": ["tech", "tech", "tech", "sports", "sports", "sports", 
                  "music", "music", "music", "tech", "tech", "tech"]
    }
    df = pd.DataFrame(data)

    # Зберігаємо датасет у файл dataset/dataset.csv
    if not os.path.exists("dataset"):
        os.makedirs("dataset")
    df.to_csv("dataset/dataset.csv", index=False)
    return df

# Тренуємо моделі
def train_models():
    df = generate_dataset()
    X, y = df["text"], df["label"]
    
    # Модель 1: Naive Bayes
    model_1 = make_pipeline(CountVectorizer(), MultinomialNB())
    model_1.fit(X, y)
    if not os.path.exists("models"):
        os.makedirs("models")
    pickle.dump(model_1, open("models/model_1.pkl", "wb"))
    
    # Модель 2: SVM
    model_2 = make_pipeline(CountVectorizer(), SVC(probability=True))
    model_2.fit(X, y)
    pickle.dump(model_2, open("models/model_2.pkl", "wb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    model_choice = data.get("model", "model_1")  # model_1 or model_2

    # Завантажуємо обрану модель
    model_path = f"models/{model_choice}.pkl"
    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found!"}), 404
    
    model = pickle.load(open(model_path, "rb"))
    prediction = model.predict([text])[0]
    return jsonify({"text": text, "prediction": prediction})

if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")
    train_models()
    app.run(host="0.0.0.0", port=5000)
