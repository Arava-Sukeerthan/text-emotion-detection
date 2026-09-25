import joblib


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("emotion_model.pkl")

vectorizer = joblib.load(
    "tfidf_vectorizer.pkl"
)


print("\n===================================")
print("TEXT EMOTION DETECTOR")
print("===================================")

print("Type 'exit' to stop.")


# ==========================================
# PREDICTION LOOP
# ==========================================

while True:

    text = input("\nEnter your text: ")

    if text.lower().strip() == "exit":

        print("\nProgram closed.")
        break


    if not text.strip():

        print("Please enter some text.")
        continue


    # Convert text into TF-IDF
    text_vector = vectorizer.transform(
        [text]
    )


    # Prediction
    prediction = model.predict(
        text_vector
    )[0]


    # Probability
    probabilities = model.predict_proba(
        text_vector
    )[0]


    confidence = max(
        probabilities
    ) * 100


    print("\n-----------------------------")

    print(
        "Predicted Emotion:",
        prediction.upper()
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print("-----------------------------")