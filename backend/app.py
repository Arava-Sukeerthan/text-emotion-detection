import os

from flask import Flask, jsonify, request

from emotion_model import predict_emotion

app = Flask(__name__)


@app.get("/")
def home():
    return jsonify({
        "message": "Emotion API is running",
        "endpoints": [
            "/health",
            "/predict",
            "/api/predict",
        ],
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
@app.post("/api/predict")
def predict_route():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "") or "").strip()

    if not text:
        return jsonify({"error": "Please send a text value in JSON body."}), 400

    emotion, confidence = predict_emotion(text)
    return jsonify({
        "emotion": emotion,
        "confidence": confidence,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
