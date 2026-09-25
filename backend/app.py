import os

from flask import Flask, jsonify, request, send_from_directory

from emotion_model import predict_emotion

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.get("/")
def home():
    if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
        return send_from_directory(FRONTEND_DIR, "index.html")
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


@app.route("/predict", methods=["GET", "POST", "OPTIONS"])
@app.route("/api/predict", methods=["GET", "POST", "OPTIONS"])
def predict_route():
    if request.method == "OPTIONS":
        return "", 200

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
