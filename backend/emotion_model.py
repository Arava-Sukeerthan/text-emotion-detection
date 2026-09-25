import os
from typing import Tuple

EMOTION_KEYWORDS = {
    "joy": ["happy", "joy", "love", "excited", "amazing", "great", "wonderful", "delighted", "smile", "fun"],
    "sadness": ["sad", "cry", "hurt", "depressed", "lonely", "upset", "tired", "broken", "grief", "disappointed"],
    "anger": ["angry", "mad", "furious", "hate", "rage", "annoyed", "upset", "rage", "frustrated"],
    "fear": ["fear", "afraid", "scared", "panic", "anxious", "nervous", "terrified", "worried"],
    "surprise": ["surprised", "shock", "wow", "unexpected", "amazed", "shocked", "astonished"],
    "love": ["love", "adore", "affection", "like", "romantic", "cherish", "care"],
}


def keyword_fallback(text: str) -> Tuple[str, float]:
    cleaned = text.lower()
    scores = {emotion: 0 for emotion in EMOTION_KEYWORDS}

    for emotion, words in EMOTION_KEYWORDS.items():
        for word in words:
            if word in cleaned:
                scores[emotion] += 1

    if not any(scores.values()):
        return "joy", 0.0

    best_emotion, best_score = max(scores.items(), key=lambda item: item[1])
    confidence = round((best_score / max(1, len(cleaned.split()))) * 100, 2)
    return best_emotion, min(confidence, 99.99)


def load_classifier():
    try:
        from transformers import pipeline

        model_name = os.getenv(
            "EMOTION_MODEL",
            "bhadresh-savani/distilbert-base-uncased-emotion",
        )
        return pipeline("text-classification", model=model_name, tokenizer=model_name, top_k=None)
    except Exception:
        return None


def predict_emotion(text: str) -> Tuple[str, float]:
    if not text or not text.strip():
        return None, 0.0

    classifier = load_classifier()

    if classifier is None:
        return keyword_fallback(text)

    try:
        result = classifier(text)[0]
        best = None
        best_score = 0.0

        for item in result:
            label = str(item.get("label", "")).lower()
            score = float(item.get("score", 0.0))

            if label in {"joy", "sadness", "anger", "fear", "surprise", "love"}:
                if score > best_score:
                    best = label
                    best_score = score

        if best is None:
            return keyword_fallback(text)

        return best, round(best_score * 100, 2)
    except Exception:
        return keyword_fallback(text)
