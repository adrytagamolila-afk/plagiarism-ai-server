from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)
CORS(app)  # allow Android / other clients to connect


@app.get("/")
def home():
    return jsonify({
        "ok": True,
        "service": "plagiarism-ai-server",
        "endpoints": {
            "health": "GET /health",
            "check": "POST /check (JSON)"
        },
        "example_body_format_1": {
            "essays": {
                "StudentA": "text...",
                "StudentB": "text..."
            }
        },
        "example_body_format_2": {
            "StudentA": "text...",
            "StudentB": "text..."
        }
    })


@app.get("/health")
def health():
    return jsonify({"ok": True})


def compute_suggested_score(similarity_percent: float) -> int:
    """
    Suggested score only (teacher gives final score).
    Returns an integer 1..5 based on similarity bands.
    """
    if similarity_percent >= 80:
        return 5
    if similarity_percent >= 60:
        return 4
    if similarity_percent >= 40:
        return 3
    if similarity_percent >= 20:
        return 2
    return 1


def normalize_essays(payload):
    """
    Accepts either:
      1) {"essays": {...}}
      2) {...}  (direct map)
    Returns: dict of {student: essay_text}
    """
    if not isinstance(payload, dict):
        return {}

    if "essays" in payload and isinstance(payload["essays"], dict):
        essays = payload["essays"]
    else:
        essays = payload

    # Clean & validate: keep only string keys and string values
    cleaned = {}
    for k, v in essays.items():
        if isinstance(k, str) and isinstance(v, str):
            text = v.strip()
            if text:
                cleaned[k.strip()] = text
    return cleaned


@app.route("/check", methods=["POST"])
def check_plagiarism():
    payload = request.get_json(silent=True)
    essays = normalize_essays(payload)

    if len(essays) < 2:
        return jsonify({
            "error": "At least 2 essays required",
            "expected_format_1": {"essays": {"StudentA": "text", "StudentB": "text"}},
            "expected_format_2": {"StudentA": "text", "StudentB": "text"}
        }), 400

    students = list(essays.keys())
    texts = [essays[name] for name in students]

    # TF-IDF cosine similarity
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf_matrix)

    results = []
    similarities = []

    for i in range(len(students)):
        for j in range(i + 1, len(students)):
            sim_percent = round(float(sim_matrix[i][j]) * 100, 2)
            similarities.append(sim_percent)

            results.append({
                "student1": students[i],
                "student2": students[j],
                "similarity": sim_percent,
                "suggestedScore": compute_suggested_score(sim_percent)
            })

    overall = {
        "studentsCount": len(students),
        "comparisons": len(results),
        "avgSimilarity": round(sum(similarities) / len(similarities), 2) if similarities else 0.0,
        "maxSimilarity": max(similarities) if similarities else 0.0
    }

    return jsonify({"results": results, "overall": overall})


if __name__ == "__main__":
    # Local only. On Render you should run with gunicorn (see below).
    app.run(host="0.0.0.0", port=5000)
