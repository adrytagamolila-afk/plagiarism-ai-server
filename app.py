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
            "check": "POST /check"
        }
    })


@app.get("/health")
def health():
    return jsonify({"ok": True})


def compute_suggested_score(similarity_percent):
    # Suggested score only (teacher gives final score)
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
    # Accept both formats:
    # 1) {"essays": {...}}
    # 2) {...}
    if not isinstance(payload, dict):
        return {}

    if "essays" in payload and isinstance(payload["essays"], dict):
        essays = payload["essays"]
    else:
        essays = payload

    cleaned = {}
    for k, v in essays.items():
        if isinstance(k, str) and isinstance(v, str) and v.strip():
            cleaned[k.strip()] = v.strip()
    return cleaned


@app.route("/check", methods=["POST"])
def check_plagiarism():
    payload = request.get_json(silent=True)
    essays = normalize_essays(payload)

    if len(essays) < 2:
        return jsonify({
            "error": "At least 2 essays required"
        }), 400

    students = list(essays.keys())
    texts = [essays[s] for s in students]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    results = []
    similarities = []

    for i in range(len(students)):
        for j in range(i + 1, len(students)):
            sim_percent = round(float(similarity_matrix[i][j]) * 100, 2)
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
        "avgSimilarity": round(sum(similarities) / len(similarities), 2),
        "maxSimilarity": max(similarities)
    }

    return jsonify({
        "results": results,
        "overall": overall
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
