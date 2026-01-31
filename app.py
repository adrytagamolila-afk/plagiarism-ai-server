from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # allow Android to connect

@app.route("/check", methods=["POST"])
def check_plagiarism():
    data = request.json
    essays = data.get("essays", {})

    if len(essays) < 2:
        return jsonify({"error": "At least 2 essays required"}), 400

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(essays.values())
    similarity = cosine_similarity(tfidf_matrix)

    students = list(essays.keys())
    results = []

    for i in range(len(students)):
        for j in range(i + 1, len(students)):
            results.append({
                "student1": students[i],
                "student2": students[j],
                "similarity": round(similarity[i][j] * 100, 2)
            })

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
