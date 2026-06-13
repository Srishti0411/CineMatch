"""
app.py
------
Flask web application for the Movie Recommender.
"""

from streamlit import title

from flask import Flask, render_template, request, jsonify
from recommender import recommender

app = Flask(__name__)

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def add_poster_url(movies):
    for m in movies:
        path = m.get("poster_path", "")
        m["poster_url"] = (TMDB_IMAGE_BASE + path) if path else ""
    return movies


@app.route("/")
def index():
    try:
        top_movies = add_poster_url(recommender.top_rated(n=12))
        genres = recommender.get_genres()
    except RuntimeError as e:
        top_movies = []
        genres = []
        print(f"Warning: {e}")
    return render_template("index.html", top_movies=top_movies, genres=genres)


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "No title provided"}), 400
        results = recommender.content_recommend(title, n=12)
        return jsonify({"error": f"Movie '{title}' not found or no results match your filters"}), 404

    return jsonify({"recommendations": add_poster_url(results), "query": title})


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])
    results = recommender.search(query)
    return jsonify(results)


@app.route("/genre/<genre_name>")
def by_genre(genre_name):
    results = recommender.top_rated(n=12, genre_filter=genre_name)
    return jsonify({"movies": add_poster_url(results), "genre": genre_name})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
