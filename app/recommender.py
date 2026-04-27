"""
recommender.py
--------------
Core recommendation logic used by the Flask app.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class MovieRecommender:
    def __init__(self):
        self.movies = None
        self.similarity = None
        self.title_index = None
        self._loaded = False

    def load(self):
        """Load pre-trained artifacts from disk."""
        if self._loaded:
            return

        movies_path = os.path.join(MODEL_DIR, "movies.pkl")
        if not os.path.exists(movies_path):
            raise RuntimeError(
                "Model artifacts not found. Please run `python train.py` first."
            )

        self.movies = pd.read_pickle(os.path.join(MODEL_DIR, "movies.pkl"))
        self.similarity = np.load(os.path.join(MODEL_DIR, "similarity.npy"))

        with open(os.path.join(MODEL_DIR, "title_index.json")) as f:
            self.title_index = json.load(f)

        self._loaded = True

    # ─────────────────────────────────────────
    # Content-Based Recommendations
    # ─────────────────────────────────────────

    def content_recommend(self, title, n=10):
        """Return top-n movies similar to the given title."""
        self.load()

        if title not in self.title_index:
            return []

        idx = self.title_index[title]
        distances = list(enumerate(self.similarity[idx]))
        distances = sorted(distances, key=lambda x: x[1], reverse=True)
        # Skip index 0 (itself)
        top_indices = [i for i, _ in distances[1: n + 1]]

        results = self.movies.iloc[top_indices].copy()
        results["score"] = [round(distances[i][1] * 100, 1) for i in range(1, n + 1)]
        return self._format_results(results)

    # ─────────────────────────────────────────
    # Popular / Top Rated (fallback)
    # ─────────────────────────────────────────

    def top_rated(self, n=12, genre_filter=None):
        """Return top-rated movies, optionally filtered by genre."""
        self.load()

        df = self.movies.copy()

        if genre_filter:
            df = df[df["genres"].apply(
                lambda g: any(genre_filter.lower() in x.lower() for x in g)
                if isinstance(g, list) else False
            )]

        # Weighted rating (IMDB formula)
        C = df["vote_average"].mean()
        m = df["vote_count"].quantile(0.70)
        df = df[df["vote_count"] >= m].copy()
        df["weighted_score"] = (
            (df["vote_count"] / (df["vote_count"] + m)) * df["vote_average"] +
            (m / (df["vote_count"] + m)) * C
        )
        df = df.sort_values("weighted_score", ascending=False).head(n)
        df["score"] = df["weighted_score"].apply(lambda x: round(x * 10, 1))
        return self._format_results(df)

    # ─────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────

    def search(self, query, n=8):
        """Fuzzy search movie titles."""
        self.load()
        q = query.lower()
        results = self.movies[
            self.movies["title"].str.lower().str.contains(q, na=False)
        ].head(n)
        return results["title"].tolist()

    # ─────────────────────────────────────────
    # All genres
    # ─────────────────────────────────────────

    def get_genres(self):
        self.load()
        all_genres = set()
        for g_list in self.movies["genres"]:
            if isinstance(g_list, list):
                for g in g_list:
                    all_genres.add(g.replace(" ", ""))
        return sorted(all_genres)

    # ─────────────────────────────────────────
    # Format helpers
    # ─────────────────────────────────────────

    def _format_results(self, df):
        results = []
        for _, row in df.iterrows():
            results.append({
                "title": row["title"],
                "vote_average": round(float(row["vote_average"]), 1),
                "release_year": str(row["release_date"])[:4] if pd.notna(row["release_date"]) else "N/A",
                "genres": row["genres"] if isinstance(row["genres"], list) else [],
                "score": float(row.get("score", 0)),
                "poster_path": row.get("poster_path", ""),
            })
        return results


# Singleton
recommender = MovieRecommender()
