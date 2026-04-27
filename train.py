"""
train.py
--------
Trains and saves both content-based and collaborative filtering models.
Run this once after downloading the dataset:
    python train.py
"""

import os
import json
import pickle
import ast
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data():
    movies_path = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
    credits_path = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")

    if not os.path.exists(movies_path) or not os.path.exists(credits_path):
        raise FileNotFoundError(
            "\n❌ Dataset not found!\n"
            "Please download from: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata\n"
            "And place both CSVs in the 'data/' folder.\n"
        )

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    print(f"✅ Loaded {len(movies)} movies")
    return movies, credits


# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────

def parse_list_col(text, key="name", limit=None):
    """Safely parse a JSON-like list column and extract a key."""
    try:
        items = ast.literal_eval(text)
        names = [i[key].replace(" ", "") for i in items]
        return names[:limit] if limit else names
    except Exception:
        return []


def get_director(crew_text):
    try:
        crew = ast.literal_eval(crew_text)
        for member in crew:
            if member.get("job") == "Director":
                return member["name"].replace(" ", "")
    except Exception:
        pass
    return ""


def preprocess(movies, credits):
    # Merge
    df = movies.merge(credits, on="title")

    # Select useful columns
    df = df[["movie_id", "title", "overview", "genres", "keywords",
         "cast", "crew", "vote_average", "vote_count", "release_date"]].copy()
    df["poster_path"] = ""
    df.dropna(inplace=True)

    # Parse columns
    df["genres"] = df["genres"].apply(lambda x: parse_list_col(x))
    df["keywords"] = df["keywords"].apply(lambda x: parse_list_col(x))
    df["cast"] = df["cast"].apply(lambda x: parse_list_col(x, limit=3))
    df["director"] = df["crew"].apply(get_director)
    df.drop("crew", axis=1, inplace=True)

    # Tokenize overview
    df["overview"] = df["overview"].apply(
        lambda x: x.split() if isinstance(x, str) else []
    )

    # Combine all tags into one string
    df["tags"] = (
        df["overview"] +
        df["genres"] +
        df["keywords"] +
        df["cast"] +
        df["director"].apply(lambda x: [x] if x else [])
    )
    df["tags"] = df["tags"].apply(lambda x: " ".join(x).lower())

    print(f"✅ Preprocessed {len(df)} movies")
    return df


# ─────────────────────────────────────────────
# 3. CONTENT-BASED MODEL
# ─────────────────────────────────────────────

def build_content_model(df):
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(df["tags"]).toarray()
    similarity = cosine_similarity(vectors)

    print("✅ Content-based similarity matrix built")
    return similarity


# ─────────────────────────────────────────────
# 4. SAVE ARTIFACTS
# ─────────────────────────────────────────────

def save_artifacts(df, similarity):
    # Save the movie dataframe (without heavy columns)
    movie_list = df[["movie_id", "title", "vote_average",
                      "vote_count", "release_date", "poster_path", "genres"]].copy()
    movie_list["genres"] = movie_list["genres"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    movie_list.to_pickle(os.path.join(MODEL_DIR, "movies.pkl"))
    np.save(os.path.join(MODEL_DIR, "similarity.npy"), similarity)

    # Save title → index map
    title_index = {title: idx for idx, title in enumerate(df["title"])}
    with open(os.path.join(MODEL_DIR, "title_index.json"), "w") as f:
        json.dump(title_index, f)

    # Save full tag data for display
    df[["title", "tags"]].to_pickle(os.path.join(MODEL_DIR, "tags.pkl"))

    print(f"✅ Artifacts saved to '{MODEL_DIR}/'")


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🎬 Movie Recommender — Training Pipeline\n" + "─" * 40)
    similarity_path = os.path.join(MODEL_DIR, "similarity.npy")

    if os.path.exists(similarity_path):
        print("✅ similarity.npy already exists, skipping rebuild")
        movies_path = os.path.join(MODEL_DIR, "movies.pkl")
        if not os.path.exists(movies_path):
            movies, credits = load_data()
            df = preprocess(movies, credits)
            similarity = np.load(similarity_path)
            save_artifacts(df, similarity)
    else:
        movies, credits = load_data()
        df = preprocess(movies, credits)
        similarity = build_content_model(df)
        save_artifacts(df, similarity)

    print("\n🎉 All done! You can now run the Flask app:\n   cd app && python app.py\n")
