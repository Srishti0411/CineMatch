# CineMatch — Movie Recommendation System

A content-based movie recommendation engine built with Python and Flask. Search any movie and get instant recommendations based on genres, keywords, cast, and plot similarity.

---

## How It Works

The recommender uses **content-based filtering** — each movie is represented as a "tag soup" combining its plot, genres, keywords, top 3 cast members, and director. These are vectorized using CountVectorizer and compared via **cosine similarity** to find the closest matches.

```
Movie → tags (plot + genres + cast + director) → CountVectorizer → Cosine Similarity → Top 10 matches
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML | scikit-learn, NumPy, pandas |
| Vectorization | CountVectorizer (bag of words) |
| Similarity | Cosine Similarity |
| Frontend | HTML, CSS, Vanilla JS |
| Dataset | TMDB 5000 Movies |

---

## Project Structure

```
movie-recommender/
├── app/
│   ├── app.py              # Flask routes
│   ├── recommender.py      # Recommendation logic
│   └── templates/
│       └── index.html      # Frontend UI
├── notebooks/
│   └── EDA_and_Model.ipynb # Exploratory analysis & model walkthrough
├── train.py                # Trains and saves model artifacts
├── requirements.txt
└── data/                   # Dataset goes here (not committed)
```

---

## Dataset

**TMDB 5000 Movie Dataset** — 5000 movies with metadata including genres, keywords, cast, crew, and plot overviews.

Source: [kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

---

## Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add dataset CSVs to data/ folder, then train
python train.py

# Run the app
cd app && python app.py
# → http://localhost:5000
```

---

## Notebook

`notebooks/EDA_and_Model.ipynb` walks through the full pipeline — data exploration, feature engineering, model building, and a comparison between CountVectorizer and TF-IDF approaches.