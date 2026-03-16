# MovieRec - ML Movie Recommendation Engine

A content-based movie recommendation system built with scikit-learn and Flask. An offline ML pipeline computes movie similarity using TF-IDF on genres and user tags, and a web app lets you browse, search, like movies, and get personalized recommendations.

## ML Approach

1. **Content profiles** — Each movie gets a text profile combining its genres and user-contributed tags from MovieLens.
2. **TF-IDF vectorization** — Profiles are transformed into TF-IDF vectors, capturing the importance of each term relative to the corpus.
3. **Cosine similarity** — Pairwise similarity between all movies is computed. The top 20 most similar movies are stored for each movie.
4. **Personalized recommendations** — When a user likes movies, similarity scores from all liked movies are aggregated and ranked to produce personalized suggestions.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Pipeline

```bash
# Step 1: Download MovieLens dataset
python pipeline/download_data.py

# Step 2: Build recommendation artifacts
python pipeline/build_recommendations.py
```

This produces `output/movies.pkl` and `output/similarity.pkl`.

## Run the Web App

```bash
flask --app app/app.py run
```

Open http://127.0.0.1:5000 to browse movies, view similar movies, like movies, and get personalized recommendations.
