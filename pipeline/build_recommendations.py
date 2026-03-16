"""Offline ML pipeline: build content-based movie recommendations."""

import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "ml-latest-small")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

TOP_N = 20


def load_data():
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    tags = pd.read_csv(os.path.join(DATA_DIR, "tags.csv"))
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    return movies, tags, ratings


def build_content_profiles(movies, tags):
    """Create a text profile for each movie from genres + tags."""
    # Genres: replace pipe separator with spaces
    movies = movies.copy()
    movies["genre_text"] = movies["genres"].str.replace("|", " ", regex=False)

    # Aggregate tags per movie
    tag_agg = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index()
        .rename(columns={"tag": "tag_text"})
    )

    movies = movies.merge(tag_agg, on="movieId", how="left")
    movies["tag_text"] = movies["tag_text"].fillna("")
    movies["content"] = movies["genre_text"] + " " + movies["tag_text"]

    return movies


def compute_ratings_stats(ratings):
    """Compute average rating and count per movie."""
    stats = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        num_ratings=("rating", "count"),
    ).reset_index()
    stats["avg_rating"] = stats["avg_rating"].round(2)
    return stats


def build_similarity(movies):
    """Compute TF-IDF vectors and pairwise cosine similarity."""
    print("Computing TF-IDF vectors ...")
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies["content"])

    print("Computing cosine similarity matrix ...")
    sim_matrix = cosine_similarity(tfidf_matrix)

    return sim_matrix


def extract_top_similar(movies, sim_matrix, top_n=TOP_N):
    """For each movie, extract the top N most similar movies."""
    movie_ids = movies["movieId"].values
    similarity_map = {}

    for idx in range(len(movie_ids)):
        scores = list(enumerate(sim_matrix[idx]))
        # Sort by score descending, skip self
        scores.sort(key=lambda x: x[1], reverse=True)
        top = []
        for other_idx, score in scores:
            if other_idx == idx:
                continue
            top.append((int(movie_ids[other_idx]), round(float(score), 4)))
            if len(top) >= top_n:
                break
        similarity_map[int(movie_ids[idx])] = top

    return similarity_map


def main():
    print("Loading data ...")
    movies, tags, ratings = load_data()

    print("Building content profiles ...")
    movies = build_content_profiles(movies, tags)

    print("Computing rating statistics ...")
    stats = compute_ratings_stats(ratings)
    movies = movies.merge(stats, on="movieId", how="left")
    movies["avg_rating"] = movies["avg_rating"].fillna(0)
    movies["num_ratings"] = movies["num_ratings"].fillna(0).astype(int)

    sim_matrix = build_similarity(movies)

    print("Extracting top similar movies ...")
    similarity_map = extract_top_similar(movies, sim_matrix)

    # Prepare movies DataFrame for saving (drop working columns)
    movies_out = movies[["movieId", "title", "genres", "avg_rating", "num_ratings"]].copy()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    movies_path = os.path.join(OUTPUT_DIR, "movies.pkl")
    sim_path = os.path.join(OUTPUT_DIR, "similarity.pkl")

    with open(movies_path, "wb") as f:
        pickle.dump(movies_out, f)
    with open(sim_path, "wb") as f:
        pickle.dump(similarity_map, f)

    print(f"Saved {len(movies_out)} movies to {movies_path}")
    print(f"Saved similarity map to {sim_path}")
    print("Done!")


if __name__ == "__main__":
    main()
