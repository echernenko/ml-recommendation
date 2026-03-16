"""Flask web application for browsing movie recommendations."""

import os
import pickle
from collections import defaultdict

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-in-production"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

# Loaded on startup
MOVIES = None  # DataFrame
SIMILARITY = None  # dict: movieId -> [(movieId, score), ...]
MOVIES_BY_ID = None  # dict: movieId -> row dict


def load_data():
    global MOVIES, SIMILARITY, MOVIES_BY_ID

    movies_path = os.path.join(OUTPUT_DIR, "movies.pkl")
    sim_path = os.path.join(OUTPUT_DIR, "similarity.pkl")

    with open(movies_path, "rb") as f:
        MOVIES = pickle.load(f)
    with open(sim_path, "rb") as f:
        SIMILARITY = pickle.load(f)

    MOVIES_BY_ID = {row["movieId"]: dict(row) for _, row in MOVIES.iterrows()}


load_data()

ITEMS_PER_PAGE = 24


@app.context_processor
def inject_liked():
    liked = session.get("liked", [])
    return {"liked_ids": liked, "liked_count": len(liked)}


@app.route("/")
def index():
    q = request.args.get("q", "").strip()
    page = request.args.get("page", 1, type=int)

    df = MOVIES.copy()

    if q:
        df = df[df["title"].str.contains(q, case=False, na=False)]

    df = df.sort_values("num_ratings", ascending=False)

    total = len(df)
    total_pages = max(1, (total + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
    page = max(1, min(page, total_pages))
    start = (page - 1) * ITEMS_PER_PAGE
    movies = df.iloc[start : start + ITEMS_PER_PAGE].to_dict("records")

    return render_template(
        "index.html",
        movies=movies,
        query=q,
        page=page,
        total_pages=total_pages,
    )


@app.route("/movie/<int:movie_id>")
def movie_detail(movie_id):
    movie = MOVIES_BY_ID.get(movie_id)
    if movie is None:
        return "Movie not found", 404

    similar_ids = SIMILARITY.get(movie_id, [])[:10]
    similar = []
    for sim_id, score in similar_ids:
        m = MOVIES_BY_ID.get(sim_id)
        if m is not None:
            similar.append({**m, "score": score})

    return render_template("movie.html", movie=movie, similar=similar)


@app.route("/like/<int:movie_id>", methods=["POST"])
def toggle_like(movie_id):
    liked = session.get("liked", [])
    if movie_id in liked:
        liked.remove(movie_id)
    else:
        liked.append(movie_id)
    session["liked"] = liked

    next_url = request.form.get("next") or request.referrer or url_for("index")
    return redirect(next_url)


@app.route("/recommendations")
def recommendations():
    liked = session.get("liked", [])
    if not liked:
        return render_template("recommendations.html", movies=[], liked_any=False)

    # Aggregate similarity scores from all liked movies
    scores = defaultdict(float)
    for mid in liked:
        for sim_id, score in SIMILARITY.get(mid, []):
            if sim_id not in liked:
                scores[sim_id] += score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
    movies = []
    for mid, agg_score in ranked:
        m = MOVIES_BY_ID.get(mid)
        if m is not None:
            movies.append({**m, "agg_score": round(agg_score, 3)})

    return render_template("recommendations.html", movies=movies, liked_any=True)


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])

    df = MOVIES[MOVIES["title"].str.contains(q, case=False, na=False)]
    df = df.sort_values("num_ratings", ascending=False).head(10)
    results = df[["movieId", "title", "genres", "avg_rating", "tags"]].to_dict("records")
    return jsonify(results)
