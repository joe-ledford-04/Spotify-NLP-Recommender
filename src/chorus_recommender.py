import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_chorus_recommender(chorus_path):
    lyrics_df = pd.read_csv(chorus_path)
    lyrics_df = lyrics_df.reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2
    )
    tfidf_matrix = vectorizer.fit_transform(lyrics_df["processed_chorus"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return lyrics_df, tfidf_matrix, similarity_matrix, vectorizer


def recommend_songs(lyrics_df, similarity_matrix,
                    seed_title, seed_artist = None,
                    k = 10):
    matches = lyrics_df[lyrics_df["track_name"].str.lower() == seed_title.lower()]
    if seed_artist:
        matches = matches[matches["artist"].str.lower() == seed_artist.lower()]
    if matches.empty:
        raise ValueError(f"Seed song '{seed_title}' not found in corpus.")

    seed_idx = matches.index[0]
    scores = similarity_matrix[seed_idx]
    ranked = scores.argsort()[::-1]
    ranked = [i for i in ranked if i != seed_idx][:k]

    recs = lyrics_df.iloc[ranked][["track_name", "artist", "chorus_fallback"]].copy()
    recs["similarity"] = scores[ranked]
    recs = recs.reset_index(drop=True)
    return recs


if __name__ == "__main__":
    lyrics_df, tfidf_matrix, similarity_matrix, vectorizer = build_chorus_recommender(
        "../data/processed/chorus_clean.csv"
    )
    recs = recommend_songs(
        lyrics_df, similarity_matrix,
        seed_title="House Money",
        seed_artist="Baby Keem",
        k=10
    )
    print(recs)
