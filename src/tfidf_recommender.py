import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_tfidf_recommender(lyrics_path: str):
    lyrics_df = pd.read_csv(lyrics_path)
    lyrics_df = lyrics_df.reset_index(drop=True)  # ensure index aligns with matrix rows

    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2
    )
    tfidf_matrix = tfidf.fit_transform(lyrics_df["processed_lyrics"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return lyrics_df, tfidf_matrix, similarity_matrix

def recommend_songs(lyrics_df: pd.DataFrame, similarity_matrix, seed_title: str,
                    seed_artist: str = None, k: int = 5) -> pd.DataFrame:
    matches = lyrics_df[lyrics_df["track_name"].str.lower() == seed_title.lower()]
    if seed_artist:
        matches = matches[matches["artist"].str.lower() == seed_artist.lower()]
    if matches.empty:
        raise ValueError(f"Seed song '{seed_title}' not found.")

    seed_idx = matches.index[0]
    similarity_scores = similarity_matrix[seed_idx]
    similar_indices = similarity_scores.argsort()[::-1]
    similar_indices = [i for i in similar_indices if i != seed_idx][:k]

    recommendations = lyrics_df.iloc[similar_indices][["track_name", "artist"]].copy()
    recommendations["similarity"] = similarity_scores[similar_indices]
    recommendations = recommendations.reset_index(drop=True)
    return recommendations

if __name__ == "__main__":
    lyrics_df, tfidf_matrix, similarity_matrix = build_tfidf_recommender(
        "../data/processed/lyrics_clean.csv"
    )
    recs = recommend_songs(
        lyrics_df,
        similarity_matrix,
        seed_title="Take a Trip",
        seed_artist="TV Girl",
        k=5
    )
    print(recs)

