import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity


def build_lsa_recommender(lyrics_path, n_components = 100):
    lyrics_df = pd.read_csv(lyrics_path)
    lyrics_df = lyrics_df.drop_duplicates(subset=["track_name", "artist"])
    lyrics_df = lyrics_df.reset_index(drop=True)

    # TF-IDF matrix (same implementation as tfidf_recommender.py)
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2
    )
    tfidf_matrix = vectorizer.fit_transform(lyrics_df["processed_lyrics"])

    # SVD to reduce to n_components latent dimensions
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_matrix = svd.fit_transform(tfidf_matrix)

    # Normalizing rows for cosine similarity 
    lsa_matrix = Normalizer(copy=False).fit_transform(lsa_matrix)

    # Computing cosine similarity on the dense LSA matrix
    similarity_matrix = cosine_similarity(lsa_matrix)

    return lyrics_df, lsa_matrix, similarity_matrix, svd, vectorizer


def recommend_songs(lyrics_df, similarity_matrix,
                    seed_title, seed_artist =  None, k = 10):
   
    matches = lyrics_df[lyrics_df["track_name"].str.lower() == seed_title.lower()]
    if seed_artist:
        matches = matches[matches["artist"].str.lower() == seed_artist.lower()]
    if matches.empty:
        raise ValueError(f"Seed song '{seed_title}' not found in dataset.")

    seed_idx = matches.index[0]
    similarity_scores = similarity_matrix[seed_idx]
    similar_indices = similarity_scores.argsort()[::-1]
    similar_indices = [i for i in similar_indices if i != seed_idx][:k]

    recommendations = lyrics_df.iloc[similar_indices][["track_name", "artist"]].copy()
    recommendations["similarity"] = similarity_scores[similar_indices]
    recommendations = recommendations.reset_index(drop=True)
    return recommendations


def get_top_topics(svd, vectorizer,
                   n_topics = 10, n_terms = 10):
    terms = vectorizer.get_feature_names_out()
    topic_dict = {}
    for i, component in enumerate(svd.components_[:n_topics]):
        top_term_indices = component.argsort()[::-1][:n_terms]
        topic_dict[f"Topic {i+1}"] = [terms[j] for j in top_term_indices]
    return pd.DataFrame(topic_dict)


def get_song_top_topics(lyrics_df, lsa_matrix,
                        seed_title, seed_artist=None,
                        n_topics=5):
    matches = lyrics_df[lyrics_df["track_name"].str.lower() == seed_title.lower()]
    if seed_artist:
        matches = matches[matches["artist"].str.lower() == seed_artist.lower()]
    if matches.empty:
        raise ValueError(f"Seed song '{seed_title}' not found in dataset.")

    seed_idx = matches.index[0]
    topic_scores = lsa_matrix[seed_idx]
    top_topic_indices = np.abs(topic_scores).argsort()[::-1][:n_topics]

    return pd.Series(
        {f"Topic {i+1}": round(topic_scores[i], 4) for i in top_topic_indices}
    )


if __name__ == "__main__":
    lyrics_df, lsa_matrix, similarity_matrix, svd, vectorizer = build_lsa_recommender(
        "../data/processed/lyrics_clean.csv",
        n_components=100
    )

    recs = recommend_songs(
        lyrics_df,
        similarity_matrix,
        seed_title="House Money",
        seed_artist="Baby Keem",
        k=10
    )
    print(recs)

    print("\nTop 5 latent topics learned:")
    print(get_top_topics(svd, vectorizer, n_topics=5, n_terms=8).to_string())
