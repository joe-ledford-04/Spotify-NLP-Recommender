import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# HuggingFace model card: https://huggingface.co/brunokreiner/lyrics-bert/tree/main
LYRICBERT_MODEL = "brunokreiner/lyrics-bert"
CACHE_FILENAME = "lyricbert_embeddings.npy"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )


def generate_embeddings(
    lyrics_df,
    cache_path,
    model_name=LYRICBERT_MODEL,
    batch_size=8,
    max_length=256
):
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Running on: {device}")

    texts = lyrics_df["processed_lyrics"].fillna("").astype(str).tolist()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        encoded_input = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = model(**encoded_input)

        batch_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        all_embeddings.append(batch_embeddings.cpu().numpy())

        print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} songs")

    embedding_matrix = np.vstack(all_embeddings)
    np.save(cache_path, embedding_matrix)
    print(f"Embeddings saved to {cache_path} | shape: {embedding_matrix.shape}")
    return embedding_matrix


def build_lyricbert_recommender(
    lyrics_path,
    cache_dir="../data/processed",
    model_name=LYRICBERT_MODEL,
    force_recompute=False
):
    lyrics_df = pd.read_csv(lyrics_path)
    lyrics_df = lyrics_df.drop_duplicates(subset=["track_name", "artist"]).reset_index(drop=True)

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, CACHE_FILENAME)

    if os.path.exists(cache_path) and not force_recompute:
        embeddings = np.load(cache_path)
        print(f"Loaded cached embeddings from {cache_path} | shape: {embeddings.shape}")

        if embeddings.shape[0] != len(lyrics_df):
            print("Cache size mismatch. Recomputing embeddings.")
            embeddings = generate_embeddings(lyrics_df, cache_path, model_name=model_name)
    else:
        embeddings = generate_embeddings(lyrics_df, cache_path, model_name=model_name)

    similarity_matrix = cosine_similarity(embeddings)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    return lyrics_df, embeddings, similarity_matrix


def recommend_songs(lyrics_df, similarity_matrix, seed_title, seed_artist=None, k=10):
    matches = lyrics_df[lyrics_df["track_name"].str.lower() == seed_title.lower()]
    if seed_artist:
        matches = matches[matches["artist"].str.lower() == seed_artist.lower()]

    if matches.empty:
        raise ValueError(f"Seed song '{seed_title}' not found in corpus.")

    seed_idx = matches.index[0]
    scores = similarity_matrix[seed_idx]
    ranked = scores.argsort()[::-1]
    ranked = [i for i in ranked if i != seed_idx][:k]

    recs = lyrics_df.iloc[ranked][["track_name", "artist"]].copy()
    recs["similarity"] = scores[ranked]
    return recs.reset_index(drop=True)


if __name__ == "__main__":
    lyrics_df, embeddings, similarity_matrix = build_lyricbert_recommender(
        lyrics_path="../data/processed/lyrics_clean.csv",
        cache_dir="../data/processed"
    )

    recs = recommend_songs(
        lyrics_df,
        similarity_matrix,
        seed_title="House Money",
        seed_artist="Baby Keem",
        k=10
    )
    print(recs)
