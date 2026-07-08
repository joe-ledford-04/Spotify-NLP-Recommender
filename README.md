# Spotify Lyric Recommendation System

> An empirical comparison of classical and transformer-based NLP methods for lyric-based music recommendation.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn)
![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-BERT-red)
![Spotify API](https://img.shields.io/badge/Spotify-Web_API-1DB954?logo=spotify)
![Genius API](https://img.shields.io/badge/Genius-API-yellow)

---

# Overview

Modern recommendation systems rely on rich representations of user preferences and item similarity. While commercial music platforms leverage collaborative filtering and proprietary listening data, this project investigates how well **song lyrics alone** can be used to recommend similar music.

Rather than implementing a single recommendation model, this project compares multiple Natural Language Processing (NLP) techniques to determine how different text representations affect recommendation quality. Classical vector-space models and modern transformer embeddings are evaluated on the same corpus of songs to better understand the strengths and limitations of lyric-based recommendation systems.

---

# Research Question

> **How does the choice of NLP representation influence the quality of lyric-based music recommendations?**

To answer this question, four independent recommendation pipelines were implemented and evaluated using the same dataset and test songs.

---

# Dataset

The recommendation corpus was built from my personal Spotify library.

- ~2,000 liked Spotify songs
- Song metadata collected using the Spotify Web API
- Lyrics retrieved through the Genius API
- Lyrics cleaned and preprocessed before modeling

This project includes a complete data collection pipeline for building the recommendation dataset from scratch.

---

# Project Pipeline

```
Spotify API
        │
        ▼
Collect Liked Songs
        │
        ▼
Retrieve Lyrics (Genius API)
        │
        ▼
Clean & Preprocess Lyrics
        │
        ▼
Generate Text Representations
        │
        ▼
Compute Similarity Scores
        │
        ▼
Return Song Recommendations
```

---

# Models Evaluated

## 1. TF-IDF

Serves as the baseline recommendation model.

Songs are represented as sparse vectors using Term Frequency–Inverse Document Frequency (TF-IDF), and recommendations are generated using cosine similarity.

**Purpose**

- Baseline lexical similarity
- Fast and interpretable
- Sensitive to exact word overlap

---

## 2. Chorus TF-IDF

This experiment investigates whether restricting the corpus to only each song's chorus improves recommendation quality.

The hypothesis is that choruses contain the most repetitive and semantically meaningful information.

---

## 3. Latent Semantic Analysis (LSA)

LSA applies Truncated Singular Value Decomposition (SVD) to TF-IDF vectors to learn lower-dimensional latent semantic representations.

Compared to TF-IDF, LSA captures relationships between words that do not necessarily co-occur directly.

---

## 4. LyricsBERT

The final model replaces sparse vector representations with contextual transformer embeddings using the pre-trained **LyricsBERT** model.

Unlike TF-IDF and LSA, transformer embeddings capture semantic meaning rather than simple lexical overlap.

---

# Evaluation

All models were evaluated using identical seed songs across multiple genres.

| Genre | Seed Song |
|--------|-----------|
| Grunge | *Iron Clad Lou* — Hum |
| Soul | *Everybody Loves the Sunshine* — Roy Ayers Ubiquity |
| Latin | *Volare* — Gipsy Kings |
| Hip-Hop | *House Money* — Baby Keem |

Keeping the evaluation consistent allowed qualitative comparisons between recommendation approaches while isolating the effect of different NLP representations.

---

# Results

The comparison produced several interesting observations.

- Lyrics alone provide limited information for generating high-quality music recommendations.
- Transformer embeddings improved semantic understanding but were still constrained by the information contained solely within lyrics.
- Hip-hop and Latin music consistently produced higher similarity scores across every model.
- Soul and grunge performed noticeably worse across all approaches.

These results suggest that lyrical repetition and vocabulary play a larger role in genres such as hip-hop and Latin music, while genres like soul and grunge may be better distinguished through musical composition rather than textual content.

---

# Technical Challenges

Some of the most significant engineering challenges included:

- Building an end-to-end data collection pipeline across multiple APIs
- Managing deprecated Spotify API functionality
- Extracting and preprocessing lyrics from external sources
- Isolating chorus sections for experimentation
- Comparing fundamentally different NLP representations under identical evaluation conditions

---

# Repository Structure

```
.
├── data_collection.ipynb
├── tfidf_recommender.ipynb
├── chorus_recommender.ipynb
├── lsa_recommender.ipynb
├── lyricsBERT_recommender.ipynb
└── README.md
```

---

# Future Improvements

Potential extensions include:

- Approximate nearest-neighbor search using FAISS
- Hybrid recommendation systems combining lyrics and audio features
- Incorporating Spotify audio analysis features
- Quantitative evaluation metrics beyond qualitative comparison
- Deploying the recommendation engine as a web application

---

# Key Takeaways

This project demonstrates how different NLP representations influence semantic similarity in a recommendation system. By implementing and comparing multiple approaches—from TF-IDF to transformer embeddings—it highlights both the capabilities and limitations of lyric-based recommendation systems while providing hands-on experience with data collection, feature engineering, modern NLP, and recommendation algorithms.
