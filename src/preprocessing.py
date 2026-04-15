import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Phrases that appear in the Genius annotation/editorial content but never in lyrics
PROSE_SIGNALS = [
    # News / journalism
    r'according to',
    r'in an interview',
    r'told reporters',
    r'press release',
    r'announced that',
    r'the new york times',
    r'rolling stone',
    # Blog / list content
    r'honorable mention',
    r'on this track',
    r'produced by',
    r'sample from',
    r'interpolates',
    r'genius annotation',
    r'this song',
    r'the music video',
    r'feat\.',
    # Competition / chart results (catches Croatian song contest etc.)
    r'\(\d+p\)',          # score format like "(172p)"
    r'semi.?final',
    r'grand final',
    # Speeches / commencement addresses
    r'applause',
    r'laughter',
    r'thank you all',
    r'class of \d{4}',
    # Novel / prose excerpts
    r'said to me',
    r'chapter \d',
    r'he said\b',
    r'she said\b',
    r'replied\b',
    r'murmured\b',
]

PROSE_PATTERN = re.compile('|'.join(PROSE_SIGNALS), re.IGNORECASE)

# Metadata contamination
METADATA_WORDS = re.compile(
    r'discogs|bootleg|remaster|deluxe edition|track listing|work in progress'
    r'|genius japan|genius annotation|translated by|translation by'
    r'|\bproducer\b|\balbum:\b|\blabel:\b|\brelease date\b',
    re.IGNORECASE
)

def is_likely_lyrics(raw_text):
    # Signal 1: hard disqualifiers
    if PROSE_PATTERN.search(raw_text):
        return False

    # Signal 2: type-token ratio on first 200 raw words
    words = re.findall(r'[a-zA-Z]+', raw_text.lower())[:200]
    if len(words) >= 100:
        ttr = len(set(words)) / len(words)
        if ttr > 0.85:
            return False

    return True

def clean_lyrics(text: str):
    text = text.lower()

    # remove Genius footer variations (e.g. "2EmbedShare URLCopyEmbedCopy")
    text = re.sub(r'\d*embed.*$', '', text, flags=re.IGNORECASE)

    # remove section headers just in case any remain
    text = re.sub(r'\[.*?\]', '', text)

    # remove punctuation (but not spaces)
    text = re.sub(r'[^\w\s]', '', text)

    # remove underscores left by \w
    text = text.replace('_', '')

    # collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # tokenize
    tokens = text.split()

    # remove stopwords
    tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS]

    # lemmatize
    lemmatizer = WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)

def preprocess(lyrics_path: str, output_path: str) -> pd.DataFrame:
    lyrics_df = pd.read_csv(lyrics_path)

    lyrics_df = lyrics_df[~(
        (lyrics_df["artist"].str.lower() == "ryo fukui") &
        (lyrics_df["track_name"].str.lower() == "my conception")
    )].copy()
    # drop rows with no lyrics fetched
    lyrics_df = lyrics_df.dropna(subset=["lyrics"])

    # droping "lyrics" that are actually poor web page scraping from the api
    metadata_words = r'discogs|bootleg|remaster|deluxe edition|track listing|work in progress'
    lyrics_df = lyrics_df[~lyrics_df['lyrics'].str.contains(metadata_words, case=False, na=False)]

    # clean
    lyrics_df["processed_lyrics"] = lyrics_df["lyrics"].apply(clean_lyrics)

    # drop rows that became empty after cleaning
    lyrics_df = lyrics_df[lyrics_df["processed_lyrics"].str.strip() != ""]

    # drop songs with suspicisouly short processed lyrics (likely metadata)
    lyrics_df = lyrics_df[lyrics_df["processed_lyrics"].str.split().str.len() >= 30]

    # drop extreme outliers (found a 220,060 token outlier in intial testing)
    lyrics_df = lyrics_df[lyrics_df["processed_lyrics"].str.split().str.len() <= 2000]
    
    # drop exact duplicates in processed lyrics 
    lyrics_df = lyrics_df.drop_duplicates(subset=['processed_lyrics'])

    lyrics_df["token_count"] = lyrics_df["processed_lyrics"].apply(lambda x: len(x.split()))
    lyrics_df = lyrics_df[lyrics_df["token_count"] >= 50]  

    lyrics_df = lyrics_df.reset_index(drop=True)
    lyrics_df.to_csv(output_path, index=False)
    print(f"Saved {len(lyrics_df)} cleaned songs to {output_path}")

    return lyrics_df

if __name__ == "__main__":
    preprocess(
        lyrics_path="../data/raw/lyrics.csv",
        output_path="../data/processed/lyrics_clean.csv"
    )