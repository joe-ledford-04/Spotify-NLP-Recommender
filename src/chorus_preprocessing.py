import re
import pandas as pd
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

SECTION_PATTERN = re.compile(r'\[([^\]]+)\]')
CHORUS_LABELS = {"chorus", "hook", "refrain"}

PROSE_SIGNALS = re.compile(
    r'according to|in an interview|told reporters|press release|announced that'
    r'|the new york times|rolling stone|honorable mention|on this track'
    r'|produced by|sample from|interpolates|genius annotation|this song'
    r'|the music video|feat\.'
    r'|\(\d+p\)|semi.?final|grand final'
    r'|applause|laughter|thank you all|class of \d{4}'
    r'|said to me|chapter \d|he said\b|she said\b|replied\b|murmured\b',
    re.IGNORECASE
)

METADATA_WORDS = re.compile(
    r'discogs|bootleg|remaster|deluxe edition|track listing|work in progress'
    r'|genius japan|genius annotation|translated by|translation by'
    r'|\bproducer\b|\balbum:\b|\blabel:\b|\brelease date\b',
    re.IGNORECASE
)

def is_likely_lyrics(raw_text):
    if PROSE_SIGNALS.search(raw_text):
        return False
    words = re.findall(r'[a-zA-Z]+', raw_text.lower())[:200]
    if len(words) >= 100:
        ttr = len(set(words)) / len(words)
        if ttr > 0.85:
            return False
    return True

def extract_sections(raw_lyrics):
    sections = {}
    current_label = None
    current_lines = []

    for line in raw_lyrics.splitlines():
        header_match = SECTION_PATTERN.match(line.strip())
        if header_match:
            # Save previous section block before starting a new one
            if current_label is not None and current_lines:
                text = "\n".join(current_lines).strip()
                if text:
                    sections.setdefault(current_label, []).append(text)
            raw_label = header_match.group(1).strip().lower()
            current_label = re.sub(r'\s*\d+\s*$', '', raw_label).strip()
            current_lines = []
        else:
            if current_label is not None:
                current_lines.append(line)

    # Flush the last block
    if current_label is not None and current_lines:
        text = "\n".join(current_lines).strip()
        if text:
            sections.setdefault(current_label, []).append(text)

    return sections


def get_chorus_text(raw_lyrics):
    sections = extract_sections(raw_lyrics)

    chorus_blocks = []
    for label, blocks in sections.items():
        if any(kw in label for kw in CHORUS_LABELS):
            chorus_blocks.extend(blocks)

    return "\n".join(chorus_blocks) if chorus_blocks else None

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d*embed.*$', '', text, flags=re.IGNORECASE)
    text = SECTION_PATTERN.sub('', text)  # strip any leftover headers
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('_', '')
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


def preprocess_chorus(raw_lyrics_path, output_path):
    df = pd.read_csv(raw_lyrics_path)
    df = df.dropna(subset=["lyrics"])
    df = df.drop_duplicates(subset=["track_name", "artist"])

    df = df[~df['lyrics'].str.contains(METADATA_WORDS, na=False)]
    before = len(df)
    df = df[df['lyrics'].apply(is_likely_lyrics)]
    prose_dropped = before - len(df)
    print(f"Dropped {prose_dropped} songs flagged as prose/annotation content")

    chorus_texts = []
    processed    = []
    keep_indices = []

    for idx, row in df.iterrows():
        chorus_raw = get_chorus_text(str(row["lyrics"]))

        if chorus_raw is None:
            continue                     

        cleaned = clean_text(chorus_raw)
        if not cleaned.strip():
            continue                   

        chorus_texts.append(chorus_raw)
        processed.append(cleaned)
        keep_indices.append(idx)

    df = df.loc[keep_indices].copy()
    df["chorus_text"]      = chorus_texts
    df["processed_chorus"] = processed

    # Token floor — lower than full lyric floor since choruses are short
    df["token_count"] = df["processed_chorus"].apply(lambda x: len(x.split()))
    df = df[df["token_count"] >= 10]
    df = df.reset_index(drop=True)

    n_original = len(pd.read_csv(raw_lyrics_path).dropna(subset=["lyrics"]))
    n_kept     = len(df)
    n_skipped  = n_original - n_kept
    print(f"Songs kept (chorus found):  {n_kept}/{n_original} ({n_kept/n_original:.1%})")
    print(f"Songs skipped (no chorus):  {n_skipped}/{n_original} ({n_skipped/n_original:.1%})")
    print(f"Saved {n_kept} songs to {output_path}")

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    preprocess_chorus(
        raw_lyrics_path="../data/raw/lyrics.csv",
        output_path="../data/processed/chorus_clean.csv"
    )
