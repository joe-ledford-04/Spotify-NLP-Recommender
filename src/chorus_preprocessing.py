import re
import pandas as pd
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

SECTION_PATTERN = re.compile(r'\[([^\]]+)\]')
CHORUS_LABELS = {"chorus", "hook", "refrain"}

def extract_sections(raw_lyrics):
    sections: dict[str, list[str]] = {}
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

    if chorus_blocks:
        return "\n".join(chorus_blocks), False

    # Fallback: no chorus tag found — return full lyrics (minus headers)
    full_text = SECTION_PATTERN.sub('', raw_lyrics)
    full_text = re.sub(r'\n{2,}', '\n', full_text).strip()
    return (full_text if full_text else None), True

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

    chorus_texts   = []
    processed      = []
    fallback_flags = []

    for _, row in df.iterrows():
        raw = str(row["lyrics"])
        chorus_raw, is_fallback = get_chorus_text(raw)

        if chorus_raw is None:
            chorus_texts.append(None)
            processed.append(None)
            fallback_flags.append(True)
            continue

        cleaned = clean_text(chorus_raw)
        chorus_texts.append(chorus_raw)
        processed.append(cleaned)
        fallback_flags.append(is_fallback)

    df = df.copy()
    df["chorus_text"]      = chorus_texts
    df["processed_chorus"] = processed
    df["chorus_fallback"]  = fallback_flags

    df = df[df["processed_chorus"].notna()]
    df = df[df["processed_chorus"].str.strip() != ""]

    df["token_count"] = df["processed_chorus"].apply(lambda x: len(x.split()))
    df = df[df["token_count"] >= 10]   # chorus floor is lower than full-lyric floor

    df = df.reset_index(drop=True)

    # Fallback rate
    n_total    = len(df)
    n_fallback = df["chorus_fallback"].sum()
    n_chorus   = n_total - n_fallback
    print(f"Songs with chorus tag:    {n_chorus}/{n_total} ({n_chorus/n_total:.1%})")
    print(f"Songs using fallback:     {n_fallback}/{n_total} ({n_fallback/n_total:.1%})")

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return df


if __name__ == "__main__":
    preprocess_chorus(
        raw_lyrics_path="../data/raw/lyrics.csv",
        output_path="../data/processed/chorus_clean.csv"
    )
