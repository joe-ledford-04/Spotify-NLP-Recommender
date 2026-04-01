import os
import time
import pandas as pd
import lyricsgenius 
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")

api = lyricsgenius.Genius(ACCESS_TOKEN)
api.remove_section_headers = False # keep [Chorus] etc. for Phase 2
api.verbose = False # supresses noisy search output
api.skip_non_songs = True # filters out interviews and annotations


def get_lyrics(metadata_path: str, output_path: str):
    """
    For each song in the metadata CSV, search Genius by title + artist
    Save lyrics CSV with columns: track_id, track_name, artist, lyrics
    """
    
    liked_songs = pd.read_csv(metadata_path)
    results = []

    for _, row in liked_songs.iterrows():
        track_id = row["track_id"]
        track_name = row["track_name"]
        artist = row["artist"]

        try:
            song = api.search_song(track_name, artist)

            if song is None:
                print(f"[NOT FOUND] {track_name} — {artist}")
                lyrics = None
            else:
                print(f"[FOUND] {track_name} — {artist}")
                lyrics = song.lyrics

        except Exception as e:
            print(f"[ERROR] {track_name} — {artist}: {e}")
            lyrics = None

        results.append({
            "track_id":   track_id,
            "track_name": track_name,
            "artist":     artist,
            "lyrics":     lyrics
        })

        time.sleep(0.5) # polite to the API

    lyrics_df = pd.DataFrame(results)
    lyrics_df.to_csv(output_path, index=False)
    print(f"\nDone. {lyrics_df['lyrics'].notna().sum()}/{len(lyrics_df)} songs fetched.")
    return lyrics_df


if __name__ == "__main__":
    get_lyrics(
        metadata_path="../data/raw/track_metadata.csv",
        output_path="../data/raw/lyrics.csv"
    )