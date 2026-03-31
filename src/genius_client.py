import os
from dotenv import load_dotenv
import lyricsgenius as genius 
import pandas as pd

load_dotenv()

ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")

api = genius.Genius(ACCESS_TOKEN)

# Remoes section headers like [Chorus] for cleaner lyrics
api.remove_selection_headers = True

liked_songs = pd.read_csv("../data/raw/track_metadata.csv")


artists = liked_songs["artist"]

def get_lyrics(artists):

    for artist in artists:
        artist = api.search_artist(artist)
        lyrics = artist.save_lyrics()
    
    lyric_path = "../data/raw/lyrics.csv"
    lyrics.keys()
    songs = lyrics.get('songs')
    lyric_df = pd.DataFrame(columns=['name', 'lyrics'])
    for x in songs:
        lyric_df = lyric_df.append({
            'name': x.get('title'),
            'lyrics': x.get('lyrics')
        }, ignore_index=True)
    
    lyric_df.to_csv(lyric_path, index=False)

    


