import time
import pandas as pd 
import spotipy

# ---------------------------------------------------------------------------
# Liked Songs
# ---------------------------------------------------------------------------

def get_liked_songs(sp):
    results = []
    offset = 0
    limit = 50 # Spotify's max per request endpoint

    while True: 
        response = sp.current_user_saved_tracks(limit=limit, offset=offset)
        items = response["items"]
        if not items:
            break
        results.extend(items)
        offset += limit
        time.sleep(0.1)

    return results

def parse_tracks(liked_songs):
    data = []
    for item in liked_songs:
        track = item["track"]
        if track is None:
            continue # skips unavaiable tracks
        data.append({
            "track_id":     track["id"],
            "track_name":   track["name"],
            "artist":       track["artists"][0]["name"],
            "artist_id":    track["artists"][0]["id"],   # needed for genre lookup
            "album":        track["album"]["name"],
            "release_date": track["album"]["release_date"],
            "added_at":     item["added_at"]
        })
    return pd.DataFrame(data)

# ---------------------------------------------------------------------------
# Audio Features (Doesn't work, was depreciated)
# ---------------------------------------------------------------------------

# def get_audio_features(sp, track_ids):
#     features = []
#     batch_size = 100 # Spotify's max for audio_features endpoint

#     for i in range(0, len(track_ids), batch_size):
#         batch = track_ids[i: i +batch_size]
#         results = sp.audio_features(batch)
#         features.extend([f for f in results if f is not None])
#         time.sleep(0.1)
    
#     df = pd.DataFrame(features)

#     # drop Spotify internal fields not useful for modeling
#     cols_to_keep = [
#         "id", "danceability", "energy", "key", "loudness", "mode",
#         "speechiness", "acousticness", "instrumentalness", "liveness",
#         "valence", "tempo", "duration_ms", "time_signature"
#     ]

#     return df[cols_to_keep].rename(columns={"id": "track_id"})

# ---------------------------------------------------------------------------
# Artist Metadata (Doesn't work, was depreciated)
# ---------------------------------------------------------------------------

# def get_artist_metadata(sp, artist_ids):
#     artists = []
#     batch_size = 50 # Spotify's max for artists endpoint

#     for i in range(0, len(artist_ids), batch_size):
#         batch = artist_ids[i : i+batch_size]
#         results = sp.artists(batch)["artists"]
#         for a in results:
#             if a is None:
#                 continue
#             artists.append({
#                 "artist_id": a["id"],
#                 "artist_genres": ", ".join(a["genres"]),
#                 "artist_popularity": a["popularity"],
#                 "artist_followers": a["followers"]["total"]
#             })
#         time.sleep(0.1)

#     return pd.DataFrame(artists)

# ---------------------------------------------------------------------------
# Spotify Recommendations (Doesn't work, depreciated)
# ---------------------------------------------------------------------------

# def get_recommendations(sp, seed_ids, limit=100):
#     response = sp.recommendations(seed_tracks=[seed_ids], limit=limit)

#     tracks = []
#     for track in response["tracks"]:
#         tracks.append({
#             "track_id":     track["id"],
#             "track_name":   track["name"],
#             "artist":       track["artists"][0]["name"],
#             "artist_id":    track["artists"][0]["id"],
#             "album":        track["album"]["name"],
#             "release_date": track["album"]["release_date"]
#         })

#     return pd.DataFrame(tracks)