import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify API credentials
CLIENT_ID = "05a6e397541c49998b71feb88c01d75b"
CLIENT_SECRET = "8718d09692654b61a77733ef481e9ce2"
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(CLIENT_ID, CLIENT_SECRET))

# Load the dataset
data_path = "dataset.csv"  # Replace with your dataset path
df = pd.read_csv(data_path)

# Extract numerical features and scale
audio_features = df[["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(audio_features)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=10, random_state=42)
df["cluster"] = kmeans.fit_predict(scaled_features)

# Streamlit App Layout
st.set_page_config(page_title="Music Recommender", layout="wide")

# Sidebar
st.markdown("---")
st.markdown("**This App was Developed by Macsonic Inc**")
with st.sidebar:
    st.image("https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_CMYK_Green.png", width=200)
    st.title("Settings")
    num_recommendations = st.slider("Number of Recommendations", 1, 20, 10)

# Header


st.title("🎵 Spotify :green[Music] Recommender")
st.markdown("**Select a song to get personalized recommendations.**")

# Song Selection
selected_song = st.selectbox(
    "Choose a song from the dataset:",
    df["track_name"].unique()
)

# Recommendation Logic
def recommend_songs(song_name, num_recs):
    # Get the cluster of the selected song
    selected_row = df[df["track_name"] == song_name].iloc[0]
    selected_cluster = selected_row["cluster"]

    # Filter songs in the same cluster
    cluster_songs = df[df["cluster"] == selected_cluster]
    
    # Compute cosine similarity within the cluster
    selected_features = cluster_songs.loc[cluster_songs["track_name"] == song_name, audio_features.columns].values
    similarities = cosine_similarity(selected_features, cluster_songs[audio_features.columns])
    
    # Get top recommendations
    cluster_songs["similarity"] = similarities[0]
    recommendations = cluster_songs.sort_values("similarity", ascending=False).iloc[1:num_recs*2]  # Get more to ensure uniqueness
    
    # Fetch additional data using Spotify API
    recommended_songs = []
    seen_tracks = set()
    for _, row in recommendations.iterrows():
        if row["track_id"] not in seen_tracks:
            try:
                track = sp.track(row["track_id"])
                recommended_songs.append({
                    "name": row["track_name"],
                    "artist": row["artists"],
                    "image": track["album"]["images"][0]["url"] if track["album"]["images"] else None,
                    "preview_url": track["preview_url"]
                })
                seen_tracks.add(row["track_id"])
                if len(recommended_songs) >= num_recs:
                    break
            except:
                pass
    return recommended_songs

# Display Recommendations
if st.button("Recommend"):
    recommendations = recommend_songs(selected_song, num_recommendations)
    
    if recommendations:
        col1, col2, col3 = st.columns(3)
        for idx, rec in enumerate(recommendations):
            with [col1, col2, col3][idx % 3]:
                if rec["image"]:
                    st.image(rec["image"], use_container_width=True)
                st.markdown(f"**{rec['name']}** by {rec['artist']}")
                if rec["preview_url"]:
                    st.audio(rec["preview_url"], format="audio/mp3")
                else:
                    st.write("No preview available.")
    else:
        st.warning("No recommendations available.")