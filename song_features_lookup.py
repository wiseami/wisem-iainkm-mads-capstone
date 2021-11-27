import requests
import pandas as pd
import datetime
import os
from os.path import exists
import time
import json
import itertools
import utils
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

with open('credentials.json') as creds:
    credentials = json.load(creds)

AUTH_URL = 'https://accounts.spotify.com/api/token'

auth_response = requests.post(AUTH_URL, {
    'grant_type': 'client_credentials',
    'client_id': credentials['CLIENT_ID'],
    'client_secret': credentials['CLIENT_SECRET'],
})

auth_response_data = auth_response.json()

access_token = auth_response_data['access_token']

headers = {
    'Authorization': 'Bearer {token}'.format(token=access_token)
}

# only for testing purposes. NEed to remove this later
market = '?market=US'

# base URL of all Spotify API endpoints
BASE_URL = 'https://api.spotify.com/v1/'

# Read in our csv lookup with all 69 Daily Song Charts
file_path = os.path.dirname(os.path.abspath(__file__)) + '\\'

playlist_scrape_lookup = pd.read_csv(file_path + 'playlist_data\\playlist_data.csv')
playlist_lookup = pd.read_csv(file_path + 'lookups\\global_top_daily_playlists.csv')

unique_tracks = playlist_scrape_lookup['track_id'].unique().tolist()
existing_audio_features_lookup = pd.read_csv(file_path + 'lookups\\track_audio_features.csv')
unique_tracks = [i for i in unique_tracks if i not in existing_audio_features_lookup['id'].tolist()]

# Create batches of 99 from unique_tracks to pass to the API for rate limit efficiency
def split(input_list, batch_size):
    for i in range(0, len(unique_tracks), batch_size):
        yield input_list[i:i + batch_size]

batch_size = 49

unique_tracks_for_api = list(split(unique_tracks, batch_size))

update_dttm = datetime.datetime.now()

final_df = pd.DataFrame(columns=['id','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature','update_dttm','name','artist','album_img','preview_url','popularity'])

# Pull audio features using track dict and write/append to file
for track_id_list in unique_tracks_for_api:
    req = requests.get(BASE_URL + 'audio-features?ids=' + (','.join(track_id_list)), headers=headers)
    feat = req.json()
    audio_features_df = utils.get_audio_features(feat)

    req = requests.get(BASE_URL + 'tracks?ids=' + (','.join(track_id_list)), headers=headers)
    feat = req.json()
    track_info_df = utils.get_track_info(feat)
    final_df = final_df.append(audio_features_df.merge(track_info_df, how='inner', on='id'), ignore_index=True)

final_df = existing_audio_features_lookup.append(audio_features_df.merge(track_info_df, how='inner', on='id')).reset_index(drop=True)

# Create/Fit Scaler and KMeans on audio features
if len(final_df) > 0:
    X = final_df.drop(columns=['id','duration_ms','update_dttm','time_signature','name','artist','album_img','preview_url', 'popularity'])
    scaler = StandardScaler().fit(X)
    data_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(data_scaled)

    k_scores = utils.kmeans_k_tuning(X_scaled, 2, 16)
    k_scores.to_csv('model/kmeans_inertia.csv', index=False)

    kmeans = KMeans(n_clusters=7) # 7 clusters was best as of 11/22/21
    kmeans.fit(X_scaled)

    # dump models to pickles for later use
    pickle.dump(scaler, open("model/scaler.pkl", "wb"))
    pickle.dump(kmeans, open("model/kmeans.pkl", "wb"))

    # checking cluster size
    clusters = kmeans.predict(X_scaled)
    pd.Series(clusters).value_counts().sort_index()
    audio_features_df_clustered = final_df.copy()
    audio_features_df_clustered["cluster"] = clusters
    audio_features_df_clustered.to_csv(file_path + 'lookups\\track_audio_features.csv', index=False)

# # For visualizing silhouette scores
# from yellowbrick.cluster import SilhouetteVisualizer
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(6, 2, figsize=(24,24))
# for i in [2, 3, 4, 5, 6,7,8,9,10,11,12]:
#     '''
#     Create KMeans instance for different number of clusters
#     '''
#     km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
#     q, mod = divmod(i, 2)
#     '''
#     Create SilhouetteVisualizer instance with KMeans instance
#     Fit the visualizer
#     '''
#     visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
#     visualizer.fit(X_scaled)