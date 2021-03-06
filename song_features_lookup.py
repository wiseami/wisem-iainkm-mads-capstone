import requests
import pandas as pd
import utils
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

# Load Spotify API info
headers, market, BASE_URL = utils.spotify_info()

# Read in our csv lookup with all 69 Daily Song Charts
playlist_scrape_lookup = pd.read_csv('playlist_data/playlist_data.csv')
playlist_lookup = pd.read_csv('lookups/global_top_daily_playlists.csv')
unique_tracks = playlist_scrape_lookup['track_id'].unique().tolist()
existing_audio_features_lookup = pd.read_csv('lookups/track_audio_features.csv')
unique_tracks = [i for i in unique_tracks if i not in existing_audio_features_lookup['track_id'].tolist()]

# Create batches of 99 from unique_tracks to pass to the API for rate limit efficiency
def split(input_list, batch_size):
    for i in range(0, len(unique_tracks), batch_size):
        yield input_list[i:i + batch_size]

batch_size = 49
unique_tracks_for_api = list(split(unique_tracks, batch_size))
final_df = pd.DataFrame(columns=['id','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo_1','duration_ms','time_signature','update_dttm','name','artist','album_img','preview_url','popularity'])

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

# Now for librosa lookup
spotify_audio_file = 'lookups/track_audio_features.csv'
MIR_file = 'lookups/MIR_features.csv'

utils.get_librosa_features(spotify_audio_file, MIR_file, 'track_id', 'lookups/full_audio_features.csv')

# Reload full file for clustering
full_audio_feats = pd.read_csv('lookups/full_audio_features.csv')

# Create/Fit Scaler and KMeans on audio features
if len(full_audio_feats) > 0:
    X = full_audio_feats.drop(columns=['track_id','duration_ms','update_dttm','time_signature','name','artist','album_img','preview_url', 'popularity'])
    if 'basic_kmeans_cluster' in X.columns:
        X.drop(columns=['basic_kmeans_cluster'], inplace=True)
    if 'advanced_kmeans_cluster' in X.columns:
        X.drop(columns=['advanced_kmeans_cluster'], inplace=True)

    X_small = X.drop(columns=['chroma', 'chroma_cens', 'mff', 'spectral_centroid', 'spectral_bandwidth', 'spectral_contrast', 'spectral_flatness', 'Spectral_Rolloff', 'poly_features', 'tonnetz', 'ZCR', 'onset_strength', 'pitch', 'magnitude', 'tempo_2'])
    basic_scaler = StandardScaler().fit(X_small)
    data_scaled = basic_scaler.transform(X_small)
    X_scaled = pd.DataFrame(data_scaled)

    basic_k_scores = utils.kmeans_k_tuning(X_scaled, 2, 16)
    basic_k_scores.to_csv('model/basic_kmeans_inertia.csv', index=False)

    basic_kmeans = KMeans(n_clusters=10, n_init=30, max_iter=500)
    basic_kmeans.fit(X_scaled)

    # dump models to pickles for later use
    pickle.dump(basic_scaler, open("model/basic_scaler.pkl", "wb"))
    pickle.dump(basic_kmeans, open("model/basic_kmeans.pkl", "wb"))

    X_adv = X[X['spectral_contrast'].notna()]
    adv_scaler = StandardScaler().fit(X_adv)
    adv_data_scaled = adv_scaler.transform(X_adv)
    X_adv_scaled = pd.DataFrame(adv_data_scaled)

    adv_k_scores = utils.kmeans_k_tuning(X_adv_scaled, 2, 16)
    adv_k_scores.to_csv('model/adv_kmeans_inertia.csv', index=False)

    adv_kmeans = KMeans(n_clusters=11, n_init=30, max_iter=500)
    adv_kmeans.fit(X_adv_scaled)

    # dump models to pickles for later use
    pickle.dump(adv_scaler, open("model/adv_scaler.pkl", "wb"))
    pickle.dump(adv_kmeans, open("model/adv_kmeans.pkl", "wb"))

    # checking cluster size
    basic_clusters = basic_kmeans.predict(X_scaled)
    pd.Series(basic_clusters).value_counts().sort_index()

    adv_clusters = adv_kmeans.predict(X_adv_scaled)
    pd.Series(adv_clusters).value_counts().sort_index()

    audio_features_df_clustered = full_audio_feats.copy()
    audio_features_df_clustered["basic_kmeans_cluster"] = basic_clusters
    audio_features_df_clustered.to_csv('lookups/full_audio_features.csv', index=False)

    adv_audio_features_df_clustered = full_audio_feats.copy()
    adv_audio_features_df_clustered = adv_audio_features_df_clustered[adv_audio_features_df_clustered['spectral_contrast'].notna()]
    adv_audio_features_df_clustered["adv_kmeans_cluster"] = adv_clusters
    adv_audio_features_df_clustered.to_csv('lookups/all_track_audio_features.csv', index=False)
