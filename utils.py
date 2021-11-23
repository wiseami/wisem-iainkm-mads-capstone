import requests
import pandas as pd
import datetime
import os
from sklearn.metrics.pairwise import cosine_similarity
import sys
import streamlit as st
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pickle
# from os.path import exists

update_dttm = datetime.datetime.now()


def spotify_info():
    # with open('credentials.json') as creds:
    #    credentials = json.load(creds)

    # Spotify token auth url
    AUTH_URL = 'https://accounts.spotify.com/api/token'
    
    # uses secrets.toml for Streamlit
    auth_response = requests.post(AUTH_URL, {
        'grant_type': 'client_credentials',
        'client_id': st.secrets['spotify_credentials']['CLIENT_ID'],
        'client_secret': st.secrets['spotify_credentials']['CLIENT_SECRET'],
    })

    auth_response_data = auth_response.json()
    access_token = auth_response_data['access_token']
    headers = {
        'Authorization': 'Bearer {token}'.format(token=access_token)
    }

    market = '?market=US' #maybe unnecessary?

    # Base URL of all Spotify API endpoints 
    BASE_URL = 'https://api.spotify.com/v1/'

    return headers, market, BASE_URL


def load_data():
    if sys.platform == 'win32':
        file_path = os.path.dirname(os.path.abspath(__file__)) + '\\'
        top_pl_df = pd.read_csv(file_path + 'lookups\\global_top_daily_playlists.csv')
        audio_features_df = pd.read_csv(file_path + 'lookups\\track_audio_features.csv')
        playlist_data_df = pd.read_csv(file_path + 'playlist_data\\2021-11-19.csv')
        global_lookup = pd.read_csv(file_path + 'lookups\\global_top_daily_playlists.csv')
        kmeans_inertia = pd.read_csv(file_path + 'model\\kmeans_inertia.csv')
    else:
        file_path = os.path.dirname(os.path.abspath(__file__)) + '/'
        top_pl_df = pd.read_csv(file_path + 'lookups/global_top_daily_playlists.csv')
        audio_features_df = pd.read_csv(file_path + 'lookups/track_audio_features.csv')
        playlist_data_df = pd.read_csv(file_path + 'playlist_data/2021-11-19.csv')
        global_lookup = pd.read_csv(file_path + 'lookups/global_top_daily_playlists.csv')
        kmeans_inertia = pd.read_csv(file_path + 'model/kmeans_inertia.csv')
    
    pl_w_audio_feats_df = playlist_data_df.merge(audio_features_df, how='inner', left_on='track_id', right_on='id')
    pl_w_audio_feats_df = pl_w_audio_feats_df.drop(columns=['market','capture_dttm','track_preview_url','track_duration', 'id', 'track_added_date', 'track_popularity', 'track_number','time_signature', 'track_artist','track_name','track_id','name','artist','album_img','preview_url','update_dttm'])
 
    return file_path, top_pl_df, audio_features_df, playlist_data_df, global_lookup, pl_w_audio_feats_df, kmeans_inertia


def normalize_spotify_audio_feats(df):
    grouped = df.groupby(by=['country'], as_index=False)
    res = grouped.agg(['sum', 'count'])
    res.columns = list(map('_'.join, res.columns.values))
    res = res.reset_index()

    ### Create Spotify audio features normalized for playlist length
    res = res.drop(columns=['danceability_count', 'energy_count', 'key_count', 'loudness_count', 'mode_count', 'speechiness_count', 'acousticness_count', 'instrumentalness_count', 'liveness_count', 'valence_count', 'tempo_count'])
    res = res.rename(columns = {'duration_ms_count':'track_count'})
    res['duration_m'] = res['duration_ms_sum'] / 1000 / 60
    res['danceability'] = res['danceability_sum'] / res['duration_m']
    res['energy'] = res['energy_sum'] / res['duration_m']
    res['key'] = res['key_sum'] / res['duration_m']
    res['loudness'] = res['loudness_sum'] / res['duration_m']
    res['mode'] = res['mode_sum'] / res['duration_m']
    res['speechiness'] = res['speechiness_sum'] / res['duration_m']
    res['acousticness'] = res['acousticness_sum'] / res['duration_m']
    res['instrumentalness'] = res['instrumentalness_sum'] / res['duration_m']
    res['liveness'] = res['liveness_sum'] / res['duration_m']
    res['valence'] = res['valence_sum'] / res['duration_m']
    res['tempo'] = res['tempo_sum'] / res['duration_m']

    playlist_audio_feature_rollup = res.drop(columns=['danceability_sum', 'energy_sum', 'key_sum', 'loudness_sum', 'mode_sum', 'speechiness_sum', 'acousticness_sum', 'instrumentalness_sum', 'liveness_sum', 'valence_sum', 'tempo_sum', 'duration_ms_sum', 'track_count','duration_m'])
    return playlist_audio_feature_rollup


# Takes json result of 'audio-features?ids=' Spotify API call and return df
def get_audio_features(feat):
    track_list = dict()
    for track in feat['audio_features']:
        track_list[track['id']] = {'danceability' : track['danceability'], 
                                   'energy' : track['energy'],
                                   'key' : track['key'],
                                   'loudness' : track['loudness'],
                                   'mode' : track['mode'],
                                   'speechiness' : track['speechiness'],
                                   'acousticness' : track['acousticness'],
                                   'instrumentalness' : track['instrumentalness'],
                                   'liveness' : track['liveness'],
                                   'valence' : track['valence'],
                                   'tempo' : track['tempo'],
                                   'duration_ms' : track['duration_ms'],
                                   'time_signature' : track['time_signature'],
                                   'update_dttm' : update_dttm
                                  }
    
    audio_features_df = pd.DataFrame.from_dict(track_list, orient='index')
    audio_features_df.index.name = 'id'
    audio_features_df.reset_index(inplace=True)
    return audio_features_df



# Takes json result of 'tracks?ids=' Spotify API call and return df
def get_track_info(feat):
    track_list = dict()
    for track in feat['tracks']:
        track_list[track['id']] = {'name': track['name'],
                                   'artist': track['artists'][0]['name'],
                                   'album_img': track['album']['images'][0]['url'],
                                   #'artist_img': track['artists'][0]['images'][0]['url'],
                                   'preview_url': track['preview_url']
                                  }
    track_info_df = pd.DataFrame.from_dict(track_list, orient='index')
    track_info_df.index.name = 'id'
    track_info_df.reset_index(inplace=True)
    return track_info_df

# def kmeans_prepro_X_scaled(audio_features_df):
#     X = audio_features_df.drop(columns=['id','duration_ms','update_dttm','time_signature','name','artist','album_img','preview_url'])
#     if exists('model/scaler.pkl'):
#         scaler = pickle.load(open("model/scaler.pkl", "rb"))
#     else:
#         scaler = StandardScaler().fit(X)
#     data_scaled = scaler.transform(X)
#     X_scaled = pd.DataFrame(data_scaled)
#     return X_scaled

# KMeans functions
def kmeans_k_tuning(df, k_min, k_max):
    #calculating inertia and silhouette scores
    inertia, silhouette = [], []
    for k in range(k_min, k_max):
        kmeans = KMeans(n_clusters=k, random_state=99)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)
        silhouette.append(silhouette_score(df, kmeans.predict(df)))
        
    ine = kmeans_k_tuning_plots(k_min=k_min, k_max=k_max, inertia=inertia, silhouette=silhouette)
    
    # storing results in df
    results = pd.DataFrame({'k': list(range(k_min, k_max)), 'inertia': inertia, 'silhouette_score': silhouette})
    return results

def kmeans_k_tuning_plots(k_min, k_max, inertia, silhouette):
    # inertia plot
    plt.figure(figsize=(16,8))
    plt.plot(range(k_min, k_max), inertia, 'bx-')
    plt.title('inertia for k between ' + str(k_min) + ' and ' + str(k_max-1))
    plt.xlabel('k')
    plt.ylabel('inertia')
    plt.xticks(np.arange(k_min, k_max+1, 1.0))
    plt.show()
    
    # silhouette score plot
    plt.figure(figsize=(16,8))
    plt.plot(range(k_min, k_max), silhouette, 'bx-')
    plt.title('silhouette score for k between ' + str(k_min) + ' and ' + str(k_max-1))
    plt.xlabel('k')
    plt.ylabel('silhouette score')
    plt.xticks(np.arange(k_min, k_max+1, 1.0))
    plt.show()
    
    return inertia

def do_kmeans_on_fly(track_df):
    X = track_df.drop(columns=['id','duration_ms','update_dttm','time_signature','name','artist','album_img','preview_url'])
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    data_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(data_scaled)

    kmeans = pickle.load(open("model/kmeans.pkl", "rb"))
    clusters = kmeans.predict(X_scaled)
    audio_features_df_clustered = track_df.copy()
    audio_features_df_clustered["cluster"] = clusters
    return audio_features_df_clustered



# takes in dfs to do cossim on the fly
def create_cossim_df(df, res, global_lookup):
    """
    df = final_df
    res = df of normalized audio features by country/playlist
    global_lookup = df of global lookup csv
    """
    cossim_df = df.drop(columns=['update_dttm', 'time_signature', 'name','artist','album_img','preview_url', 'cluster'])
    cossim_df_y = cossim_df.id
    cossim_df['duration_m'] = cossim_df['duration_ms'] / 1000 / 60
    cossim_df['danceability'] = cossim_df['danceability'] / cossim_df['duration_m']
    cossim_df['energy'] = cossim_df['energy'] / cossim_df['duration_m']
    cossim_df['key'] = cossim_df['key'] / cossim_df['duration_m']
    cossim_df['loudness'] = cossim_df['loudness'] / cossim_df['duration_m']
    cossim_df['mode'] = cossim_df['mode'] / cossim_df['duration_m']
    cossim_df['speechiness'] = cossim_df['speechiness'] / cossim_df['duration_m']
    cossim_df['acousticness'] = cossim_df['acousticness'] / cossim_df['duration_m']
    cossim_df['instrumentalness'] = cossim_df['instrumentalness'] / cossim_df['duration_m']
    cossim_df['liveness'] = cossim_df['liveness'] / cossim_df['duration_m']
    cossim_df['valence'] = cossim_df['valence'] / cossim_df['duration_m']
    cossim_df['tempo'] = cossim_df['tempo'] / cossim_df['duration_m']

    compare = cossim_df.drop(columns=['id','duration_ms','tempo', 'duration_m']).iloc[0].values

    cossim_df = cossim_df.merge(df[['id', 'name', 'artist','album_img','preview_url']], how='inner', on='id')
    cossim_df = cossim_df.drop_duplicates(subset=['name']).reset_index(drop=True)


    compare_df = res.copy()
    compare_df_y = compare_df['country']
    compare_df = compare_df.drop(columns=['country','tempo'])
    compare_df['sim'] = compare_df.apply(lambda x: cosine_similarity(compare.reshape(1,-1), x.values.reshape(1,-1))[0][0], axis=1)
    compare_df['id'] = compare_df_y

    compare_df_sort = compare_df.sort_values('sim',ascending=False)[0:5]
    compare_df_sort = compare_df_sort.merge(global_lookup[['country','name','link','playlist_img']], how='inner', left_on='id', right_on='country')
    return compare, cossim_df, compare_df_sort