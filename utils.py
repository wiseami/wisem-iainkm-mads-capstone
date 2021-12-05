import requests
import pandas as pd
import datetime
import os
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pickle
from datetime import datetime as dt
from os.path import exists
import librosa
from collections import defaultdict

now = datetime.datetime.now()
# update_dttm = datetime.datetime.now()

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

#@st.experimental_memo(ttl=86400)
def load_data():
    now = dt.now()
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    if exists('st_support_files/cache/audio_features_df.csv') and exists('st_support_files/cache/pl_w_audio_feats_df.csv') and (now - dt.fromtimestamp(os.path.getmtime('st_support_files/cache/audio_features_df.csv'))).days < 1:
        audio_features_df = pd.read_csv('st_support_files/cache/audio_features_df.csv')
        pl_w_audio_feats_df = pd.read_csv('st_support_files/cache/pl_w_audio_feats_df.csv')
        playlist_data_df = pd.read_csv('st_support_files/cache/all_pls.csv')
        #playlist_data_df = pd.read_csv('playlist_data/2021-11-30.csv')
        
    else:
        # audio_features_df = pd.read_csv('lookups/track_audio_features.csv')
        audio_features_df = pd.read_csv('lookups/all_track_audio_features.csv')
        playlist_data_df = pd.read_csv('playlist_data/playlist_data.csv')
        
        playlist_data_df = playlist_data_df[['country', 'track_id']].drop_duplicates()
        playlist_data_df.to_csv('st_support_files/cache/all_pls.csv', index=False)

        pl_w_audio_feats_df = playlist_data_df.merge(audio_features_df, how='right', on='track_id').drop_duplicates()
        pl_w_audio_feats_df['pl_count'] = pl_w_audio_feats_df.groupby('track_id')['country'].transform('size')

        audio_feat_cols = ['track_id','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo_1','duration_ms','time_signature','update_dttm','name','artist','album_img','preview_url','popularity','basic_kmeans_cluster', 'pl_count', 'adv_kmeans_cluster', 'chroma', 'chroma_cens', 'mff', 'pectral_centroid', 'spectral_bandwidth', 'spectral_contrast', 'spectral_flatness', 'Spectral_Rolloff', 'poly_features', 'tonnetz', 'ZCR', 'onset_strength', 'pitch', 'magnitude', 'tempo_2']
        audio_features_df = pl_w_audio_feats_df.copy().reset_index(drop=True)
        audio_features_df.drop(audio_features_df.columns.difference(audio_feat_cols), 1, inplace=True)
        audio_features_df.drop_duplicates(subset=['track_id'], inplace=True)
        audio_features_df.reset_index(inplace=True, drop=True)

        pl_w_audio_feats_df = pl_w_audio_feats_df.drop(columns=['track_id', 'time_signature', 'track_id','name','artist','album_img','preview_url','update_dttm'])
        pl_w_audio_feats_df = pl_w_audio_feats_df.dropna(how='any', subset=['country']).reset_index(drop=True)

        audio_features_df.to_csv('st_support_files/cache/audio_features_df.csv', index=False)
        pl_w_audio_feats_df.to_csv('st_support_files/cache/pl_w_audio_feats_df.csv', index=False)
    
    global_pl_lookup = pd.read_csv('lookups/global_top_daily_playlists.csv')
    basic_kmeans_inertia = pd.read_csv('model/basic_kmeans_inertia.csv')
    adv_kmeans_inertia = pd.read_csv('model/adv_kmeans_inertia.csv')

    return file_path, audio_features_df, playlist_data_df, global_pl_lookup, pl_w_audio_feats_df, basic_kmeans_inertia, adv_kmeans_inertia

file_path, audio_features_df, playlist_data_df, global_pl_lookup, pl_w_audio_feats_df, basic_kmeans_inertia, adv_kmeans_inertia = load_data()

#st.experimental_memo(ttl=86400)
def corr_matrix(country=None):
    #file_path, audio_features_df, playlist_data_df, global_pl_lookup, pl_w_audio_feats_df, kmeans_inertia = load_data()
    if exists('st_support_files/cache/audio_feat_corr.csv') and (now - dt.fromtimestamp(os.path.getmtime('st_support_files/cache/audio_feat_corr.csv'))).days < 1:
        audio_feat_corr = pd.read_csv('st_support_files/cache/audio_feat_corr.csv')
        audio_feat_corr_ct1 = pd.read_csv('st_support_files/cache/audio_feat_corr_ct1.csv')
        audio_feat_corr_ct2 = pd.read_csv('st_support_files/cache/audio_feat_corr_ct2.csv')
    else:
        audio_feat_corr = audio_features_df.drop(columns=['time_signature','update_dttm','name','artist','album_img','preview_url', 'duration_ms'])
        audio_feat_corr = audio_feat_corr.corr().stack().reset_index().rename(columns={0: 'correlation', 'level_0': 'variable 1', 'level_1': 'variable 2'})
        audio_feat_corr.to_csv('st_support_files/cache/audio_feat_corr.csv', index=False)
    
        audio_feat_corr_ct1 = audio_feat_corr.copy()[(audio_feat_corr['variable 1']!='pl_count') & (audio_feat_corr['variable 1']!='popularity') & (audio_feat_corr['variable 2']!='pl_count') & (audio_feat_corr['variable 2']!='popularity')]
        audio_feat_corr_ct1['correlation_label'] = audio_feat_corr_ct1['correlation'].map('{:.2f}'.format)
        audio_feat_corr_ct1.to_csv('st_support_files/cache/audio_feat_corr_ct1.csv', index=False)
    
        audio_feat_corr_ct2 = audio_feat_corr.copy()[(audio_feat_corr['variable 1']=='pl_count') | (audio_feat_corr['variable 1']=='popularity')]
        audio_feat_corr_ct2['correlation_label'] = audio_feat_corr_ct2['correlation'].map('{:.2f}'.format)
        audio_feat_corr_ct2.to_csv('st_support_files/cache/audio_feat_corr_ct2.csv', index=False)
    
    return audio_feat_corr, audio_feat_corr_ct1, audio_feat_corr_ct2

#st.experimental_memo(ttl=86400)
def normalize_spotify_audio_feats(df):
    if exists('st_support_files/cache/playlist_audio_feature_rollup.csv') and (now - dt.fromtimestamp(os.path.getmtime('st_support_files/cache/playlist_audio_feature_rollup.csv'))).days < 1:
        playlist_audio_feature_rollup = pd.read_csv('st_support_files/cache/playlist_audio_feature_rollup.csv')
    else:
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
        res['popularity'] = res['popularity_sum'] / res['duration_m']
        res['pl_count'] = res['pl_count_sum'] / res['duration_m']

        playlist_audio_feature_rollup = res.drop(columns=['danceability_sum', 'energy_sum', 'key_sum', 'loudness_sum', 'mode_sum', 'speechiness_sum', 'acousticness_sum', 'instrumentalness_sum', 'liveness_sum', 'valence_sum', 'tempo_sum', 'duration_ms_sum', 'track_count','duration_m', 'cluster_sum','cluster_count', 'popularity_sum','popularity_count'])
        playlist_audio_feature_rollup.to_csv('st_support_files/cache/playlist_audio_feature_rollup.csv', index=False)
    
    return playlist_audio_feature_rollup

#st.experimental_memo(ttl=86400)
def top3_songs(df):
    """ df = playlist_data_df"""
    if exists('st_support_files/cache/top3_songs.csv') and (now - dt.fromtimestamp(os.path.getmtime('st_support_files/cache/top3_songs.csv'))).days < 1:
        top3_songs = pd.read_csv('st_support_files/cache/top3_songs.csv')
    else:
        top3_songs = pd.DataFrame(df.groupby(['track_name', 'track_artist','track_id'])['country'].nunique().sort_values(ascending=False).reset_index()).head(3)
        top3_songs.columns = ['Track Name', 'Artist', 'Track ID', '# Playlist Appearances']
        top3_songs = top3_songs.merge(audio_features_df[['track_id','album_img','preview_url']], how='inner', left_on='Track ID', right_on='track_id')
        top3_songs.to_csv('st_support_files/cache/top3_songs.csv', index=False)
    return top3_songs

def normalize_spotify_audio_feats_2(df):
    ### Create Spotify audio features normalized for playlist length
    df = df.copy()
    df['duration_m'] = df['duration_ms'] / 1000 / 60
    df['danceability'] = df['danceability'] / df['duration_m']
    df['energy'] = df['energy'] / df['duration_m']
    df['key'] = df['key'] / df['duration_m']
    df['loudness'] = df['loudness'] / df['duration_m']
    df['mode'] = df['mode'] / df['duration_m']
    df['speechiness'] = df['speechiness'] / df['duration_m']
    df['acousticness'] = df['acousticness'] / df['duration_m']
    df['instrumentalness'] = df['instrumentalness'] / df['duration_m']
    df['liveness'] = df['liveness'] / df['duration_m']
    df['valence'] = df['valence'] / df['duration_m']
    df['tempo'] = df['tempo'] / df['duration_m']
    df['popularity'] = df['popularity'] / df['duration_m']
    df['pl_count'] = df['pl_count'] / df['duration_m']
    playlist_audio_feature_rollup = df.drop(columns=['duration_ms','duration_m','cluster'])
    return playlist_audio_feature_rollup


# Takes json result of 'audio-features?ids=' Spotify API call and return df
def get_audio_features(feat):
    track_list = dict()
    for track in feat['audio_features']:
        if track != None:
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
                                    'update_dttm' : now
                                    }
    
    audio_features_df = pd.DataFrame.from_dict(track_list, orient='index')
    audio_features_df.index.name = 'id'
    audio_features_df.reset_index(inplace=True)
    return audio_features_df



# Takes json result of 'tracks?ids=' Spotify API call and return df
def get_track_info(feat):
    track_list = dict()
    for track in feat['tracks']:
        if track != None:
            track_list[track['id']] = {'name': track['name'],
                                    'artist': track['artists'][0]['name'],
                                    'album_img': track['album']['images'][0]['url'],
                                    #'artist_img': track['artists'][0]['images'][0]['url'],
                                    'preview_url': track['preview_url'],
                                    'popularity': track['popularity']
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
    X = track_df.drop(columns=['id','duration_ms','update_dttm','time_signature','name','artist','album_img','preview_url', 'popularity'])
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

audio_feat_dict = {
            "Acousticness":"A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.",
            "Danceability":"Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.",
            "Energy":"Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.",
            "Instrumentalness":"Predicts whether a track contains no vocals. ""Ooh"" and ""aah"" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly ""vocal"". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.",
            "Key":"The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on.",
            "Liveness":"Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.",
            "Loudness":"The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.",
            "Mode":"Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.",
            "Speechiness":"Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.",
            "Tempo":"The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.",
            "Valence":"A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)."    
            }

### Librosa Music Features Collection
def load_tracks(list_file_name):
    df = pd.read_csv(list_file_name)
    df = df[df['preview_url'].notna()]
    df.rename(columns={'id': 'track_id'}, inplace=True)
    return df


def extract_mp3(url, save_folder, file_name):
    doc = requests.get(url)
    with open(save_folder+file_name, 'wb') as f:
        f.write(doc.content)


def delete_mp3(file):
    os.remove(file)


# get features from librosa
def get_features(file):
    y, sr = librosa.load(file)
    S = np.abs(librosa.stft(y)) #spectral magnitude
    onset_env = librosa.onset.onset_strength(y)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean()
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr).mean()
    mff = librosa.feature.mfcc(y=y, sr=sr).mean()
    spec_cen = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spec_band = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spec_cont = librosa.feature.spectral_contrast(S=S, sr=sr).mean()
    spec_flat = librosa.feature.spectral_flatness(y=y).mean()
    roll = librosa.feature.spectral_rolloff(y).mean()
    poly = librosa.feature.poly_features(S=S, order=1).mean()
    ton = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()

    onset = onset_env.mean()
    pitch, mag = librosa.piptrack(y=y, sr=sr)
    pitch = pitch.mean()
    mag = mag.mean()
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

    output = {
              'chroma': [chroma],
              'chroma_cens': [chroma_cens],
              'mff': [mff],
              'spectral_centroid': [spec_cen],
              'spectral_bandwidth': [spec_band],
              'spectral_contrast': [spec_cont],
              'spectral_flatness': [spec_flat],
              'Spectral_Rolloff': [roll],
              'poly_features': [poly],
              'tonnetz': [ton],
              'ZCR': [zcr],
              'onset_strength': [onset],
              'pitch': [pitch],
              'magnitude': [mag],
              'tempo': [tempo]
              }
    return output

def process_tracks(source_file, end_file, max_rows=0):
    global new, end_df, df_feat
    features_dict = defaultdict(list)
    source_df = load_tracks(source_file)

    # handle limiting the number of rows to process for testing purposes
    if max_rows > 0:
        source_df = source_df.head(max_rows)
    else:
        source_df = source_df

    # handle creating a file if it doesn't already exist
    if not os.path.isfile(end_file):
        # global new
        print('{} does not exist. Creating new file from scratch.'.format(end_file))
        new = True
        tracks = source_df
    else:
        # global end_df
        end_df = pd.read_csv(end_file)
        tracks = source_df[~source_df['track_id'].isin(end_df['track_id'])]



    if len(tracks) > 0:
        # global df_feat
        total = len(tracks)
        print('{} tracks to process'.format(total))
        batchsize = 50
        for i in range(0, total, batchsize):
            features_dict = defaultdict(list)
            if i+batchsize > total:
                batch = tracks.iloc[i: -1]
            else:
                batch = tracks.iloc[i: i+batchsize]
            for num, tup in enumerate(batch.iterrows()):
                idx, row = tup
                # print('Processing track {} of {}.'.format(num+1, total))
                track_id = row.track_id
                url = row.preview_url
                audio_folder = 'audio_files/'
                file_name = str(track_id)+'.mp3'
                audio_file = audio_folder+file_name

                extract_mp3(url, audio_folder, file_name)

                temp_features = get_features(audio_file)

                features_dict['track_id'].append(track_id)
                for key, value in temp_features.items():
                    features_dict[key].append(value)

                delete_mp3(audio_file)
            df_feat = pd.DataFrame(features_dict)

            if not os.path.isfile(end_file):
                df_feat.to_csv(end_file, index=False)
                # return df_feat
            else:
                end_df = end_df.append(df_feat, ignore_index=True)
                end_df.to_csv(end_file, index=False)
                # return end_df
            print('Tracks {} through {} finished.'.format(i, i+batchsize))
    else:
        print('No new tracks to process.')
    print('Feature extraction function complete.')


def dedupe(end_file, column):
    df = pd.read_csv(end_file)
    og_len = len(df)
    dupes = og_len - len(df.drop_duplicates(subset=[column], keep='first'))
    df.drop_duplicates(subset=[column], keep='first', inplace=True, ignore_index=True)
    df.to_csv(end_file, index=False)
    print('Dedupe complete. {} duplicates removed.'.format(dupes))


def combine_csv(source_file, end_file, column, new_file):
    df_1 = pd.read_csv(source_file)
    df_2 = pd.read_csv(end_file)
    for d in [df_1, df_2]:
        if 'id' in d.columns.to_list():
            d.rename(columns={'id':'track_id'}, inplace=True)
    df_f = df_1.merge(df_2, how='left', on=[column])
    df_f.to_csv(new_file, index=False)
    print('Files combined.')


def conv_type(end_file):
    df = pd.read_csv(end_file)
    col_name = ['chroma', 'chroma_cens', 'mff', 'spectral_centroid',
                'spectral_bandwidth', 'spectral_contrast', 'spectral_flatness',
                'Spectral_Rolloff', 'poly_features', 'tonnetz', 'ZCR', 'onset_strength',
                'pitch', 'magnitude', 'tempo']
    for col in col_name:
        df[col] = [i.strip('array()[]') if not type(i) is float else i for i in df[col]]
        df[col] = pd.to_numeric(df[col])
    df.to_csv(end_file, index=False)
    print('Columns converted to float type.')


def get_librosa_features(source_file, end_file, column, final_end_file, max_rows=0):
    process_tracks(source_file, end_file)
    dedupe(end_file, column)
    conv_type(end_file)
    combine_csv(source_file, end_file, column, final_end_file)