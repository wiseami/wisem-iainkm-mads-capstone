import requests
import pandas as pd
import datetime
import os
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
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
    """
    Gathers necessary Spotify API information.
    """
    # Spotify token auth url
    AUTH_URL = 'https://accounts.spotify.com/api/token'
    
    # Pull creds from secrets.toml for Streamlit
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

    market = '?market=US'

    # Base URL of all Spotify API endpoints 
    BASE_URL = 'https://api.spotify.com/v1/'

    return headers, market, BASE_URL


#@st.experimental_memo(ttl=86400)
def load_data():
    """Loads datasets for use in other utils and other scripts"""
    now = dt.now()
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    if exists('st_support_files/cache/audio_features_df.csv') and exists('st_support_files/cache/pl_w_audio_feats_df.csv') and (now - dt.fromtimestamp(os.path.getmtime('st_support_files/cache/audio_features_df.csv'))).days < 1:
        audio_features_df = pd.read_csv('st_support_files/cache/audio_features_df.csv')
        pl_w_audio_feats_df = pd.read_csv('st_support_files/cache/pl_w_audio_feats_df.csv')
        playlist_data_df = pd.read_csv('st_support_files/cache/all_pls.csv')
        
    else:
        audio_features_df = pd.read_csv('lookups/all_track_audio_features.csv')
        playlist_data_df = pd.read_csv('playlist_data/playlist_data.csv')
        playlist_data_df = playlist_data_df[['country', 'track_id']].drop_duplicates()
        playlist_data_df.to_csv('st_support_files/cache/all_pls.csv', index=False)

        pl_w_audio_feats_df = playlist_data_df.merge(audio_features_df, how='right', on='track_id').drop_duplicates()
        pl_w_audio_feats_df['pl_count'] = pl_w_audio_feats_df.groupby('track_id')['country'].transform('size')

        audio_feat_cols = ['track_id','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo_1','duration_ms','time_signature','update_dttm','name','artist','album_img','preview_url','popularity','basic_kmeans_cluster', 'pl_count', 'adv_kmeans_cluster', 'chroma', 'chroma_cens', 'mff', 'spectral_centroid', 'spectral_bandwidth', 'spectral_contrast', 'spectral_flatness', 'Spectral_Rolloff', 'poly_features', 'tonnetz', 'ZCR', 'onset_strength', 'pitch', 'magnitude', 'tempo_2']
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
def corr_matrix():
    """ Creates correlation matrix data and cache for Streamlit."""
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
def top3_songs(playlist_data_df):
    """ Finds top 3 most played songs across all playlists.
        Create cache for Streamlit performance.
    """
    if exists('st_support_files/cache/top3_songs.csv') and (now - dt.fromtimestamp(os.path.getmtime('st_support_files/cache/top3_songs.csv'))).days < 1:
        top3_songs = pd.read_csv('st_support_files/cache/top3_songs.csv')
    else:
        top3_songs = pd.DataFrame(playlist_data_df.groupby(['track_id'])['country'].nunique().sort_values(ascending=False).reset_index()).head(3)
        top3_songs = top3_songs.merge(audio_features_df[['track_id','artist','name','album_img','preview_url']], how='inner', on='track_id')
        top3_songs.columns = ['Track ID', '# Playlist Appearances', 'Artist', 'Track Name', 'album_img','preview_url']
        top3_songs.to_csv('st_support_files/cache/top3_songs.csv', index=False)
    
    return top3_songs


def get_audio_features(feat):
    """Takes json result of 'audio-features?ids=' Spotify API call and return df """
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
                                       'tempo_1' : track['tempo'],
                                       'duration_ms' : track['duration_ms'],
                                       'time_signature' : track['time_signature'],
                                       'update_dttm' : now
                                      }
    
    audio_features_df = pd.DataFrame.from_dict(track_list, orient='index')
    audio_features_df.index.name = 'id'
    audio_features_df.reset_index(inplace=True)
    
    return audio_features_df


def get_track_info(feat):
    """Takes json result of 'tracks?ids=' Spotify API call and return df"""
    track_list = dict()
    for track in feat['tracks']:
        if track != None:
            track_list[track['id']] = {'name': track['name'],
                                       'artist': track['artists'][0]['name'],
                                       'album_img': track['album']['images'][0]['url'],
                                       'preview_url': track['preview_url'],
                                       'popularity': track['popularity']
                                      }
    track_info_df = pd.DataFrame.from_dict(track_list, orient='index')
    track_info_df.index.name = 'id'
    track_info_df.reset_index(inplace=True)
    
    return track_info_df


# KMeans functions
def kmeans_k_tuning(X_scaled, k_min, k_max):
    """Calculate inertia and silhouette scores for KMeans clusters
    
        X_scaled: df run through StandardScaler() function
        k_min: minimum KMeans clusters. Cannot be less than 2.
        k_max: maximum KMeans clusters
    """
    inertia, silhouette = [], []
    for k in range(k_min, k_max):
        kmeans = KMeans(n_clusters=k, random_state=99)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        silhouette.append(silhouette_score(X_scaled, kmeans.predict(X_scaled)))
        
    ine = kmeans_k_tuning_plots(k_min=k_min, k_max=k_max, inertia=inertia, silhouette=silhouette)
    
    # storing results in df
    results = pd.DataFrame({'k': list(range(k_min, k_max)), 'inertia': inertia, 'silhouette_score': silhouette})
    return results


def kmeans_k_tuning_plots(k_min, k_max, inertia, silhouette):
    """Generates inertia and silhouette score plots for each KMeans cluster"""
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
    """Used in recommendation engine for songs with no preview url to assign
       a "basic" KMeans cluster

       track_df: result of utils.get_track_info(track_info) merged with 
                 utils.get_audio_features(audio_feats)
    """
    X = track_df.drop(columns=['track_id','duration_ms','update_dttm','time_signature','name','artist','album_img','preview_url', 'popularity'])
    basic_scaler = pickle.load(open("model/basic_scaler.pkl", "rb"))
    data_scaled = basic_scaler.transform(X)
    X_scaled = pd.DataFrame(data_scaled)

    basic_kmeans = pickle.load(open("model/basic_kmeans.pkl", "rb"))
    clusters = basic_kmeans.predict(X_scaled)
    audio_features_df_clustered = track_df.copy()
    audio_features_df_clustered["basic_kmeans_cluster"] = clusters
    return audio_features_df_clustered


### Librosa Music Features Collection
def load_tracks(list_file_name):
    """Loads a track list csv"""
    df = pd.read_csv(list_file_name)
    df = df[df['preview_url'].notna()]
    df.rename(columns={'id': 'track_id'}, inplace=True)
    return df


def extract_mp3(url, save_folder, file_name):
    """Downloads mp3 using preview url from Spotify API output"""
    doc = requests.get(url)
    with open(save_folder+file_name, 'wb') as f:
        f.write(doc.content)


def delete_mp3(file):
    """Deletes mp3 after extracting audio features"""
    os.remove(file)


def get_features(file):
    """Utilizes Librosa to extract audio features from mp3
    
       file: any mp3 file
    """
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
              'tempo_2': [tempo]
              }
    return output


def process_tracks(source_file, end_file, max_rows=0):
    """Checks for any new tracks in track lookup files and attempt to get
       Librosa audio features.

       source_file : path of track list csv
       end_file : output file path
    """
    global new, end_df, df_feat
    features_dict = defaultdict(list)
    source_df = load_tracks(source_file)

    # Create a file if it doesn't already exist
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
                
            else:
                end_df = end_df.append(df_feat, ignore_index=True)
                end_df.to_csv(end_file, index=False)
                
            print('Tracks {} through {} finished.'.format(i, i+batchsize))
    else:
        print('No new tracks to process.')
    print('Feature extraction function complete.')


def dedupe(end_file, column):
    """Removes duplicates based on certain column"""
    df = pd.read_csv(end_file)
    og_len = len(df)
    dupes = og_len - len(df.drop_duplicates(subset=[column], keep='first'))
    df.drop_duplicates(subset=[column], keep='first', inplace=True, ignore_index=True)
    df.to_csv(end_file, index=False)
    print('Dedupe complete. {} duplicates removed.'.format(dupes))


def combine_csv(source_file, end_file, column, new_file):
    """Combines output csvs"""
    df_1 = pd.read_csv(source_file)
    df_2 = pd.read_csv(end_file)
    for d in [df_1, df_2]:
        if 'id' in d.columns.to_list():
            d.rename(columns={'id':'track_id'}, inplace=True)
    df_f = df_1.merge(df_2, how='left', on=[column])
    df_f.to_csv(new_file, index=False)
    print('Files combined.')


def conv_type(end_file):
    """Formatting and fixes datatypes"""
    df = pd.read_csv(end_file)
    col_name = ['chroma', 'chroma_cens', 'mff', 'spectral_centroid',
                'spectral_bandwidth', 'spectral_contrast', 'spectral_flatness',
                'Spectral_Rolloff', 'poly_features', 'tonnetz', 'ZCR', 'onset_strength',
                'pitch', 'magnitude', 'tempo_2']
    for col in col_name:
        df[col] = [i.strip('array()[]') if not type(i) is float else i for i in df[col]]
        df[col] = pd.to_numeric(df[col])
    df.to_csv(end_file, index=False)
    print('Columns converted to float type.')


def get_librosa_features(source_file, end_file, column, final_end_file, max_rows=0):
    """Combines several of the previous utilites to get all Librosa features"""
    process_tracks(source_file, end_file)
    dedupe(end_file, column)
    conv_type(end_file)
    combine_csv(source_file, end_file, column, final_end_file)


def do_kmeans_advanced_on_fly(track_df):
    """Used in recommendation engine for songs with preview url to assign
       a "advanced" KMeans cluster

       track_df: result of utils.get_track_info(track_info) merged with 
                 utils.get_audio_features(audio_feats)
    """
    #track_df = pd.read_csv('audio_files/df.csv')
    if not pd.isna(track_df['preview_url'][0]):
        extract_mp3(track_df['preview_url'][0], 'audio_files/', track_df['track_id'][0]+'.mp3')
        track_dict = get_features('audio_files/'+track_df['track_id'][0]+'.mp3')
        delete_mp3('audio_files/'+track_df['track_id'][0]+'.mp3')
        for k,v in track_dict.items():
            track_dict[k] = v[0]
        track_dict['tempo_2'] = track_dict['tempo_2'][0]
        track_dict['track_id'] = track_df['track_id'][0]
        track_adv_df = pd.DataFrame([track_dict])
        track_df_final = track_df.merge(track_adv_df, on = 'track_id')

        X = track_df_final[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo_1',
                            'chroma', 'chroma_cens', 'mff', 'spectral_centroid',
                            'spectral_bandwidth', 'spectral_contrast', 'spectral_flatness',
                            'Spectral_Rolloff', 'poly_features', 'tonnetz', 'ZCR', 'onset_strength',
                            'pitch', 'magnitude', 'tempo_2']
                          ]
        adv_scaler = pickle.load(open("model/adv_scaler.pkl", "rb"))
        data_scaled = adv_scaler.transform(X)
        X_scaled = pd.DataFrame(data_scaled)

        adv_kmeans = pickle.load(open("model/adv_kmeans.pkl", "rb"))
        clusters = adv_kmeans.predict(X_scaled)
        audio_features_df_clustered = track_df_final.copy()
        audio_features_df_clustered["adv_kmeans_cluster"] = clusters
        return audio_features_df_clustered

    else:
        return None


def Recommendizer(final_df):
    if pd.isna(final_df['preview_url'][0]):
        #BASIC KMEANS
        bas_final_df = do_kmeans_on_fly(final_df)

        pl_feat_merge = playlist_data_df.merge(audio_features_df, how='inner', on='track_id')
        clusters_by_country = pl_feat_merge.groupby(['country', 'basic_kmeans_cluster'])['track_id'].count().sort_values(ascending=False).reset_index()

        # gets songs from top playlist with most number of songs in the same cluster
        tops = clusters_by_country[clusters_by_country['basic_kmeans_cluster']==bas_final_df['basic_kmeans_cluster'].item()].sort_values(by='track_id', ascending=False)[0:1]
        top_pl_track_ids = playlist_data_df[playlist_data_df['country'] == tops['country'].item()]['track_id']

        cossim_df = audio_features_df[audio_features_df['track_id'].isin(top_pl_track_ids)]
        cossim_df = cossim_df[spot_feats]
        cossim_df_y = cossim_df['track_id']
        cossim_df = cossim_df.drop(columns=['track_id'])

        compare_df = bas_final_df.copy()
        compare_df = compare_df[spot_feats]
        compare_df_y = compare_df['track_id']
        compare_df = compare_df.drop(columns=['track_id'])

        cossim_df_f = cossim_df.copy()[compare_df.columns.tolist()]

        basic_scaler = pickle.load(open("model/basic_scaler.pkl", "rb"))
        scaled_cossim = basic_scaler.transform(cossim_df_f)
        scaled_compare = basic_scaler.transform(compare_df)

        cossim_df_f['basic_sim'] = cosine_similarity(scaled_cossim, scaled_compare)

        #cossim_df_f['basic_sim'] = cossim_df_f.apply(lambda x: cosine_similarity(compare_df.values.reshape(1,-1), x.values.reshape(1,-1))[0][0], axis=1)
        cossim_df_f['track_id'] = cossim_df_y
        cossim_df_f = cossim_df_f[cossim_df_f['basic_sim'] < 1]
        cossim_df_sort = cossim_df_f.sort_values('basic_sim',ascending=False)[0:5]
        cossim_df_sort = cossim_df_sort.merge(audio_features_df[['track_id','name','artist','album_img','preview_url']], how='inner', on='track_id')

        compare_df['track_id'] = compare_df_y
        compare_df = compare_df.merge(bas_final_df[['track_id','name','artist','album_img','preview_url']], how='inner', on='track_id')

        final_playlist = global_pl_lookup[global_pl_lookup['country']==tops['country'].item()]
    
    else:
        #ADV KMEANS
        adv_df = do_kmeans_advanced_on_fly(final_df)
        pl_feat_merge = playlist_data_df.merge(audio_features_df, how='inner', on='track_id')
        clusters_by_country = pl_feat_merge.groupby(['country', 'adv_kmeans_cluster'])['track_id'].count().sort_values(ascending=False).reset_index()
        
        tops = clusters_by_country[clusters_by_country['adv_kmeans_cluster']==adv_df['adv_kmeans_cluster'].item()].sort_values(by='track_id', ascending=False)[0:1]
        top_pl_track_ids = playlist_data_df[playlist_data_df['country'] == tops['country'].item()]['track_id']
        
        cossim_df = audio_features_df[audio_features_df['track_id'].isin(top_pl_track_ids)]
        cossim_df = cossim_df[feats_to_keep]
        cossim_df_y = cossim_df['track_id']
        cossim_df = cossim_df.drop(columns=['track_id'])

        compare_df = adv_df.copy()
        compare_df = compare_df[feats_to_keep]
        compare_df_y = compare_df['track_id']
        compare_df = compare_df.drop(columns=['track_id'])

        cossim_df_f = cossim_df.copy()[compare_df.columns.tolist()]

        adv_scaler = pickle.load(open("model/adv_scaler.pkl", "rb"))
        scaled_cossim = adv_scaler.transform(cossim_df_f)
        scaled_compare = adv_scaler.transform(compare_df)

        cossim_df_f['adv_sim'] = cosine_similarity(scaled_cossim, scaled_compare)

        #cossim_df_f['adv_sim'] = cossim_df_f.apply(lambda x: cosine_similarity(compare_df.values.reshape(1,-1), x.values.reshape(1,-1))[0][0], axis=1)
        cossim_df_f['track_id'] = cossim_df_y
        cossim_df_f = cossim_df_f[cossim_df_f['adv_sim'] < 1]
        cossim_df_sort = cossim_df_f.sort_values('adv_sim',ascending=False)[0:5]
        cossim_df_sort = cossim_df_sort.merge(audio_features_df[['track_id','name','artist','album_img','preview_url']], how='inner', on='track_id')

        compare_df['track_id'] = compare_df_y
        compare_df = compare_df.merge(adv_df[['track_id','name','artist','album_img','preview_url']], how='inner', on='track_id')

        final_playlist = global_pl_lookup[global_pl_lookup['country']==tops['country'].item()]

    return compare_df, final_playlist, cossim_df_sort

        



"""Dictionaries used elsewhere"""
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
            "Valence":"A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).",
            "Chroma":"An element of pitch, chroma features correlate closely to harmony and are often used in chord recognition, music alignment/synchronization, and song identification.",
            "Chroma CENS":"A variant of Chroma, Chroma Energy Normalized (CENS) also correlates closely to harmony but incorporates additional normalization steps in its derivation that make it more robust to dynamic, timber, and articulation.",
            "MFF":"Mel-Frequency Cepstral Coefficients are a set of features that describe the overall shape of a song's spectral envelope and are often used to describe timbre.",
            "Spectral Centroid":"Indicates where the center is of a song along a spectrogram and is often associated with the brightness of a song.",
            "Spectral Bandwidth":"The variance from the spectral centroid which has correlation to timbre.",
            "Spectral Contrast":"The difference between peaks and valleys in a song’s spectrogram. ",
            "Spectral Flatness":"A measure of how tone-like or noise-like a song is.",
            "Spectral Rolloff":"The frequency below which 85% of the spectral energy lies.",
            "Poly Features":"The coefficients of fitting an nth-order polynomial to the columns of a spectrogram. In our implementation, a linear polynomial was used.",
            "Tonnetz":"Tonnetz is an n-dimensional mesh that maps the tonal relationships of a song, which is another way of representing harmonic relationships.",
            "Zero Crossing Rate":"The rate at which a signal changes from positive to negative and vice versa, which is representative of percussive sounds.",
            "Onset Strength":"Method of measuring the onset of notes.",
            "Pitch":"Pitch of a song. How the human ear hears and understands the frequency of a sound wave.",
            "Magnitude":"b"
            }

spot_feats = ['track_id','danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence',
                'tempo_1']

lib_feats = ['chroma',
    'chroma_cens', 'mff', 'spectral_bandwidth', 'spectral_contrast',
    'spectral_flatness', 'Spectral_Rolloff', 'poly_features',
    'tonnetz', 'ZCR', 'onset_strength', 'pitch', 'magnitude', 'tempo_2']

feats_to_keep = ['track_id', 'danceability', 'energy', 'key', 'loudness', 'mode','speechiness', 'acousticness', 'instrumentalness', 'liveness','valence', 'tempo_1', 'chroma', 
                        'chroma_cens', 'mff', 'spectral_centroid', 'spectral_bandwidth','spectral_contrast', 'spectral_flatness', 'Spectral_Rolloff','poly_features', 'tonnetz', 
                        'ZCR', 'onset_strength', 'pitch','magnitude', 'tempo_2']

feats_to_show_streamlit = ['artist', 'name','danceability','energy','key','loudness','mode','speechiness','acousticness',
            'instrumentalness','liveness','valence', 'album_img','preview_url']