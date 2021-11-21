import requests
import pandas as pd
import datetime
import os
from os.path import exists
import time
import json
import itertools
from sklearn.metrics.pairwise import cosine_similarity

update_dttm = datetime.datetime.now()


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





# takes in dfs to do cossim on the fly
def create_cossim_df(df, res, global_lookup):
    """
    df = final_df
    res = df of normalized audio features by country/playlist
    global_lookup = df of global lookup csv
    """
    cossim_df = df.drop(columns=['update_dttm', 'time_signature', 'name','artist','album_img','preview_url'])
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