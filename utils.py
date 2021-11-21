import requests
import pandas as pd
import datetime
import os
from os.path import exists
import time
import json
import itertools

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