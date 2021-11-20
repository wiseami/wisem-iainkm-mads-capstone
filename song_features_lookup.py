import requests
import pandas as pd
import datetime
import os
from os.path import exists
import time
import json
import itertools

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

# Create batches of 99 from unique_tracks to pass to the API for rate limit efficiency
def split(input_list, batch_size):
    for i in range(0, len(unique_tracks), batch_size):
        yield input_list[i:i + batch_size]

batch_size = 49

unique_tracks_for_api = list(split(unique_tracks, batch_size))

update_dttm = datetime.datetime.now()

# Function to pull audio features into dictionary
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
    return track_list

def get_track_info(feat):
    track_list = dict()
    for track in feat['tracks']:
        track_list[track['id']] = {'name': track['name'],
                                   'artist': track['artists'][0]['name'],
                                   'album_img': track['album']['images'][0]['url'],
                                   #'artist_img': track['artists'][0]['images'][0]['url'],
                                   'preview_url': track['preview_url']
                                  }
    return track_list

# Pull audio features using track dict and write/append to file
for track_id_list in unique_tracks_for_api:
    req = requests.get(BASE_URL + 'audio-features?ids=' + (','.join(track_id_list)), headers=headers)
    feat = req.json()
    audio_features_df = pd.DataFrame.from_dict(get_audio_features(feat), orient='index')
    audio_features_df.index.name = 'id'
    audio_features_df.reset_index(inplace=True)

    req = requests.get(BASE_URL + 'tracks?ids=' + (','.join(track_id_list)), headers=headers)
    feat = req.json()
    track_info_df = pd.DataFrame.from_dict(get_track_info(feat), orient='index')
    track_info_df.index.name = 'id'
    track_info_df.reset_index(inplace=True)

    final_df = audio_features_df.merge(track_info_df, how='inner', on='id')

    if exists(file_path + 'lookups\\track_audio_features.csv') is False:
        final_df.to_csv(file_path + 'lookups\\track_audio_features.csv', index=False)

    else:
        existing_audio_features_lookup = pd.read_csv(file_path + 'lookups\\track_audio_features.csv')
        new_recs = final_df[~final_df['id'].isin(existing_audio_features_lookup['id'])]
        new_recs.to_csv(file_path + 'lookups\\track_audio_features.csv', mode='a',header=False, index=False)