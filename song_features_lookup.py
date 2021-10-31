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
playlist_lookup = pd.read_csv(file_path + 'lookups\\global_top_daily_playlists.csv')

playlist_scrape_lookup = pd.read_csv(file_path + 'playlist_data\\playlist_data.csv')
playlist_lookup = pd.read_csv(file_path + 'lookups\\global_top_daily_playlists.csv')

unique_tracks = playlist_scrape_lookup['track_id'].unique().tolist()

#requests.get(BASE_URL + 'audio-features/4pt5fDVTg5GhEvEtlz9dKk', headers=headers).json()

# Create batches of 99 from unique_tracks to pass to the API for rate limit efficiency
def split(input_list, batch_size):
    for i in range(0, len(unique_tracks), batch_size):
        yield input_list[i:i + batch_size]

batch_size = 99

unique_tracks_for_api = list(split(unique_tracks, batch_size))

unique_tracks_for_api = unique_tracks_for_api[0:2] # get rid of this after testing

# function to pull audio features
def get_audio_features(feat):
    track_list = []
    for track in feat['audio_features']:
        track_l = []
        track_l.append(track['id'])
        track_l.append(track['danceability'])
        track_l.append(track['energy'])
        track_l.append(track['key'])
        track_l.append(track['loudness'])
        track_l.append(track['mode'])
        track_l.append(track['speechiness'])
        track_l.append(track['acousticness'])
        track_l.append(track['instrumentalness'])
        track_l.append(track['liveness'])
        track_l.append(track['valence'])
        track_l.append(track['tempo'])
        track_l.append(track['duration_ms'])
        track_l.append(track['time_signature'])
        track_list.append(track_l)
    return track_list

## Need to create lists of 100 (api max) ids each to pass them in all at once and iterate through
for track_id_list in unique_tracks_for_api:
    req = requests.get(BASE_URL + 'audio-features?ids=' + (','.join(track_id_list)), headers=headers)
    feat = req.json()
    audio_features_list = get_audio_features(feat)
    audio_features_df = pd.DataFrame(audio_features_list, columns=(['id', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms','time_signature']))

if exists(file_path + 'lookups\\track_audio_features.csv') is False:
    audio_features_df.to_csv(file_path + 'lookups\\track_audio_features.csv', index=False)
else:
    existing_audio_features_lookup = pd.read_csv(file_path + 'lookups\\track_audio_features.csv')



#audio_features_df = pd.DataFrame(audio_features_list, columns=(['id', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms','time_signature']))
#audio_features_df


# steps
# write out a file, if it doesn't exist
#     if it does, append
# if id already exists in file, don't write it (maybe update?)




# if exists(file_path + 'lookups\\track_audio_features.csv') is False:
#     audio_features_df.to_csv(file_path + 'lookups\\track_audio_features.csv', index=False)




#     f_name = str(track_df['capture_dttm'][0].date())
#     track_df.to_csv(file_path + 'playlist_data\\' + f_name + '.csv', index=False)
#     track_df.to_csv(file_path + 'playlist_data\\playlist_data.csv', mode='a',header=False, index=False)  



# if 200 == requests.get(BASE_URL + 'audio-features/' + '0gplL1WMoJ6iYaPgMCL0gX', headers=headers):
#     print('ya')



# req = requests.get(BASE_URL + 'audio-features/' + '0gplL1WMoJ6iYaPgMCL0gX', headers=headers)
# req.status_code