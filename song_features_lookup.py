import requests
import pandas as pd
import datetime
import os
import time
import json

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

audio_features_list = []
for t_id in unique_tracks:
    feat = requests.get(BASE_URL + 'audio-features/' + t_id, headers=headers).json()
    track_list = []
    track_list.append(feat['id'])
    track_list.append(feat['danceability'])
    track_list.append(feat['energy'])
    track_list.append(feat['key'])
    track_list.append(feat['loudness'])
    track_list.append(feat['mode'])
    track_list.append(feat['speechiness'])
    track_list.append(feat['acousticness'])
    track_list.append(feat['instrumentalness'])
    track_list.append(feat['liveness'])
    track_list.append(feat['valence'])
    track_list.append(feat['tempo'])
    track_list.append(feat['duration_ms'])
    track_list.append(feat['time_signature'])
    audio_features_list.append(track_list)

audio_features_df = pd.DataFrame(audio_features_list, columns=(['id', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms','time_signature']))
audio_features_df


    f_name = str(track_df['capture_dttm'][0].date())
    track_df.to_csv(file_path + 'playlist_data\\' + f_name + '.csv', index=False)
    track_df.to_csv(file_path + 'playlist_data\\playlist_data.csv', mode='a',header=False, index=False)  



# if 200 == requests.get(BASE_URL + 'audio-features/' + '0gplL1WMoJ6iYaPgMCL0gX', headers=headers):
#     print('ya')



# req = requests.get(BASE_URL + 'audio-features/' + '0gplL1WMoJ6iYaPgMCL0gX', headers=headers)
# req.status_code