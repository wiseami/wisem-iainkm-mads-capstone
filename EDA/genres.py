import requests
import pandas as pd
import json

# Setup for Spotify API
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

# base URL of all Spotify API endpoints
BASE_URL = 'https://api.spotify.com/v1/'

tracks = pd.read_csv('./lookups/all_track_audio_features.csv')
tracks = tracks['track_id'].to_list()

# Break list of tracks into batches for Spotify API
def split(input_list, batch_size):
    for i in range(0, len(tracks), batch_size):
        yield input_list[i:i + batch_size]

batch_size = 49
unique_tracks_for_api = list(split(tracks, batch_size))

genres_list = []

# Pull audio features using track dict and write/append to file
for track_id_list in unique_tracks_for_api:
    req = requests.get(BASE_URL + 'tracks?ids=' + (','.join(track_id_list)), headers=headers).json()
    for track in req['tracks']:
        artist = track['artists'][0]['id']
        genres = requests.get(BASE_URL + 'artists/' + artist, headers=headers).json()
        if 'genres' in genres:
            genres = genres['genres']
            for genre in genres:
                genres_list.append(genre)

# Collect genres and find the most common
from collections import Counter
c = Counter(genres_list) 
c.most_common()

# Since Spotify genres are pretty granular, this splits them into each word
# in an attempt to see them at a slightly higher level
uniques = []
for item in genres_list:
    for thing in item.split():
        uniques.append(thing)

u = Counter(uniques)
u.most_common()