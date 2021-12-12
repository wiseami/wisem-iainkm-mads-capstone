import requests
import pandas as pd
from os.path import exists
from collections import Counter
import utils

"""This script finds all the unique genres from our overall track list
   and since Spotify stores them at very granular levels, attempts to
   create higher-level aggregates
"""

if not exists('EDA\genres_list.csv'):
    # Setup for Spotify API
    headers, market, BASE_URL = utils.spotify_info()

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

    dict = {'genres':genres_list}
    pd.DataFrame(dict).to_csv('EDA\genres_list.csv', index=False)

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

else:
    # Load the already generated list
    genres_list = pd.read_csv('EDA\genres_list.csv', index_col=0)
    # Collect genres and find the most common
    print(genres_list.groupby(['genres'])['genres'].count().sort_values(ascending=False))

    # Since Spotify genres are pretty granular, this splits them into each word
    # in an attempt to see them at a slightly higher level
    uniques = []
    for item in genres_list['genres']:
        for thing in item.split():
            uniques.append(thing)

    u = Counter(uniques)
    u.most_common()
