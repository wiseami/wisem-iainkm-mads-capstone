import requests
import pandas as pd
import datetime
import os
import json
import utils

# Import Spotify info
headers, market, BASE_URL = utils.spotify_info()

# Read in our csv lookup with all 69 Daily Song Charts
file_path = os.path.dirname(os.path.abspath(__file__)) + '/'
playlist_lookup = pd.read_csv(file_path + 'lookups/global_top_daily_playlists.csv')

# Get the most recent updated time stamp from the first song in the Global playlist
# This tells us if the playlists were updated yet today or not since it doesn't seem to be any sort of set schedule on the Spotify side
playlist_tracks = requests.get(BASE_URL + 'playlists/' + playlist_lookup['id'][0] + '/tracks', headers=headers)
playlist_tracks = playlist_tracks.json()

for tracks in playlist_tracks['items'][:1]:
    new_updated_date = pd.to_datetime(tracks['added_at']).date()

# Get the last updated time stamp from master file
playlist_scrape_lookup = pd.read_csv(file_path + 'playlist_data/playlist_data.csv')
last_updated_date = pd.to_datetime(playlist_scrape_lookup['track_added_date']).dt.date.max()

# Check the new updated time stamp against the max value in our existing playlist data
# If it is, load up the new data into its own file and append to the master file
# If not, it'll check again when running the automated process
if new_updated_date > last_updated_date:
    final_list = []
    update_dttm = datetime.datetime.now()
    for pl_id in playlist_lookup.index:
        playlist_tracks = requests.get(BASE_URL + 'playlists/' + playlist_lookup['id'].iloc[pl_id] + '/tracks', headers=headers)
        playlist_tracks = playlist_tracks.json()
        track_num = 1
        for tracks in playlist_tracks['items']:
            track_list = []
            track_list.append(playlist_lookup['country'].iloc[pl_id])
            track_list.append(playlist_lookup['market'].iloc[pl_id])
            track_list.append(update_dttm)
            track_list.append(tracks['track']['artists'][0]['name'])
            track_list.append(tracks['track']['name'])
            track_list.append(tracks['track']['id'])
            track_list.append(tracks['track']['popularity'])
            track_list.append(track_num)
            track_list.append(tracks['track']['preview_url'])
            track_list.append(tracks['track']['duration_ms'])
            track_list.append(tracks['added_at'])
            final_list.append(track_list)
            track_num += 1

    track_df = pd.DataFrame(final_list, columns=(['country','market','capture_dttm','track_artist','track_name','track_id','track_popularity','track_number','track_preview_url','track_duration','track_added_date']))

    f_name = str(track_df['capture_dttm'][0].date())
    track_df.to_csv(file_path + 'playlist_data/daily_data/' + f_name + '.csv', index=False)
    track_df.to_csv(file_path + 'playlist_data/playlist_data.csv', mode='a',header=False, index=False)    

    print('New playlist data loaded')
    #exit(0)

else:
    print('Playlists not yet updated')
    #exit(1)