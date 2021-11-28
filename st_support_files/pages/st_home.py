import pandas as pd
import streamlit as st
import requests
import tqdm
import altair as alt
#from altair.utils.schemapi import SchemaValidationError
import numpy as np
import utils
import pickle
import sys
import os
from datetime import datetime as dt
from os.path import exists

### Start building out Streamlit assets
# st.set_page_config(
#     layout="wide",
#     menu_items = {'About':"Capstone project for University of Michigan's Master of Applied Data Science program by Mike Wise and Iain King-Moore"}
#     )

now = dt.now()
def write():
    """Used to write the page in the app.py file"""
    
### App config stuff - Loading data, creating data caches, etc.
    with st.container():
        # Spotify info
        headers, market, SPOTIFY_BASE_URL = utils.spotify_info()

        # Function to load and cache data for streamlit performance
        @st.experimental_memo(ttl=86400)
        def load_data():
            file_path = os.path.dirname(os.path.abspath(__file__)) + '/'
            if exists('st_support_files/audio_features_df.csv') and exists('st_support_files/pl_w_audio_feats_df.csv') and (now - dt.fromtimestamp(os.path.getmtime('st_support_files/audio_features_df.csv'))).days < 1:
                audio_features_df = pd.read_csv('st_support_files/audio_features_df.csv')
                pl_w_audio_feats_df = pd.read_csv('st_support_files/pl_w_audio_feats_df.csv')
                playlist_data_df = pd.read_csv('playlist_data/2021-11-19.csv')
                
            else:
                audio_features_df = pd.read_csv('lookups/track_audio_features.csv')
                playlist_data_df = pd.read_csv('playlist_data/2021-11-19.csv')

                pl_w_audio_feats_df = playlist_data_df.merge(audio_features_df, how='right', left_on='track_id', right_on='id')
                pl_w_audio_feats_df['pl_count'] = pl_w_audio_feats_df.groupby('track_id')['country'].transform('size')

                audio_feat_cols = ['id','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature','update_dttm','name','artist','album_img','preview_url','popularity','cluster', 'pl_count']
                audio_features_df = pl_w_audio_feats_df.copy().reset_index(drop=True)
                audio_features_df.drop(audio_features_df.columns.difference(audio_feat_cols), 1, inplace=True)
                audio_features_df.drop_duplicates(subset=['id'], inplace=True)
                audio_features_df.reset_index(inplace=True, drop=True)

                pl_w_audio_feats_df = pl_w_audio_feats_df.drop(columns=['market','capture_dttm','track_preview_url','track_duration', 'id', 'track_added_date', 'track_popularity', 'track_number','time_signature', 'track_artist','track_name','track_id','name','artist','album_img','preview_url','update_dttm'])
                pl_w_audio_feats_df = pl_w_audio_feats_df.dropna(how='any', subset=['country']).reset_index(drop=True)

                audio_features_df.to_csv('st_support_files/audio_features_df.csv', index=False)
                pl_w_audio_feats_df.to_csv('st_support_files/pl_w_audio_feats_df.csv', index=False)
            
            global_pl_lookup = pd.read_csv('lookups/global_top_daily_playlists.csv')
            kmeans_inertia = pd.read_csv('model/kmeans_inertia.csv')

            return file_path, audio_features_df, playlist_data_df, global_pl_lookup, pl_w_audio_feats_df, kmeans_inertia

        # load necessary data using function
        file_path, audio_features_df, playlist_data_df, global_pl_lookup, pl_w_audio_feats_df, kmeans_inertia = load_data()

        # Normalize spotify audio features and create playlist rollups
        st.experimental_memo(ttl=86400)
        def normalize_spotify_audio_feats(df):
            if exists('st_support_files/playlist_audio_feature_rollup.csv') and (now - dt.fromtimestamp(os.path.getmtime('st_support_files/playlist_audio_feature_rollup.csv'))).days < 1:
                playlist_audio_feature_rollup = pd.read_csv('st_support_files/playlist_audio_feature_rollup.csv')
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
                playlist_audio_feature_rollup.to_csv('st_support_files/playlist_audio_feature_rollup.csv', index=False)
            
            return playlist_audio_feature_rollup
        
        playlist_audio_feature_rollup = normalize_spotify_audio_feats(pl_w_audio_feats_df)

    ### Sidebar config stuff
    # with st.container():
    #     st.sidebar.write("Testing Sidebar")
    #     show_source_code = st.sidebar.checkbox("Show Source Code", True)
    #     #st.sidebar.button()

    ### Beginning
    st.title('Spotify Streamlit')
    st.markdown('---')

    ### Top 3 song based on pl appearance
    with st.container():
        st.subheader('Top 3 Songs Based on number of playlist appearances')
        st.write("While the first day of scraping playlists came back with 3,450 total songs, only about half of those were unique. Because of that, we have tons of tracks that show up on multiple playlists. We're looking at a total of 69 daily playlists - 68 country-specific and 1 global - and these songs below show up on multiple different country playlists.")

        # st.markdown('---')
        st.experimental_memo(ttl=86400)
        def top3_songs(df):
            """ df = playlist_data_df"""
            if exists('st_support_files/top3_songs.csv') and (now - dt.fromtimestamp(os.path.getmtime('st_support_files/top3_songs.csv'))).days < 1:
                top3_songs = pd.read_csv('st_support_files/top3_songs.csv')
            else:
                top3_songs = pd.DataFrame(df.groupby(['track_name', 'track_artist','track_id'])['country'].count().sort_values(ascending=False).reset_index()).head(3)
                top3_songs.columns = ['Track Name', 'Artist', 'Track ID', '# Playlist Appearances']
                top3_songs = top3_songs.merge(audio_features_df[['id','album_img','preview_url']], how='inner', left_on='Track ID', right_on='id')
                top3_songs.to_csv('st_support_files/top3_songs.csv', index=False)
            return top3_songs

        top_songs_df = top3_songs(playlist_data_df)

        top_songs = st.columns(3)
        for i in range(0,3):
            top_songs[i].metric(label='Playlist appearances', value=int(top_songs_df['# Playlist Appearances'][i]))
            top_songs[i].markdown('**' + top_songs_df['Artist'][i] + " - " + top_songs_df['Track Name'][i] + '**')
            top_songs[i].image(top_songs_df['album_img'][i])
            if pd.isna(top_songs_df['preview_url'][i]) == False:
                top_songs[i].audio(top_songs_df['preview_url'][i])


        st.write("Let's take a look at the audio features computed and captured by Spotify for these three songs.")
        feature_names_to_show = ['artist', 'name','danceability','energy','key','loudness','mode','speechiness','acousticness',
                        'instrumentalness','liveness','valence','tempo']
        st.table(audio_features_df[0:3][feature_names_to_show])

        st.write('testing different ways to show code if we want')
        # if show_source_code:
        #     st.subheader("Source code")
        #     st.code("""top_songs = st.columns(3)
        # for i in range(0,3):
        #     top_songs[i].metric(label='Playlist appearances', value=int(df['# Playlist Appearances'][i]))
        #     top_songs[i].markdown('**' + df['Artist'][i] + " - " + df['Track Name'][i] + '**')
        #     top_songs[i].image(df['album_img'][i])
        #     if pd.isna(df['preview_url'][i]) == False:
        #         top_songs[i].audio(df['preview_url'][i])
        #     """)

        with st.expander("Source Code"):
            st.code("""top_songs = st.columns(3)
        for i in range(0,3):
            top_songs[i].metric(label='Playlist appearances', value=int(df['# Playlist Appearances'][i]))
            top_songs[i].markdown('**' + df['Artist'][i] + " - " + df['Track Name'][i] + '**')
            top_songs[i].image(df['album_img'][i])
            if pd.isna(df['preview_url'][i]) == False:
                top_songs[i].audio(df['preview_url'][i])
            """)

    st.markdown('---')

if __name__ == "__main__":
    write()