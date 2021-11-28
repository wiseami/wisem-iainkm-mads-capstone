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
import st_support_files.pages.st_home as st_home
import st_support_files.pages.st_density as st_density

### Start building out Streamlit assets
# st.set_page_config(
#     layout="wide",
#     menu_items = {'About':"Capstone project for University of Michigan's Master of Applied Data Science program by Mike Wise and Iain King-Moore"}
#     )

now = dt.now()
def write():
    """Used to write the page in the app.py file"""

    @st.experimental_memo(ttl=86400)
    def load_data():
        file_path = os.path.dirname(os.path.abspath(__file__)) + '/'
        if exists('st_support_files/audio_features_df.csv') and exists('st_support_files/pl_w_audio_feats_df.csv'):
            if (now - dt.fromtimestamp(os.path.getmtime('st_support_files/audio_features_df.csv'))).days < 1:
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

        ### KMeans
    with st.container():
        st.title('KMeans')
        st.write('checking inertia and silhouette scores to find best k')
        alt_chart1, alt_chart2 = st.columns(2)
        alt_intertia = alt.Chart(kmeans_inertia[['k','inertia']]).mark_line().encode(
            x='k:O',
            y=alt.Y('inertia', scale=alt.Scale(domain=[12000,24000]))
        )
        alt_chart1.altair_chart(alt_intertia)

        alt_silhouette = alt.Chart(kmeans_inertia[['k','silhouette_score']]).mark_line().encode(
            x='k:O',
            y=alt.Y('silhouette_score', scale=alt.Scale(domain=[.1,.2]))
        )
        alt_chart2.altair_chart(alt_silhouette)

    st.markdown('---')

if __name__ == "__main__":
    write()