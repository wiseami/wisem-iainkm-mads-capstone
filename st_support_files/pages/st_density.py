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

def write():
    """Used to write the page in the app.py file"""
    ### Density Plots
    @st.experimental_memo(ttl=86400)
    def load_data():
        now = dt.now()
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
    
    
    
    with st.container():
        st.experimental_memo(ttl=86400)
        def dens_plots():
            #if exists('st_support_files/altair/dens_chart1.json') and (now - dt.fromtimestamp(os.path.getmtime('st_support_files/altair/dens_chart1.json'))).days < 1:
                #chart1 = alt.Chart.from_json('st_support_files/altair/dens_chart1.json')
                #chart1 = 'st_support_files/altair/dens_chart1.png'
            
            feature_names = ['danceability','energy','key','loudness','mode','speechiness','acousticness',
                            'instrumentalness','liveness','valence','tempo', 'duration_ms', 'country']

            df_feat = pl_w_audio_feats_df[feature_names]
            
            charts = []
            for feat in feature_names:

                charts.append(alt.Chart(df_feat).transform_density(
                    density=feat,
                    groupby=['country']
                ).mark_line().encode(
                    alt.X('value:Q',title=feat),
                    alt.Y('density:Q'),
                    alt.Color('country:N',legend=None),
                    tooltip='country'
                ))

            chart1 = charts[0]
            chart2 = charts[1] 
            chart3 = charts[3]
            chart4 = charts[9]

            # if 'dens_charts' not in st.session_state:
            #     st.session_state['dens_charts'] = charts
            return chart1, chart2, chart3, chart4
        
        chart1, chart2, chart3, chart4 = dens_plots()

        st.title('Density Plots')
        st.write("""Knowing we have 69 playlists makes these visuals not-so-easy to consume, but it seemed worth showing the density plots for a couple of audio features across all countries where each line is a country. 
                    Definitions on left directly from Spotify's [API documentation.](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features)""")

        col1, col2 = st.columns([1,2])
        col1.markdown('**Danceability** - Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.')
        col2.altair_chart(chart1, use_container_width=True)

        col1, col2 = st.columns([1,2])
        col1.markdown('**Energy** - Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.')
        col2.altair_chart(chart2, use_container_width=True)

        col1, col2 = st.columns([1,2])
        col1.markdown('**Loudness** - The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.')
        col2.altair_chart(chart3, use_container_width=True)

        col1, col2 = st.columns([1,2])
        col1.markdown('**Valence** - A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).')
        col2.altair_chart(chart4, use_container_width=True)

    st.markdown('---')


if __name__ == "__main__":
    write()