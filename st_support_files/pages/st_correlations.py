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

    ### Correlations
    with st.container():
        st.header('Correlations')
        st.write("Some text here about correlations and audio features...")

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
        
        ### Correlation matrix
        st.experimental_memo(ttl=86400)
        def corr_matrix():
            if exists('st_support_files/audio_feat_corr.csv') and (now - dt.fromtimestamp(os.path.getmtime('st_support_files/audio_feat_corr.csv'))).days < 1:
                audio_feat_corr = pd.read_csv('st_support_files/audio_feat_corr.csv')
                audio_feat_corr_ct1 = pd.read_csv('st_support_files/audio_feat_corr_ct1.csv')
                audio_feat_corr_ct2 = pd.read_csv('st_support_files/audio_feat_corr_ct1.csv')
            else:
                audio_feat_corr = audio_features_df.drop(columns=['time_signature','update_dttm','name','artist','album_img','preview_url', 'duration_ms'])
                audio_feat_corr = audio_feat_corr.corr().stack().reset_index().rename(columns={0: 'correlation', 'level_0': 'variable 1', 'level_1': 'variable 2'})
                audio_feat_corr.to_csv('st_support_files/audio_feat_corr.csv', index=False)
            
                audio_feat_corr_ct1 = audio_feat_corr.copy()[(audio_feat_corr['variable 1']!='pl_count') & (audio_feat_corr['variable 1']!='popularity') & (audio_feat_corr['variable 2']!='pl_count') & (audio_feat_corr['variable 2']!='popularity')]
                audio_feat_corr_ct1['correlation_label'] = audio_feat_corr_ct1['correlation'].map('{:.2f}'.format)
                audio_feat_corr_ct1.to_csv('st_support_files/audio_feat_corr_ct1.csv', index=False)
            
                audio_feat_corr_ct2 = audio_feat_corr.copy()[(audio_feat_corr['variable 1']=='pl_count') | (audio_feat_corr['variable 1']=='popularity')]
                audio_feat_corr_ct2['correlation_label'] = audio_feat_corr_ct2['correlation'].map('{:.2f}'.format)
                audio_feat_corr_ct2.to_csv('st_support_files/audio_feat_corr_ct2.csv', index=False)
            
            return audio_feat_corr, audio_feat_corr_ct1, audio_feat_corr_ct2
        
        st.title("Correlation Matrix")
        #audio_feat_corr = corr_matrix()
        #audio_feat_corr_ct1 = audio_feat_corr.copy()[(audio_feat_corr['variable 1']!='pl_count') & (audio_feat_corr['variable 1']!='popularity') & (audio_feat_corr['variable 2']!='pl_count') & (audio_feat_corr['variable 2']!='popularity')]
        #audio_feat_corr_ct1['correlation_label'] = audio_feat_corr_ct1['correlation'].map('{:.2f}'.format)

        audio_feat_corr, audio_feat_corr_ct1, audio_feat_corr_ct2 = corr_matrix()

        base = alt.Chart(audio_feat_corr_ct1).encode(
            x='variable 2:O',
            y='variable 1:O'    
        )

        # Text layer with correlation labels
        # Colors are for easier readability
        text = base.mark_text().encode(
            text=alt.condition(
                alt.datum.correlation == 1,
                alt.value(''),
                'correlation_label'
            ),
            color=alt.condition(
                alt.datum.correlation > 0.5, 
                alt.value('white'),
                alt.value('black')
            )
        )

        # The correlation heatmap itself
        cor_plot = base.mark_rect().encode(
            color=alt.Color('correlation:Q', scale=alt.Scale(scheme='greenblue'))
        )

        col1, col3, col2 = st.columns([3,1,7])
        col1.write("Now, let's take country out of the equation and have a closer look at the different individual audio features across all distinct tracks.")
        col2.altair_chart(cor_plot + text, use_container_width=True)

        base = alt.Chart(audio_feat_corr_ct2).encode(
            x='variable 2:O',
            y='variable 1:O'    
        )

        # Text layer with correlation labels
        # Colors are for easier readability
        text = base.mark_text().encode(
            text=alt.condition(
                alt.datum.correlation == 1,
                alt.value(''),
                'correlation_label'
            ),
            color=alt.condition(
                alt.datum.correlation > 0.5, 
                alt.value('white'),
                alt.value('black')
            )
        )

        # The correlation heatmap itself
        cor_plot = base.mark_rect().encode(
            color=alt.Color('correlation:Q', scale=alt.Scale(scheme='orangered'))
        )

    ### Popularity features correlation
    st.subheader("Popularity Correlation Matrix")
    with st.container():
        col1, col3, col2 = st.columns([3,1,7])
        col1.write("But how do these audio features correlate with popularity features?")
        col2.altair_chart(cor_plot + text, use_container_width=True)
        st.write("Interesting to see that neither popularity (a Spotify measure) or playlist count (number of distinct market playlists a song shows up on) correlate very highly with any specific audio feature.")
        st.text("\n")

    ### Correlation matrix market selector
    st.subheader("Market-specific Correlations")
    with st.container():
        col1, col3, col2 = st.columns([3,1,7])
        country_selector = global_pl_lookup['country'].tolist()
        col1.write("Pick any of the markets Spotify generates a playlist for to see how the different features correlate to one another just within that market.")
        choice = col1.selectbox('Choose a market', country_selector)
        choice_df = pl_w_audio_feats_df[pl_w_audio_feats_df['country'] == choice]
        #choice_df = utils.normalize_spotify_audio_feats_2(choice_df)
        audio_feat_corr = choice_df.loc[:, ~choice_df.columns.isin(['popularity','pl_count'])].corr().stack().reset_index().rename(columns={0: 'correlation', 'level_0': 'variable 1', 'level_1': 'variable 2'})
        audio_feat_corr['correlation_label'] = audio_feat_corr['correlation'].map('{:.2f}'.format)

        base = alt.Chart(audio_feat_corr).encode(
            x='variable 2:O',
            y='variable 1:O'    
        )

        # Text layer with correlation labels
        # Colors are for easier readability
        text = base.mark_text().encode(
            text=alt.condition(
                alt.datum.correlation == 1,
                alt.value(''),
                'correlation_label'
            ),
            color=alt.condition(
                alt.datum.correlation > 0.5, 
                alt.value('white'),
                alt.value('black')
            )
        )

        # The correlation heatmap itself
        cor_plot = base.mark_rect().encode(
            color=alt.Color('correlation:Q', scale=alt.Scale(scheme='greenblue'))
        )

        col2.altair_chart(cor_plot + text, use_container_width=True)

    ### Audio features definitions expander
    with st.container():
        audio_feat_dict = {
            "Acousticness":"A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.",
            "Danceability":"Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.",
            "Energy":"Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.",
            "Instrumentalness":"Predicts whether a track contains no vocals. ""Ooh"" and ""aah"" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly ""vocal"". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.",
            "Key":"The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on.",
            "Liveness":"Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.",
            "Loudness":"The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.",
            "Mode":"Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.",
            "Speechiness":"Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.",
            "Tempo":"The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.",
            "Valence":"A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)."    
            }

        with st.expander("Audio feature definitions"):
            for k in audio_feat_dict:
                st.markdown('**' + k +'** - ' + audio_feat_dict[k])
            col1,col2 = st.columns([3,1])
            col2.markdown("Source: [Spotify's API documentation.](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features)")

        st.write("It looks like there are a handful of audio features that have high correlations with others.")

    st.markdown('---')

if __name__ == "__main__":
    write()