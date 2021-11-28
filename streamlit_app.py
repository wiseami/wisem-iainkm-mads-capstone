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
st.set_page_config(
    layout="wide",
    menu_items = {'About':"Capstone project for University of Michigan's Master of Applied Data Science program by Mike Wise and Iain King-Moore"}
    )

now = dt.now()

### App config stuff - Loading data, creating data caches, etc.
with st.container():
    # Spotify info
    headers, market, SPOTIFY_BASE_URL = utils.spotify_info()

    # Function to load and cache data for streamlit performance
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

                audio_features_df.to_csv('st_support_files/audio_features_df.csv')
                pl_w_audio_feats_df.to_csv('st_support_files/pl_w_audio_feats_df.csv')
        
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
            playlist_audio_feature_rollup.to_csv('st_support_files/playlist_audio_feature_rollup.csv')
        
        return playlist_audio_feature_rollup
    
    playlist_audio_feature_rollup = normalize_spotify_audio_feats(pl_w_audio_feats_df)

### Sidebar config stuff
with st.container():
    st.sidebar.write("Testing Sidebar")
    show_source_code = st.sidebar.checkbox("Show Source Code", True)
    #st.sidebar.button()

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
            top3_songs.to_csv('st_support_files/top3_songs.csv')
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
    if show_source_code:
        st.subheader("Source code")
        st.code("""top_songs = st.columns(3)
    for i in range(0,3):
        top_songs[i].metric(label='Playlist appearances', value=int(df['# Playlist Appearances'][i]))
        top_songs[i].markdown('**' + df['Artist'][i] + " - " + df['Track Name'][i] + '**')
        top_songs[i].image(df['album_img'][i])
        if pd.isna(df['preview_url'][i]) == False:
            top_songs[i].audio(df['preview_url'][i])
        """)

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

### Density Plots
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

    st.header('Density Plots')
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

### Correlations
with st.container():
    st.header('Correlations')
    st.write("Some text here about correlations and audio features...")

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
            audio_feat_corr.to_csv('st_support_files/audio_feat_corr.csv')
        
            audio_feat_corr_ct1 = audio_feat_corr.copy()[(audio_feat_corr['variable 1']!='pl_count') & (audio_feat_corr['variable 1']!='popularity') & (audio_feat_corr['variable 2']!='pl_count') & (audio_feat_corr['variable 2']!='popularity')]
            audio_feat_corr_ct1['correlation_label'] = audio_feat_corr_ct1['correlation'].map('{:.2f}'.format)
            audio_feat_corr_ct1.to_csv('st_support_files/audio_feat_corr_ct1.csv')
        
            audio_feat_corr_ct2 = audio_feat_corr.copy()[(audio_feat_corr['variable 1']=='pl_count') | (audio_feat_corr['variable 1']=='popularity')]
            audio_feat_corr_ct2['correlation_label'] = audio_feat_corr_ct2['correlation'].map('{:.2f}'.format)
            audio_feat_corr_ct2.to_csv('st_support_files/audio_feat_corr_ct2.csv')
        
        return audio_feat_corr, audio_feat_corr_ct1, audio_feat_corr_ct2
    
    st.subheader("Correlation Matrix")
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
    col2.altair_chart(st.session_state.corr_plot + text, use_container_width=True)

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

### KMeans
with st.container():
    st.header('KMeans')
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

### Recommendations
with st.container():
    st.subheader('Recommendations')
    st.write('Search for artist to get top 5 songs. Clicking on a song checks our lookups first and if the song isnt there itll run a lookup against spotify API, bring audio features back.') 
    #### testing search bar idea
    search_term = st.text_input('Search an artist', 'Adele')

    search_term = 'Adele' #only here for testing
    search = requests.get(SPOTIFY_BASE_URL + 'search?q=artist:' + search_term + '&type=artist', headers=headers)
    search = search.json()

    feats_to_show_streamlit = ['artist', 'name','danceability','energy','key','loudness','mode','speechiness','acousticness',
                    'instrumentalness','liveness','valence', 'album_img','preview_url']
        
    for item in search['artists']['items'][0:1]:
        searchy = requests.get(SPOTIFY_BASE_URL + 'artists/' + item['id'] + '/top-tracks?market=US', headers=headers).json()
        st.write('Pick one of these top 5 songs for this artist.')
        for top_tracks in searchy['tracks'][0:5]:
            if st.button(top_tracks['name']):
                if audio_features_df['id'].str.contains(top_tracks['id']).any():
                    final_df = audio_features_df[audio_features_df['id']==top_tracks['id']]
                    st.write('on file')
                    st.dataframe(final_df[feats_to_show_streamlit])
                else:
                    audio_feats = requests.get(SPOTIFY_BASE_URL + 'audio-features?ids=' + top_tracks['id'], headers=headers).json()
                    audio_features_df_new = utils.get_audio_features(audio_feats)
                    
                    track_info = requests.get(SPOTIFY_BASE_URL + 'tracks?ids=' + top_tracks['id'], headers=headers).json()
                    track_info_df = utils.get_track_info(track_info)

                    final_df = audio_features_df_new.merge(track_info_df, how='inner', on='id')
                    final_df = utils.do_kmeans_on_fly(final_df)
                    st.write('grabbed from API')
                    st.dataframe(final_df[feats_to_show_streamlit])

    try:
        st.markdown('---')
        
        # can probably do this ealier. Maybe even in utils and just load this in like the rest?
        pl_feat_merge = playlist_data_df.merge(audio_features_df, how='inner', left_on='track_id', right_on='id')
        clusters_by_country = pl_feat_merge.groupby(['country', 'cluster'])['id'].count().sort_values(ascending=False).reset_index()

        # gets songs from top playlist with most number of songs in the same cluster
        tops = clusters_by_country[clusters_by_country['cluster']==final_df['cluster'].item()].sort_values(by='id', ascending=False)[0:1]
        top_pl_track_ids = playlist_data_df[playlist_data_df['country'] == tops['country'].item()]['track_id']
        cossim_df = audio_features_df[audio_features_df['id'].isin(top_pl_track_ids)]
        cossim_df = cossim_df.drop(columns=['update_dttm', 'time_signature', 'name','artist','album_img','preview_url', 'cluster', 'popularity'])
        cossim_df_y = cossim_df['id']
        cossim_df = cossim_df.drop(columns=['id','tempo','duration_ms'])

        compare_df = final_df.copy()
        compare_df = compare_df.drop(columns=['update_dttm', 'time_signature', 'name','artist','album_img','preview_url', 'cluster', 'popularity'])
        compare_df_y = compare_df['id']
        compare_df = compare_df.drop(columns=['id','tempo','duration_ms'])

        from sklearn.metrics.pairwise import cosine_similarity

        cossim_df['sim'] = cossim_df.apply(lambda x: cosine_similarity(compare_df.values.reshape(1,-1), x.values.reshape(1,-1))[0][0], axis=1)
        cossim_df['id'] = cossim_df_y
        cossim_df = cossim_df[cossim_df['sim'] < 1]
        cossim_df_sort = cossim_df.sort_values('sim',ascending=False)[0:5]
        cossim_df_sort = cossim_df_sort.merge(audio_features_df[['id','name','artist','album_img','preview_url']], how='inner', on='id')

        compare_df['id'] = compare_df_y
        compare_df = compare_df.merge(final_df[['id','name','artist','album_img','preview_url']], how='inner', on='id')

        col1, col2 = st.columns([1,1])
        col1.markdown("If you like: *" + compare_df['name'].iloc[0] + "* by " + compare_df['artist'].iloc[0])
        col1.image(compare_df['album_img'].iloc[0])
        col1.audio(compare_df['preview_url'].iloc[0])
        #col1.dataframe(compare_df[feats_to_show_streamlit])



        final_playlist = global_pl_lookup[global_pl_lookup['country']==tops['country'].item()]
        col2.markdown("You might like: ["+ final_playlist['name'].item() +"](" + final_playlist['link'].item() + ")")
        col2.image(final_playlist['playlist_img'].item())
        #col2.markdown("[![this is an image link]("+final_playlist['playlist_img'].item()+")]("+final_playlist['link'].item()+")")
        col2.dataframe(cossim_df_sort[feats_to_show_streamlit])

        for x in cossim_df_sort[feats_to_show_streamlit].index:
            col2.write(cossim_df_sort[feats_to_show_streamlit]['artist'].iloc[x] + " - " + cossim_df_sort[feats_to_show_streamlit]['name'].iloc[x])
            if pd.isna(cossim_df_sort[feats_to_show_streamlit]['preview_url'].iloc[x]) == False:
                col2.audio(cossim_df_sort[feats_to_show_streamlit]['preview_url'].iloc[x])
    except:
        pass

st.markdown('---')

#st.subheader('Wait, how did you do that?')