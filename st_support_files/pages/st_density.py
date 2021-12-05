import streamlit as st
import altair as alt
import utils

def write():
    """Used to write the page in the streamlit_app.py file"""
    ### Density Plots
    # load necessary data using function
    file_path, audio_features_df, playlist_data_df, global_pl_lookup, pl_w_audio_feats_df, basic_kmeans_inertia, adv_kmeans_inertia = utils.load_data()
    
    with st.container():
        st.experimental_memo(ttl=86400)
        def spotify_dens_plots():            
            spotify_feature_names = ['danceability','energy','key','loudness','mode','speechiness','acousticness',
                            'instrumentalness','liveness','valence','tempo_1', 'country']

            df_feat = pl_w_audio_feats_df[spotify_feature_names]
            
            charts = []
            for feat in spotify_feature_names:

                charts.append(alt.Chart(df_feat).transform_density(
                    density=feat,
                    groupby=['country']
                ).mark_line().encode(
                    alt.X('value:Q',title=feat),
                    alt.Y('density:Q'),
                    alt.Color('country:N',legend=None),
                    tooltip='country'
                ))

            return charts

        st.title('Density Plots')
        st.write("""Knowing we have 69 playlists makes these visuals not-so-easy to consume, but it seemed worth showing the density plots for a couple of audio features across all countries where each line is a country. 
                    Definitions on left directly from Spotify's [API documentation.](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features)""")

        charts = spotify_dens_plots()

        col1, col2 = st.columns([1,2])
        col1.markdown('**Danceability** - Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.')
        col2.altair_chart(charts[0], use_container_width=True)

        col1, col2 = st.columns([1,2])
        col1.markdown('**Energy** - Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.')
        col2.altair_chart(charts[1], use_container_width=True)

        col1, col2 = st.columns([1,2])
        col1.markdown('**Loudness** - The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.')
        col2.altair_chart(charts[3], use_container_width=True)

        col1, col2 = st.columns([1,2])
        col1.markdown('**Valence** - A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).')
        col2.altair_chart(charts[9], use_container_width=True)

    with st.container():
        def librosa_dens_plots():
            librosa_feature_names = ['chroma', 'chroma_cens', 'mff',
                        'spectral_centroid', 'spectral_bandwidth', 'spectral_contrast',
                        'spectral_flatness', 'Spectral_Rolloff', 'poly_features', 'tonnetz',
                        'ZCR', 'onset_strength', 'pitch', 'magnitude', 'tempo_2','country']
            df_feat = pl_w_audio_feats_df[librosa_feature_names]
            
            charts = []
            for feat in librosa_feature_names:

                charts.append(alt.Chart(df_feat).transform_density(
                    density=feat,
                    groupby=['country']
                ).mark_line().encode(
                    alt.X('value:Q',title=feat),
                    alt.Y('density:Q'),
                    alt.Color('country:N',legend=None),
                    tooltip='country'
                ))

            return charts

        charts = librosa_dens_plots()

        col1, col2 = st.columns([1,2])
        col1.markdown('chroma')
        col2.altair_chart(charts[0], use_container_width=True)

        col1, col2 = st.columns([1,2])
        col1.markdown('chroma_cens')
        col2.altair_chart(charts[1], use_container_width=True)

        col1, col2 = st.columns([1,2])
        col1.markdown('mff')
        col2.altair_chart(charts[2], use_container_width=True)

        col1, col2 = st.columns([1,2])
        col1.markdown('spectral_centroid')
        col2.altair_chart(charts[3], use_container_width=True)



    with st.container():
        st.write("Pick any of the markets Spotify generates a playlist for and any audio feature to see market comparisons.")
        
        col1, col2 = st.columns(2)
        country_selector = global_pl_lookup['country'].tolist()
        country_choice = col1.multiselect('Choose a market', country_selector, default='global')
        
        all_feature_names = ['danceability','energy','loudness','speechiness','acousticness',
                             'instrumentalness','liveness','valence','tempo_1', 'chroma', 'chroma_cens', 'mff',
                        'spectral_centroid', 'spectral_bandwidth', 'spectral_contrast',
                        'spectral_flatness', 'Spectral_Rolloff', 'poly_features', 'tonnetz',
                        'ZCR', 'onset_strength', 'pitch', 'magnitude', 'tempo_2','country'] #, 'key', 'mode']
        feature_selector = all_feature_names[:-1]

        feature_choice = col2.selectbox('Choose an audio feature', feature_selector)
        
        if country_choice and feature_choice:
            df_feat = pl_w_audio_feats_df[all_feature_names]
            
            choice_df = df_feat[['country',feature_choice]][df_feat['country'].isin(country_choice)]

            st.altair_chart(alt.Chart(choice_df).transform_density(
                    density=feature_choice,
                    groupby=['country']
                ).mark_line().encode(
                    alt.X('value:Q',title=feature_choice),
                    alt.Y('density:Q'),
                    alt.Color('country:N'),
                    tooltip='country'
                ), use_container_width=True)

        else:
            st.markdown('')
            st.error('Pick at least one country!')
    
    ### Audio features definitions expander
    with st.container():
        with st.expander("Audio feature definitions"):
            for k in utils.audio_feat_dict:
                st.markdown('**' + k +'** - ' + utils.audio_feat_dict[k])
            col1,col2 = st.columns([3,1])
            col2.markdown("Source: [Spotify's API documentation.](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features)")

if __name__ == "__main__":
    write()