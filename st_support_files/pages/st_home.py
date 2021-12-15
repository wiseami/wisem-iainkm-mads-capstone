import pandas as pd
import streamlit as st
import utils

"""This builds out the home page of the Streamlit app"""

def write():
    """Used to write the page in the streamlit_app.py file"""
    
    # Use utils.load_data() to bring in all necessary data
    file_path, audio_features_df, playlist_data_df, global_pl_lookup, pl_w_audio_feats_df, basic_kmeans_inertia, adv_kmeans_inertia = utils.load_data()

    st.title('Music Affinity Across Geographical Boundaries')
    st.markdown('---')

    with st.container():
        st.subheader('Utilizing the Spotify API')
        st.write("""Spotify's API - free to use with a developer account - grants access to an incredible amount of data. We spent roughly 30 days scraping
                    data from their [daily song charts by country](https://open.spotify.com/genre/charts-regional) to find the most popular songs. This
                    gave us a large enough dataset to work with to then pull audio features and then further analyze exactly what type of music is most
                    popular.
                """)

        st.write("If you want more information on how we used this data to analyze music across the world, check out our [Medium](link) post.")
    
    ### Top 3 song based on pl appearance
    with st.container():
        st.write("""Utilizing our final dataset, made up of 68 country-specific and 1 global playlist, we're able to see that there are definitely songs
                    that are popular in multiple countries. In fact, below are a the top 3 songs based on playlist appearance and you can see that these
                    three songs are on over 90% of all the playlists.
                 """)
        st.write("")
        st.subheader('Top 3 songs based on number of playlist appearances')
        top_songs_df = utils.top3_songs(playlist_data_df)

        top_songs = st.columns(3)
        for i in range(0,3):
            top_songs[i].metric(label='Playlist appearances', value=int(top_songs_df['# Playlist Appearances'][i]))
            top_songs[i].markdown('**' + top_songs_df['Artist'][i] + " - " + top_songs_df['Track Name'][i] + '**')
            top_songs[i].image(top_songs_df['album_img'][i])
            if pd.isna(top_songs_df['preview_url'][i]) == False:
                top_songs[i].audio(top_songs_df['preview_url'][i])
            else:
                top_songs[i].error('No audio preview available.')
        st.write("")
        st.write("Below are the audio features computed and captured by Spotify for these three songs.")
        
        spot_feature_names_to_show = ['artist', 'name','danceability','energy','key','loudness','mode','speechiness','acousticness',
                        'instrumentalness','liveness','valence','tempo_1']

        lib_feat_to_show = ['artist', 'name', 'chroma', 'chroma_cens', 'mff', 'spectral_centroid',
                'spectral_bandwidth', 'spectral_contrast', 'spectral_flatness',
                'Spectral_Rolloff', 'poly_features', 'tonnetz', 'ZCR', 'onset_strength',
                'pitch', 'magnitude', 'tempo_2']

        st.dataframe(audio_features_df[0:3].sort_values(by='pl_count', ascending=False)[spot_feature_names_to_show])
        st.write("")
        st.write("""As outlined in our Medium article, we also utilized Librosa to extract additional audio features from any song that has a 30 second
                preview, as shown here.""")
        st.dataframe(audio_features_df[0:3].sort_values(by='pl_count', ascending=False)[lib_feat_to_show])
        st.write("")
        st.write("""In the following pages, we use both sets of audio features to drive some interactive visualizations, and ultimately,
                    a recommendation tool to find new music.
                """)

if __name__ == "__main__":
    write()