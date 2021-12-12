import pandas as pd
import streamlit as st
import utils

"""This builds out the home/intro page of the Streamlit app"""

def write():
    """Used to write the page in the streamlit_app.py file"""
    
    # Use utils.load_data() to bring in all necessary data
    file_path, audio_features_df, playlist_data_df, global_pl_lookup, pl_w_audio_feats_df, basic_kmeans_inertia, adv_kmeans_inertia = utils.load_data()

    st.title('Music Affinity Across Geographical Boundaries')
    st.markdown('---')

    ### Top 3 song based on pl appearance
    with st.container():
        st.subheader('Top 3 Songs Based on number of playlist appearances')
        st.write("""While the first day of scraping playlists came back with 3,450 total songs, only about half of those were unique. Because of that, 
        we have tons of tracks that show up on multiple playlists. We're looking at a total of 69 daily playlists - 68 country-specific and 1 global - 
        and these songs below show up on multiple different country playlists.""")

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

        st.write("Let's take a look at the audio features computed and captured by Spotify for these three songs.")
        
        feature_names_to_show = ['artist', 'name','danceability','energy','key','loudness','mode','speechiness','acousticness',
                        'instrumentalness','liveness','valence','tempo_1']
        
        st.table(audio_features_df[0:3][feature_names_to_show])
        with st.expander("Source Code"):
            st.code("""
with st.container():
    st.subheader('Top 3 Songs Based on number of playlist appearances')
    st.write("While the first day of scraping playlists came back with 3,450 total songs, only about half of those were unique. Because of that, 
    we have tons of tracks that show up on multiple playlists. We're looking at a total of 69 daily playlists - 68 country-specific and 1 global - 
    and these songs below show up on multiple different country playlists.")

    top_songs_df = utils.top3_songs(playlist_data_df)

    top_songs = st.columns(3)
    for i in range(0,3):
        top_songs[i].metric(label='Playlist appearances', value=int(top_songs_df['# Playlist Appearances'][i]))
        top_songs[i].markdown('**' + top_songs_df['Artist'][i] + " - " + top_songs_df['Track Name'][i] + '**')
        top_songs[i].image(top_songs_df['album_img'][i])
        if pd.isna(top_songs_df['preview_url'][i]) == False:
            top_songs[i].audio(top_songs_df['preview_url'][i])

    st.write("Let's take a look at the audio features computed and captured by Spotify for these three songs.")
    
    feature_names_to_show = ['artist', 'name','danceability','energy','key','loudness','mode','speechiness','acousticness',
                    'instrumentalness','liveness','valence','tempo_1']
    
    st.table(audio_features_df[0:3][feature_names_to_show])
        """)

if __name__ == "__main__":
    write()