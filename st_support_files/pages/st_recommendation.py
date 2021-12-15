import pandas as pd
import streamlit as st
import requests
import utils
import streamlit.components.v1 as components

"""This builds out the Recommendations page in the Streamlit app"""

def write():
    """Used to write the page in the streamlit_app.py file"""
    # Spotify info
    headers, market, SPOTIFY_BASE_URL = utils.spotify_info()
    
    # load necessary data using function
    file_path, audio_features_df, playlist_data_df, global_pl_lookup, pl_w_audio_feats_df, basic_kmeans_inertia, adv_kmeans_inertia = utils.load_data()

    ### Recommendations
    with st.container():
        st.title('Better Living through Music Recommendation')
        st.write("""After analyzing music features across different playlists in the world, our next thought was how we could leverage that into a real-world tool.
                    While companies like Spotify build recommendations into their apps based on listening preferences, we thought it would be interesting to build
                    something where a user could search for an artist or song and find a world playlist that has similar features. It may suggest songs you've heard of
                    or it could suggest ones that aren't in even in your native tongue but the audio features are similar to what you searched for.
                 """)
        
        feats_to_show_streamlit = ['artist', 'name','danceability','energy','key','loudness','mode','speechiness','acousticness',
                    'instrumentalness','liveness','valence', 'album_img','preview_url']
        
        search_type = st.radio("Pick one:", ['Artist','Song name'])
        
        if search_type == 'Artist':
            search_term1 = st.text_input('Search an artist')
            if search_term1:
                search = requests.get(SPOTIFY_BASE_URL + 'search?q=artist:' + search_term1 + '&type=artist', headers=headers)
                search = search.json()
            
                button_count_1 = 1
                with st.spinner(text='Searching...'):
                    for item in search['artists']['items'][0:1]:
                        searchy = requests.get(SPOTIFY_BASE_URL + 'artists/' + item['id'] + '/top-tracks?market=US', headers=headers).json()
                        st.write('Pick one of these top 5 songs for this artist.')
                        for top_tracks in searchy['tracks'][0:5]:
                            button_text = top_tracks['artists'][0]['name'] + ' - ' + top_tracks['name']
                            if st.button(button_text, key = button_count_1):
                                if audio_features_df['track_id'].str.contains(top_tracks['id']).any():
                                    final_df = audio_features_df[audio_features_df['track_id']==top_tracks['id']]
                                    final_df = final_df.drop(columns=['pl_count']).reset_index(drop=True)

                                else:
                                    audio_feats = requests.get(SPOTIFY_BASE_URL + 'audio-features?ids=' + top_tracks['id'], headers=headers).json()
                                    audio_features_df_new = utils.get_audio_features(audio_feats)
                                    
                                    track_info = requests.get(SPOTIFY_BASE_URL + 'tracks?ids=' + top_tracks['id'], headers=headers).json()
                                    track_info_df = utils.get_track_info(track_info)

                                    final_df = audio_features_df_new.merge(track_info_df, how='inner', on='id')
                                    final_df.rename(columns={'id':'track_id'}, inplace=True)
                                    final_df.to_csv('audio_files/df.csv', index=False)

                            button_count_1 += 2
        
        elif search_type == 'Song name':
            search_term2 = st.text_input('Search a song name')
            if search_term2:
                search = requests.get(SPOTIFY_BASE_URL + 'search?q=track:' + search_term2 + '&type=track', headers=headers)
                search = search.json()

                button_count_2 = 2
                with st.spinner(text='Searching...'):
                    st.write('Pick the one that most resembles your search.')
                    for item in search['tracks']['items'][0:5]:
                        button_text_2 = item['artists'][0]['name'] + ' - ' + item['name']
                        if st.button(button_text_2, key=button_count_2):
                            if audio_features_df['track_id'].str.contains(item['id']).any():
                                final_df = audio_features_df[audio_features_df['track_id']==item['id']]
                                final_df = final_df.drop(columns=['pl_count']).reset_index(drop=True)
                            
                            else:
                                audio_feats = requests.get(SPOTIFY_BASE_URL + 'audio-features?ids=' + item['id'], headers=headers).json()
                                audio_features_df_new = utils.get_audio_features(audio_feats)
                                
                                track_info = requests.get(SPOTIFY_BASE_URL + 'tracks?ids=' + item['id'], headers=headers).json()
                                track_info_df = utils.get_track_info(track_info)

                                final_df = audio_features_df_new.merge(track_info_df, how='inner', on='id')
                                final_df.rename(columns={'id':'track_id'}, inplace=True)
                                final_df.to_csv('audio_files/df.csv', index=False)
                        button_count_2 += 2
        
        st.markdown('---')            
        with st.spinner(text='Recommendizer running...'):
            try:
                compare_df, final_playlist, cossim_df_sort = utils.Recommendizer(final_df)

                col1, col3, col2 = st.columns([2,1,2])
                with col1:
                    col1.subheader("If you like...")
                    col1.markdown("*" + compare_df['name'].iloc[0] + "* by " + compare_df['artist'].iloc[0])
                    col1.image(compare_df['album_img'].iloc[0])
                    components.iframe('https://open.spotify.com/embed/track/' + compare_df['track_id'].iloc[0], height=75)

                with col2:
                    st.subheader("You might like...")
                    st.markdown(final_playlist['name'].item())
                    st.image(final_playlist['playlist_img'].item())
                    #st.write("Top 5 closest matching songs in this playlist")
                    for x in cossim_df_sort[feats_to_show_streamlit].index:
                        components.iframe('https://open.spotify.com/embed/track/' + cossim_df_sort['track_id'].iloc[x], height=75)
                    st.write("")
                    st.write("")
                    st.write("Not so much? Check out the rest of the playlist.")
                    components.iframe('https://open.spotify.com/embed/playlist/' + final_playlist['id'].iloc[0], height = 500)
            
            except:
                pass

    with st.expander("How does this work?"):
        st.write("""[summarize how this works]""")

if __name__ == "__main__":
    write()
