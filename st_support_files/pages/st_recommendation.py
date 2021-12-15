import pandas as pd
import streamlit as st
import requests
import utils
from sklearn.metrics.pairwise import cosine_similarity
import pickle

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
        st.write("""Search for artist to get top 5 songs. Clicking on a song checks our lookups first and if the song isnt there itll run a lookup against Spotify API, 
                    bring audio features back.
                """) 

        spot_feats = ['track_id','danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence',
                'tempo_1']

        lib_feats = ['chroma',
            'chroma_cens', 'mff', 'spectral_bandwidth', 'spectral_contrast',
            'spectral_flatness', 'Spectral_Rolloff', 'poly_features',
            'tonnetz', 'ZCR', 'onset_strength', 'pitch', 'magnitude', 'tempo_2']
        
        feats_to_keep = ['track_id', 'danceability', 'energy', 'key', 'loudness', 'mode','speechiness', 'acousticness', 'instrumentalness', 'liveness','valence', 'tempo_1', 'chroma', 
                                'chroma_cens', 'mff', 'spectral_centroid', 'spectral_bandwidth','spectral_contrast', 'spectral_flatness', 'Spectral_Rolloff','poly_features', 'tonnetz', 
                                'ZCR', 'onset_strength', 'pitch','magnitude', 'tempo_2']
        
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
                if pd.isna(final_df['preview_url'][0]):
                    #BASIC KMEANS
                    bas_final_df = utils.do_kmeans_on_fly(final_df)

                    pl_feat_merge = playlist_data_df.merge(audio_features_df, how='inner', on='track_id')
                    clusters_by_country = pl_feat_merge.groupby(['country', 'basic_kmeans_cluster'])['track_id'].count().sort_values(ascending=False).reset_index()

                    # gets songs from top playlist with most number of songs in the same cluster
                    tops = clusters_by_country[clusters_by_country['basic_kmeans_cluster']==bas_final_df['basic_kmeans_cluster'].item()].sort_values(by='track_id', ascending=False)[0:1]
                    top_pl_track_ids = playlist_data_df[playlist_data_df['country'] == tops['country'].item()]['track_id']

                    cossim_df = audio_features_df[audio_features_df['track_id'].isin(top_pl_track_ids)]
                    cossim_df = cossim_df[spot_feats]
                    cossim_df_y = cossim_df['track_id']
                    cossim_df = cossim_df.drop(columns=['track_id'])

                    compare_df = bas_final_df.copy()
                    compare_df = compare_df[spot_feats]
                    compare_df_y = compare_df['track_id']
                    compare_df = compare_df.drop(columns=['track_id'])

                    cossim_df_f = cossim_df.copy()[compare_df.columns.tolist()]

                    basic_scaler = pickle.load(open("model/basic_scaler.pkl", "rb"))
                    scaled_cossim = basic_scaler.transform(cossim_df_f)
                    scaled_compare = basic_scaler.transform(compare_df)

                    cossim_df_f['basic_sim'] = cosine_similarity(scaled_cossim, scaled_compare)

                    #cossim_df_f['basic_sim'] = cossim_df_f.apply(lambda x: cosine_similarity(compare_df.values.reshape(1,-1), x.values.reshape(1,-1))[0][0], axis=1)
                    cossim_df_f['track_id'] = cossim_df_y
                    cossim_df_f = cossim_df_f[cossim_df_f['basic_sim'] < 1]
                    cossim_df_sort = cossim_df_f.sort_values('basic_sim',ascending=False)[0:5]
                    cossim_df_sort = cossim_df_sort.merge(audio_features_df[['track_id','name','artist','album_img','preview_url']], how='inner', on='track_id')

                    compare_df['track_id'] = compare_df_y
                    compare_df = compare_df.merge(bas_final_df[['track_id','name','artist','album_img','preview_url']], how='inner', on='track_id')

                    #st.write('Basic KMeans approach')
                    col1, col3, col2 = st.columns([2,1,2])
                    col1.subheader("If you like...")
                    col1.markdown("[" + compare_df['name'].iloc[0] + "](https://open.spotify.com/track/" + compare_df['track_id'].iloc[0] + ") by " + compare_df['artist'].iloc[0])
                    col1.image(compare_df['album_img'].iloc[0])
                    if pd.isna(compare_df['preview_url'].iloc[0]) == False:
                        col1.audio(compare_df['preview_url'].iloc[0])
                    col1.dataframe(compare_df)

                    final_playlist = global_pl_lookup[global_pl_lookup['country']==tops['country'].item()]
                    col2.subheader("You might like...")
                    col2.markdown("["+ final_playlist['name'].item() +"](" + final_playlist['link'].item() + ")")
                    col2.image(final_playlist['playlist_img'].item())
                    #col2.markdown("[![this is an image link]("+final_playlist['playlist_img'].item()+")]("+final_playlist['link'].item()+")")
                    #col2.dataframe(cossim_df_sort[feats_to_show_streamlit])

                    for x in cossim_df_sort[feats_to_show_streamlit].index:
                        col2.write("[" + cossim_df_sort[feats_to_show_streamlit]['name'].iloc[x] + "](https://open.spotify.com/track/" + cossim_df_sort['track_id'].iloc[x] + ") by " + cossim_df_sort[feats_to_show_streamlit]['artist'].iloc[x])
                        if pd.isna(cossim_df_sort[feats_to_show_streamlit]['preview_url'].iloc[x]) == False:
                            col2.audio(cossim_df_sort[feats_to_show_streamlit]['preview_url'].iloc[x])
                    col2.dataframe(cossim_df_sort)
                    
                
                else:
                    adv_df = utils.do_kmeans_advanced_on_fly(final_df) #utils.do_kmeans_advanced_on_fly(final_df)

                    pl_feat_merge = playlist_data_df.merge(audio_features_df, how='inner', on='track_id')
                    clusters_by_country = pl_feat_merge.groupby(['country', 'adv_kmeans_cluster'])['track_id'].count().sort_values(ascending=False).reset_index()
                    
                    tops = clusters_by_country[clusters_by_country['adv_kmeans_cluster']==adv_df['adv_kmeans_cluster'].item()].sort_values(by='track_id', ascending=False)[0:1]
                    top_pl_track_ids = playlist_data_df[playlist_data_df['country'] == tops['country'].item()]['track_id']
                    
                    cossim_df = audio_features_df[audio_features_df['track_id'].isin(top_pl_track_ids)]
                    cossim_df = cossim_df[feats_to_keep]
                    cossim_df_y = cossim_df['track_id']
                    cossim_df = cossim_df.drop(columns=['track_id'])

                    compare_df = adv_df.copy()
                    compare_df = compare_df[feats_to_keep]
                    compare_df_y = compare_df['track_id']
                    compare_df = compare_df.drop(columns=['track_id'])

                    cossim_df_f = cossim_df.copy()[compare_df.columns.tolist()]

                    adv_scaler = pickle.load(open("model/adv_scaler.pkl", "rb"))
                    scaled_cossim = adv_scaler.transform(cossim_df_f)
                    scaled_compare = adv_scaler.transform(compare_df)

                    cossim_df_f['adv_sim'] = cosine_similarity(scaled_cossim, scaled_compare)

                    #cossim_df_f['adv_sim'] = cossim_df_f.apply(lambda x: cosine_similarity(compare_df.values.reshape(1,-1), x.values.reshape(1,-1))[0][0], axis=1)
                    cossim_df_f['track_id'] = cossim_df_y
                    cossim_df_f = cossim_df_f[cossim_df_f['adv_sim'] < 1]
                    cossim_df_sort = cossim_df_f.sort_values('adv_sim',ascending=False)[0:5]
                    cossim_df_sort = cossim_df_sort.merge(audio_features_df[['track_id','name','artist','album_img','preview_url']], how='inner', on='track_id')

                    compare_df['track_id'] = compare_df_y
                    compare_df = compare_df.merge(adv_df[['track_id','name','artist','album_img','preview_url']], how='inner', on='track_id')

                    #st.write('Advanced KMeans approach')
                    col1, col3, col2 = st.columns([2,1,2])
                    col1.subheader("If you like...")
                    col1.markdown("[" + compare_df['name'].iloc[0] + "](https://open.spotify.com/track/" + compare_df['track_id'].iloc[0] + ") by " + compare_df['artist'].iloc[0])
                    col1.image(compare_df['album_img'].iloc[0])
                    if pd.isna(compare_df['preview_url'].iloc[0]) == False:
                        col1.audio(compare_df['preview_url'].iloc[0])
                    col1.dataframe(compare_df)

                    final_playlist = global_pl_lookup[global_pl_lookup['country']==tops['country'].item()]
                    col2.subheader("You might like...")
                    col2.markdown("["+ final_playlist['name'].item() +"](" + final_playlist['link'].item() + ")")
                    col2.image(final_playlist['playlist_img'].item())
                    #col2.markdown("[![this is an image link]("+final_playlist['playlist_img'].item()+")]("+final_playlist['link'].item()+")")
                    #col2.dataframe(cossim_df_sort[feats_to_show_streamlit])

                    for x in cossim_df_sort[feats_to_show_streamlit].index:
                        col2.write("[" + cossim_df_sort[feats_to_show_streamlit]['name'].iloc[x] + "](https://open.spotify.com/track/" + cossim_df_sort['track_id'].iloc[x] + ") by " + cossim_df_sort[feats_to_show_streamlit]['artist'].iloc[x])
                        if pd.isna(cossim_df_sort[feats_to_show_streamlit]['preview_url'].iloc[x]) == False:
                            col2.audio(cossim_df_sort[feats_to_show_streamlit]['preview_url'].iloc[x])
                    col2.dataframe(cossim_df_sort)

            except:
                pass
    with st.expander("How does this work?"):
        st.write("""[summarize how this works]""")

if __name__ == "__main__":
    write()
