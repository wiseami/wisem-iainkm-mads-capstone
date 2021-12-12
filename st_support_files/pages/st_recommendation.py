import pandas as pd
import streamlit as st
import requests
import utils
from sklearn.metrics.pairwise import cosine_similarity

def write():
    """Used to write the page in the streamlit_app.py file"""
    # Spotify info
    headers, market, SPOTIFY_BASE_URL = utils.spotify_info()
    
    ### Density Plots
    # load necessary data using function
    file_path, audio_features_df, playlist_data_df, global_pl_lookup, pl_w_audio_feats_df, basic_kmeans_inertia, adv_kmeans_inertia = utils.load_data()

    ### Recommendations
    with st.container():
        st.subheader('Recommendations')
        st.write('Search for artist to get top 5 songs. Clicking on a song checks our lookups first and if the song isnt there itll run a lookup against spotify API, bring audio features back.') 

        search_term = st.text_input('Search an artist')

        search_term = 'johnny cash' #only here for testing
        search = requests.get(SPOTIFY_BASE_URL + 'search?q=artist:' + search_term + '&type=artist', headers=headers)
        search = search.json()

        feats_to_show_streamlit = ['artist', 'name','danceability','energy','key','loudness','mode','speechiness','acousticness',
                        'instrumentalness','liveness','valence', 'album_img','preview_url']
        
        with st.spinner(text='Recommendizer running...'):
            for item in search['artists']['items'][0:1]:
                searchy = requests.get(SPOTIFY_BASE_URL + 'artists/' + item['id'] + '/top-tracks?market=US', headers=headers).json()
                st.write('Pick one of these top 5 songs for this artist.')
                for top_tracks in searchy['tracks'][0:5]:
                    if st.button(top_tracks['name']):
                        if audio_features_df['track_id'].str.contains(top_tracks['id']).any():
                            final_df = audio_features_df[audio_features_df['track_id']==top_tracks['id']]
                            final_df = final_df.drop(columns=['pl_count']).reset_index(drop=True)
                            st.write('Features already on record:')
                            st.dataframe(final_df[feats_to_show_streamlit])
                        else:
                            audio_feats = requests.get(SPOTIFY_BASE_URL + 'audio-features?ids=' + top_tracks['id'], headers=headers).json()
                            audio_features_df_new = utils.get_audio_features(audio_feats)
                            
                            track_info = requests.get(SPOTIFY_BASE_URL + 'tracks?ids=' + top_tracks['id'], headers=headers).json()
                            track_info_df = utils.get_track_info(track_info)

                            final_df = audio_features_df_new.merge(track_info_df, how='inner', on='id')
                            final_df.rename(columns={'id':'track_id'}, inplace=True)
                            final_df.to_csv('audio_files/df.csv', index=False)
                            st.write('Features pulled from API:')
                            st.dataframe(final_df[feats_to_show_streamlit])

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
                            

            try:
                st.markdown('---')
                
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
                    #cossim_df = cossim_df.drop(columns=['update_dttm', 'time_signature', 'name','artist','album_img','preview_url', 'basic_kmeans_cluster', 'popularity', 'pl_count', 'adv_kmeans_cluster','duration_ms'])
                    #cossim_df = cossim_df.drop(columns=lib_feats)
                    cossim_df_y = cossim_df['track_id']
                    cossim_df = cossim_df.drop(columns=['track_id'])

                    compare_df = bas_final_df.copy()
                    compare_df = compare_df[spot_feats]
                    #compare_df = compare_df.drop(columns=['update_dttm', 'time_signature', 'name','artist','album_img','preview_url', 'basic_kmeans_cluster', 'popularity', 'duration_ms'])
                    compare_df_y = compare_df['track_id']
                    compare_df = compare_df.drop(columns=['track_id'])

                    cossim_df_f = cossim_df.copy()[compare_df.columns.tolist()]

                    cossim_df_f['basic_sim'] = cossim_df_f.apply(lambda x: cosine_similarity(compare_df.values.reshape(1,-1), x.values.reshape(1,-1))[0][0], axis=1)
                    cossim_df_f['track_id'] = cossim_df_y
                    cossim_df_f = cossim_df_f[cossim_df_f['basic_sim'] < 1]
                    cossim_df_sort = cossim_df_f.sort_values('basic_sim',ascending=False)[0:5]
                    cossim_df_sort = cossim_df_sort.merge(audio_features_df[['track_id','name','artist','album_img','preview_url']], how='inner', on='track_id')

                    compare_df['track_id'] = compare_df_y
                    compare_df = compare_df.merge(bas_final_df[['track_id','name','artist','album_img','preview_url']], how='inner', on='track_id')

                    st.write('Basic KMeans approach')
                    col1, col2 = st.columns([1,1])
                    col1.markdown("If you like: *" + compare_df['name'].iloc[0] + "* by " + compare_df['artist'].iloc[0])
                    col1.image(compare_df['album_img'].iloc[0])
                    col1.dataframe(compare_df[feats_to_show_streamlit])
                    if pd.isna(compare_df['preview_url'].iloc[0]) == False:
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

                    
                
                else:
                    adv_df = utils.do_kmeans_advanced_on_fly(final_df) #utils.do_kmeans_advanced_on_fly(final_df)

                    pl_feat_merge = playlist_data_df.merge(audio_features_df, how='inner', on='track_id')
                    clusters_by_country = pl_feat_merge.groupby(['country', 'adv_kmeans_cluster'])['track_id'].count().sort_values(ascending=False).reset_index()
                    
                    tops = clusters_by_country[clusters_by_country['adv_kmeans_cluster']==adv_df['adv_kmeans_cluster'].item()].sort_values(by='track_id', ascending=False)[0:1]
                    top_pl_track_ids = playlist_data_df[playlist_data_df['country'] == tops['country'].item()]['track_id']
                    
                    cossim_df = audio_features_df[audio_features_df['track_id'].isin(top_pl_track_ids)]
                    cossim_df = cossim_df[feats_to_keep]
                    #cossim_df = cossim_df.drop(columns=['update_dttm', 'time_signature', 'name','artist','album_img','preview_url', 'basic_kmeans_cluster', 'popularity', 'pl_count', 'adv_kmeans_cluster','duration_ms'])
                    cossim_df_y = cossim_df['track_id']
                    cossim_df = cossim_df.drop(columns=['track_id'])

                    
                    compare_df = adv_df.copy()
                    compare_df = compare_df[feats_to_keep]
                    #compare_df = compare_df.drop(columns=['update_dttm', 'time_signature', 'name','artist','album_img','preview_url', 'popularity','duration_ms', 'adv_kmeans_cluster'])
                    compare_df_y = compare_df['track_id']
                    compare_df = compare_df.drop(columns=['track_id'])

                    cossim_df_f = cossim_df.copy()[compare_df.columns.tolist()]

                    cossim_df_f['adv_sim'] = cossim_df_f.apply(lambda x: cosine_similarity(compare_df.values.reshape(1,-1), x.values.reshape(1,-1))[0][0], axis=1)
                    cossim_df_f['track_id'] = cossim_df_y
                    cossim_df_f = cossim_df_f[cossim_df_f['adv_sim'] < 1]
                    cossim_df_sort = cossim_df_f.sort_values('adv_sim',ascending=False)[0:5]
                    cossim_df_sort = cossim_df_sort.merge(audio_features_df[['track_id','name','artist','album_img','preview_url']], how='inner', on='track_id')

                    compare_df['track_id'] = compare_df_y
                    compare_df = compare_df.merge(adv_df[['track_id','name','artist','album_img','preview_url']], how='inner', on='track_id')

                    st.write('Advanced KMeans approach')
                    col1, col2 = st.columns([1,1])
                    col1.markdown("If you like: *" + compare_df['name'].iloc[0] + "* by " + compare_df['artist'].iloc[0])
                    col1.image(compare_df['album_img'].iloc[0])
                    col1.dataframe(compare_df[feats_to_show_streamlit])
                    if pd.isna(compare_df['preview_url'].iloc[0]) == False:
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

if __name__ == "__main__":
    write()
