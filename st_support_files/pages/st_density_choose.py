import streamlit as st
import altair as alt
import utils

"""This builds out the 'choose your own density plot' page of the Streamlit app"""

def write():
    """Used to write the page in the streamlit_app.py file"""
    ### Density Plots
    # load necessary data using function
    file_path, audio_features_df, playlist_data_df, global_pl_lookup, pl_w_audio_feats_df, basic_kmeans_inertia, adv_kmeans_inertia = utils.load_data()
    
    
    
    with st.container():
        st.title("Data Distribution via Density Plot")
        st.write("""One way to quickly see how the data is distrubted across markets is to take the audio features for each
                    individual song, roll those up to the country and plot said distribution. By default, this opens with
                    all 69 lists selected.""")
        st.write("""Expand filter options on the right side and select or deselect any of the markets Spotify generates a 
                    playlist for and any audio feature to see market comparisons.""")
        
        col1, col2 = st.columns([3,1])
        with col2.expander('Filter Options'):
            all_feature_names = ['danceability','energy','loudness','speechiness','acousticness',
                                'instrumentalness','liveness','valence','tempo_1', 'chroma', 'chroma_cens', 'mff',
                            'spectral_centroid', 'spectral_bandwidth', 'spectral_contrast',
                            'spectral_flatness', 'Spectral_Rolloff', 'poly_features', 'tonnetz',
                            'ZCR', 'onset_strength', 'pitch', 'magnitude', 'tempo_2','country'] #, 'key', 'mode']
            feature_selector = all_feature_names[:-1]
            feature_choice = st.selectbox('Choose an audio feature', feature_selector)

            country_selector = global_pl_lookup['country'].tolist()
            country_choice = st.multiselect('Choose a market', country_selector, default=country_selector)
        
        if country_choice and feature_choice:
            df_feat = pl_w_audio_feats_df[all_feature_names]
            
            choice_df = df_feat[['country',feature_choice]][df_feat['country'].isin(country_choice)]

            col1.altair_chart(alt.Chart(choice_df).transform_density(
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