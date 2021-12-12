import streamlit as st
import altair as alt
import utils
from datetime import datetime as dt


### Start building out Streamlit assets

now = dt.now()
def write():
    """Used to write the page in the streamlit_app.py file"""

    ### Correlations
    with st.container():
        st.header('Correlations')
        st.write("Some text here about correlations and audio features...")

        # load necessary data using function
        file_path, audio_features_df, playlist_data_df, global_pl_lookup, pl_w_audio_feats_df, basic_kmeans_inertia, adv_kmeans_inertia = utils.load_data()
        
        # Correlation matrix        
        st.title("Correlation Matrix")

        audio_feat_corr, audio_feat_corr_ct1, audio_feat_corr_ct2 = utils.corr_matrix()

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
        cor_plot1 = base.mark_rect().encode(
            color=alt.Color('correlation:Q', scale=alt.Scale(scheme='greenblue'))
        )

        st.write("Now, let's take country out of the equation and have a closer look at the different individual audio features across all distinct tracks.")
        st.altair_chart(cor_plot1 + text, use_container_width=True)

    ### Correlations between spotify and librosa features
    with st.container():
        st.header('Correlations - Spotiy vs Librosa')

        audio_feat_corr = audio_features_df.drop(columns=['time_signature','update_dttm','name','artist','album_img','preview_url', 'duration_ms'])
        audio_feat_corr = audio_feat_corr.corr().stack().reset_index().rename(columns={0: 'correlation', 'level_0': 'variable 1', 'level_1': 'variable 2'})
        #audio_feat_corr.to_csv('st_support_files/cache/audio_feat_corr.csv', index=False)

        audio_feat_corr_ct1 = audio_feat_corr.copy()[(audio_feat_corr['variable 1']!='pl_count') & (audio_feat_corr['variable 1']!='popularity') & (audio_feat_corr['variable 2']!='pl_count') & (audio_feat_corr['variable 2']!='popularity')]
        audio_feat_corr_ct1['correlation_label'] = audio_feat_corr_ct1['correlation'].map('{:.2f}'.format)
        #audio_feat_corr_ct1.to_csv('st_support_files/cache/audio_feat_corr_ct1.csv', index=False)

        # audio_feat_corr_ct2 = audio_feat_corr.copy()[(audio_feat_corr['variable 1']=='pl_count') | (audio_feat_corr['variable 1']=='popularity')]
        # audio_feat_corr_ct2['correlation_label'] = audio_feat_corr_ct2['correlation'].map('{:.2f}'.format)
        # #audio_feat_corr_ct2.to_csv('st_support_files/cache/audio_feat_corr_ct2.csv', index=False)

        # audio_feat_corr['variable 1'].unique()

        spot_feats = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence',
            'tempo_1']

        lib_feats = ['chroma',
            'chroma_cens', 'mff', 'spectral_bandwidth', 'spectral_contrast',
            'spectral_flatness', 'Spectral_Rolloff', 'poly_features',
            'tonnetz', 'ZCR', 'onset_strength', 'pitch', 'magnitude',
            'tempo_2']

        audio_feat_corr_ct1 = audio_feat_corr.copy()[(audio_feat_corr['variable 1'].isin(spot_feats)) & (audio_feat_corr['variable 2'].isin(lib_feats))]
        audio_feat_corr_ct1['correlation_label'] = audio_feat_corr_ct1['correlation'].map('{:.2f}'.format)
                
        # Correlation matrix        
        #audio_feat_corr, audio_feat_corr_ct1, audio_feat_corr_ct2 = utils.corr_matrix()

        base = alt.Chart(audio_feat_corr_ct1).encode(
            x='variable 2:O',
            y='variable 1:O'    
        )

        # Text layer with correlation labels
        # Colors are for easier readability
        text = base.mark_text().encode(
            text='correlation_label',
            color=alt.condition(
                alt.datum.correlation > 0.5, 
                alt.value('white'),
                alt.value('black')
            )
        )

        # The correlation heatmap itself
        cor_plot1 = base.mark_rect().encode(
            color=alt.Color('correlation:Q', scale=alt.Scale(scheme='greenblue'))
        )

        st.altair_chart(cor_plot1 + text, use_container_width=True)

    ### Popularity features correlation
    st.subheader("Popularity Correlation Matrix")
    with st.container():
        st.write("But how do these audio features correlate with popularity features?")
        
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
        cor_plot2 = base.mark_rect().encode(
            color=alt.Color('correlation:Q', scale=alt.Scale(scheme='orangered'))
        )
        
        st.altair_chart(cor_plot2 + text, use_container_width=True)
        st.write("Interesting to see that neither popularity (a Spotify measure) or playlist count (number of distinct market playlists a song shows up on) correlate very highly with any specific audio feature.")
        st.text("\n")

    ### Audio features definitions expander
    with st.container():
        with st.expander("Audio feature definitions"):
            for k in utils.audio_feat_dict:
                st.markdown('**' + k +'** - ' + utils.audio_feat_dict[k])
            col1,col2 = st.columns([3,1])
            col2.markdown("Source: [Spotify's API documentation.](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features)")

        st.write("It looks like there are a handful of audio features that have high correlations with others.")

if __name__ == "__main__":
    write()