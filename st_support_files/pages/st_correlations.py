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
        with st.expander("Audio feature definitions"):
            for k in utils.audio_feat_dict:
                st.markdown('**' + k +'** - ' + utils.audio_feat_dict[k])
            col1,col2 = st.columns([3,1])
            col2.markdown("Source: [Spotify's API documentation.](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features)")

        st.write("It looks like there are a handful of audio features that have high correlations with others.")

if __name__ == "__main__":
    write()