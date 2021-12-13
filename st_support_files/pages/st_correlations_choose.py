import streamlit as st
import altair as alt
import utils
from datetime import datetime as dt

"""This builds out the 'Choose your own correlations' page in the Streamlit app"""

now = dt.now()
def write():
    """Used to write the page in the streamlit_app.py file"""

    ### Correlations
    with st.container():
        st.title('Correlations Matrices')
        st.write("""One part of our overall data analysis was taking audio features from Spotify and Librosa. Of course, since they both provide what,
                    on the surface seem like similar data points, it's important to understand if there's any duplication of metrics across the two
                    data sources. One way to analyze that is through correlation matrices.""")

        # load necessary data using function
        file_path, audio_features_df, playlist_data_df, global_pl_lookup, pl_w_audio_feats_df, basic_kmeans_inertia, adv_kmeans_inertia = utils.load_data()

    ### Correlation matrix market selector
    with st.container():
        col1, col2 = st.columns([5,1])
        col1.subheader("Market-specific Audio Feature Correlations")
        col1.write("""Here, you have the option to look at multiple or individual markets and see how the audio features from both Spotify and Librosa
                      compare to one another. Expand the filter options on the right to get started!""")
        country_selector = global_pl_lookup['country'].tolist()
        with col2.expander('Filter Options'):
            choice = st.multiselect('Choose a market', country_selector, default=country_selector)
        if choice:
            choice_df = pl_w_audio_feats_df[pl_w_audio_feats_df['country'].isin(choice)]
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

            col1.altair_chart(cor_plot + text, use_container_width=True)
        else:
            col1.markdown('')
            col1.error('Pick at least one country!')
        
        col1.subheader("Popularity Correlation Matrix")
        if choice:
            choice_df = pl_w_audio_feats_df[pl_w_audio_feats_df['country'].isin(choice)]
            audio_feat_corr = choice_df.corr().stack().reset_index().rename(columns={0: 'correlation', 'level_0': 'variable 1', 'level_1': 'variable 2'})

            audio_feat_corr_ct2 = audio_feat_corr.copy()[audio_feat_corr['variable 1']=='popularity']
            audio_feat_corr_ct2['correlation_label'] = audio_feat_corr_ct2['correlation'].map('{:.2f}'.format)

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
            
            col1.write("Taking it one step further, are there any features which correlate highly with popularity in a given market or set of markets?")
            col1.altair_chart(cor_plot2 + text, use_container_width=True)
        else:
            col1.markdown('')
            col1.error('Pick at least one country!')
        col1.text("\n")

    ### Audio features definitions expander
    with st.container():
        with st.expander("Audio feature definitions"):
            for k in utils.audio_feat_dict:
                st.markdown('**' + k +'** - ' + utils.audio_feat_dict[k])
            col1,col2 = st.columns([3,1])
            col2.markdown("Source: [Spotify's API documentation.](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features)")

if __name__ == "__main__":
    write()