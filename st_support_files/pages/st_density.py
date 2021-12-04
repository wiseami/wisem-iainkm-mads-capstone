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
        # def dens_plots():            
        #     feature_names = ['danceability','energy','key','loudness','mode','speechiness','acousticness',
        #                     'instrumentalness','liveness','valence','tempo', 'country']

        #     df_feat = pl_w_audio_feats_df[feature_names]
            
        #     charts = []
        #     for feat in feature_names:

        #         charts.append(alt.Chart(df_feat).transform_density(
        #             density=feat,
        #             groupby=['country']
        #         ).mark_line().encode(
        #             alt.X('value:Q',title=feat),
        #             alt.Y('density:Q'),
        #             alt.Color('country:N',legend=None),
        #             tooltip='country'
        #         ))

        #     chart1 = charts[0]
        #     chart2 = charts[1]
        #     chart3 = charts[3]
        #     chart4 = charts[9]

        #     return chart1, chart2, chart3, chart4, charts
        
        # chart1, chart2, chart3, chart4, charts = dens_plots()

        st.title('Density Plots')
        st.write("""Knowing we have 69 playlists makes these visuals not-so-easy to consume, but it seemed worth showing the density plots for a couple of audio features across all countries where each line is a country. 
                    Definitions on left directly from Spotify's [API documentation.](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features)""")
        
        country_selector = global_pl_lookup['country'].tolist()
        
        st.write("Pick any of the markets Spotify generates a playlist for to see how the different features correlate to one another just within that market.")
        choice = st.multiselect('Choose a market', country_selector, default=country_selector)
        #choice = 'global'
        feature_names = ['danceability','energy','key','loudness','mode','speechiness','acousticness',
                             'instrumentalness','liveness','valence','tempo', 'country']
        if choice:
            df_feat = pl_w_audio_feats_df[feature_names]
            
            choice_df = df_feat[df_feat['country'].isin(choice)]

            feature_names = ['danceability','energy','key','loudness','mode','speechiness','acousticness',
                                'instrumentalness','liveness','valence','tempo', 'country']

                
            charts = []
            for feat in feature_names:

                charts.append(alt.Chart(choice_df).transform_density(
                    density=feat,
                    groupby=['country']
                ).mark_line().encode(
                    alt.X('value:Q',title=feat),
                    alt.Y('density:Q'),
                    alt.Color('country:N',legend=None),
                    tooltip='country'
                ))

            chart1 = charts[0]
            chart2 = charts[1]
            chart3 = charts[3]
            chart4 = charts[9]
            
            # for feat in feature_names:
            #     alt.Chart(choice_df).transform_density(
            #             density=feat,
            #             groupby=['country']
            #         ).mark_line().encode(
            #             alt.X('value:Q'),
            #             alt.Y('density:Q'),
            #             alt.Color('country:N',legend=None),
            #             tooltip='country'
            #         )
            #return chart1, chart2, chart3, chart4, charts
            
            #chart1, chart2, chart3, chart4, charts = dens_plots()
            

            col1, col2 = st.columns([1,2])
            col1.markdown('**Danceability** - Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.')
            col2.altair_chart(charts[0], use_container_width=True)

            col1, col2 = st.columns([1,2])
            col1.markdown('**Energy** - Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.')
            col2.altair_chart(charts[1], use_container_width=True)

            col1, col2 = st.columns([1,2])
            col1.markdown('**Loudness** - The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.')
            col2.altair_chart(chart3, use_container_width=True)

            col1, col2 = st.columns([1,2])
            col1.markdown('**Valence** - A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).')
            col2.altair_chart(chart4, use_container_width=True)

            for chart in charts:
                st.altair_chart(chart)
        else:
            st.markdown('')
            st.error('Pick at least one country!')


        # choice = col1.selectbox('Choose a market', country_selector)#, default=country_selector)
        # choice = 'australia'
        # choice_df = pl_w_audio_feats_df[pl_w_audio_feats_df['country'].isin([choice])]
        # country_selector = global_pl_lookup['country'].tolist()
        # col1, col2 = st.columns([1,2])
        # col1.write("Pick any of the markets Spotify generates a playlist for to see how the different features correlate to one another just within that market.")

if __name__ == "__main__":
    write()