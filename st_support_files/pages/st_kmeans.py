import streamlit as st
import altair as alt
import utils

### Start building out Streamlit assets
def write():
    """Used to write the page in the streamlit_app.py file"""

    # load necessary data using function
    file_path, audio_features_df, playlist_data_df, global_pl_lookup, pl_w_audio_feats_df, basic_kmeans_inertia, adv_kmeans_inertia = utils.load_data()

    ### KMeans
    with st.container():
        st.title('KMeans basic')
        st.write('checking inertia and silhouette scores to find best k')
        #st.write("need a ton more here I think")
        alt_chart1, alt_chart2 = st.columns(2)
        alt_intertia = alt.Chart(basic_kmeans_inertia[['k','inertia']]).mark_line().encode(
            x='k:O',
            y=alt.Y('inertia', scale=alt.Scale(domain=[14000,28000]))
        )
        alt_chart1.altair_chart(alt_intertia)

        alt_silhouette = alt.Chart(basic_kmeans_inertia[['k','silhouette_score']]).mark_line().encode(
            x='k:O',
            y=alt.Y('silhouette_score', scale=alt.Scale(domain=[.11,.2]))
        )
        alt_chart2.altair_chart(alt_silhouette)
    
    st.markdown('---')

    with st.container():
        st.title('KMeans adv')
        st.write('checking inertia and silhouette scores to find best k')
        #st.write("need a ton more here I think")
        alt_chart1, alt_chart2 = st.columns(2)
        alt_intertia = alt.Chart(adv_kmeans_inertia[['k','inertia']]).mark_line().encode(
            x='k:O',
            y=alt.Y('inertia', scale=alt.Scale(domain=[25000,50000]))
        )
        alt_chart1.altair_chart(alt_intertia)

        alt_silhouette = alt.Chart(adv_kmeans_inertia[['k','silhouette_score']]).mark_line().encode(
            x='k:O',
            y=alt.Y('silhouette_score', scale=alt.Scale(domain=[.06,.18]))
        )
        alt_chart2.altair_chart(alt_silhouette)

if __name__ == "__main__":
    write()