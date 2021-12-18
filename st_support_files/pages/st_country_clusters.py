#Import statements
from numpy import exp
import plotly.express as px
import streamlit as st
import utils
import pandas as pd

"""This builds out the 'clusters' page of the Streamlit app"""

def write():
    st.title("Clustering")
    st.markdown('---')
    st.write("""Clustering can be a very powerful tool to spot patterns that emerge across a dataset.""")
    
    ### Country Clustering
    with st.container():
    # load necessary data using function
        final = utils.country_clusters()

        fig = px.scatter_3d(final, x='PC1', y='PC2', z='PC3', color='cluster',
                            hover_data=['country','cluster'], width=1000,
                            height=1000)
        fig.update_layout(showlegend=False)

        st.subheader("Clusters by Country")
        st.write("""We took an average of every feature for all the songs that appeared on a country’s playlist 
                    to create a sort of unique audio feature fingerprint. This gave us a granularity of data
                    we could then use KMeans clustering for to identify similarities (and differences) across countries.
                 """)
        
        st.write("""Using PCA to reduce the dimensionality of our data set, we're able to see the 6 KMeans clusters - 
                    defined using interia and silhouette scoring - in a 3D space.""")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    with st.container():
        final = pd.read_csv('st_support_files\cache\hier_clusters.csv')
        df = pd.read_csv('lookups/all_track_audio_features.csv')
        df.dropna(axis=0, inplace=True)

        final['name'] = df['name']
        final['artist'] = df['artist']
        final['cluster'] = final['cluster'].astype(str)

        col1, col2 = st.columns([3,1])
        cluster_selector = ["0","1","2","3","4","5","6","7","8","9"]
        with col2.expander('Filter Options', expanded=True):
            choice = st.multiselect('Choose a cluster', cluster_selector, default=cluster_selector)
            if choice:
                choice_df = final[final['cluster'].isin(choice)]
        
                fig = px.scatter_3d(choice_df, x='PC1', y='PC2', z='PC3', color='cluster', color_discrete_map={
                    "0":'#140c89',
                    "1":'#4507a2',
                    "2":'#6e04ab',
                    "3":'#9617a0',
                    "4":'#B63687',
                    "5":'#D0566a',
                    "6":'#E5784e',
                    "7":'#f49e2c',
                    "8":'#F8c900',
                    "9":'#eff900'
                    },
                    category_orders={"cluster": ["0","1","2","3","4","5","6","7","8","9"]},
                                    hover_name='name',hover_data=['artist'], width=1000,
                                    height=1000)
            
                
                col1.subheader("Clusters by Individual Track")
                col1.write("""Here, we've left this at the individual track level and created 10 clusters based on
                            our findings with regard to distinct genres. Using PCA to reduce the dimensionality of our data set, we're able to see the 10 Agglomerative Hierearchical clusters in a 3D space.
                        """)
                
                col1.write("""Because it's such a large dataset, you have the ability to filter clusters in or out of the visual on the right side.""")
                col1.plotly_chart(fig, use_container_width=True)

            else:
                col1.markdown('')
                col1.error('Pick at least one cluster!')
if __name__ == "__main__":
    write()