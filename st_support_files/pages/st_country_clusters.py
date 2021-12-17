#Import statements
import plotly.express as px
import streamlit as st
import utils

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
    


if __name__ == "__main__":
    write()