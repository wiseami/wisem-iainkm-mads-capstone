import pandas as pd
import streamlit as st
import requests
import tqdm
import altair as alt
#from altair.utils.schemapi import SchemaValidationError
import numpy as np
import utils
import pickle

### Spotify info
headers, market, SPOTIFY_BASE_URL = utils.spotify_info()

# load necessary data using function
file_path, top_pl_df, audio_features_df, playlist_data_df, global_lookup, pl_w_audio_feats_df, kmeans_inertia = utils.load_data()

# Normalize spotify audio features and create playlist rollups
playlist_audio_feature_rollup = utils.normalize_spotify_audio_feats(pl_w_audio_feats_df)

### Start building out Streamlit assets
st.set_page_config(layout="wide")
st.title('Spotify Streamlit')
st.write('this is a test')
st.markdown('---')
st.subheader('Top 3 Songs Based on number of playlist appearances')
st.write("While the first day of scraping playlists came back with 3,450 total songs, only about half of those were unique. Because of that, we have tons of tracks that show up on multiple playlists. We're looking at a total of 69 daily playlists - 68 country-specific and 1 global - and these songs below show up on multiple different country playlists.")

# st.markdown('---')
df = pd.DataFrame(playlist_data_df.groupby(['track_name', 'track_artist','track_id'])['country'].count().sort_values(ascending=False).reset_index()).head()
df.columns = ['Track Name', 'Artist', 'Track ID', '# Playlist Appearances']
df = df.merge(audio_features_df[['id','album_img','preview_url']], how='inner', left_on='Track ID', right_on='id')

top_songs = st.columns(3)
for i in range(0,3):
    top_songs[i].metric(label='Playlist appearances', value=int(df['# Playlist Appearances'][i]))
    top_songs[i].markdown('**' + df['Artist'][i] + " - " + df['Track Name'][i] + '**')
    top_songs[i].image(df['album_img'][i])
    if pd.isna(df['preview_url'][i]) == False:
        top_songs[i].audio(df['preview_url'][i])


st.write("Let's take a look at the audio features computed and captured by Spotify for these three songs.")
feature_names_to_show = ['artist', 'name','danceability','energy','key','loudness','mode','speechiness','acousticness',
                'instrumentalness','liveness','valence','tempo']
st.table(audio_features_df[0:3][feature_names_to_show])

feature_names = ['danceability','energy','key','loudness','mode','speechiness','acousticness',
                'instrumentalness','liveness','valence','tempo', 'duration_ms', 'country']

df_feat = pl_w_audio_feats_df[feature_names]

charts = []
for feat in feature_names:

    charts.append(alt.Chart(df_feat).transform_density(
        density=feat,
        groupby=['country']
    ).mark_line().encode(
        alt.X('value:Q',title=feat),
        alt.Y('density:Q'),
        alt.Color('country:N',legend=None),
        tooltip='country'
    ))

st.markdown('---')
st.header('Density Plots')
st.write("""Knowing we have 69 playlists makes these visuals not-so-easy to consume, but it seemed worth showing the density plots for a couple of audio features across all countries where each line is a country. 
            Definitions on left directly from Spotify's [API documentation.](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features)""")

col1, col2 = st.columns([1,2])
col1.markdown('**Danceability** - Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.')
col2.altair_chart(charts[0], use_container_width=True)

col1, col2 = st.columns([1,2])
col1.markdown('**Energy** - Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.')
col2.altair_chart(charts[1], use_container_width=True)

col1, col2 = st.columns([1,2])
col1.markdown('**Loudness** - The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.')
col2.altair_chart(charts[3], use_container_width=True)

col1, col2 = st.columns([1,2])
col1.markdown('**Valence** - A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).')
col2.altair_chart(charts[9], use_container_width=True)


st.markdown('---')
st.header('Correlations')

audio_feat_corr = playlist_audio_feature_rollup.corr().stack().reset_index().rename(columns={0: 'correlation', 'level_0': 'variable 1', 'level_1': 'variable 2'})
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

col1, col2 = st.columns([1,2])
col1.write("Now, let's take country out of the equation and have a closer look at the different individual audio features and how they correlate with one another. In this case, we created an aggregate country-playlist value of each of the individual song audio features and normalized for total duration of the playlist.")
col2.altair_chart(cor_plot + text, use_container_width=True)

audio_feat_dict = {
    "Acousticness":"A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.",
    "Danceability":"Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.",
    "Energy":"Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.",
    "Instrumentalness":"Predicts whether a track contains no vocals. ""Ooh"" and ""aah"" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly ""vocal"". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.",
    "Key":"The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on.",
    "Liveness":"Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.",
    "Loudness":"The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.",
    "Mode":"Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.",
    "Speechiness":"Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.",
    "Tempo":"The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.",
    "Valence":"A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)."    
    }

with st.expander("Audio feature definitions"):
    for k in audio_feat_dict:
        st.markdown('**' + k +'** - ' + audio_feat_dict[k])
    col1,col2 = st.columns([3,1])
    col2.markdown("Source: [Spotify's API documentation.](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features)")

st.write("It looks like there are a handful of audio features that have high correlations with others.")

st.markdown('---')
st.header('KMeans')
st.write('checking inertia and silhouette scores to find best k')
alt_chart1, alt_chart2 = st.columns(2)
alt_intertia = alt.Chart(kmeans_inertia[['k','inertia']]).mark_line().encode(
    x='k:O',
    y=alt.Y('inertia', scale=alt.Scale(domain=[12000,24000]))
)
alt_chart1.altair_chart(alt_intertia)

alt_silhouette = alt.Chart(kmeans_inertia[['k','silhouette_score']]).mark_line().encode(
    x='k:O',
    y=alt.Y('silhouette_score', scale=alt.Scale(domain=[.1,.2]))
)
alt_chart2.altair_chart(alt_silhouette)


st.markdown('---')
st.write('Search for artist to get top 5 songs. Clicking on a song checks our lookups first and if the song isnt there itll run a lookup against spotify API, bring audio features back.')  
st.write('next is to get cossim on the fly')
#### testing search bar idea
search_term = st.text_input('Search an artist', 'Adele')

#search_term = 'Adele'
search = requests.get(SPOTIFY_BASE_URL + 'search?q=artist:' + search_term + '&type=artist', headers=headers)
search = search.json()

for item in search['artists']['items'][0:1]:
    searchy = requests.get(SPOTIFY_BASE_URL + 'artists/' + item['id'] + '/top-tracks?market=US', headers=headers).json()
    st.write('Pick one of these top 5 songs for this artist.')
    for top_tracks in searchy['tracks'][0:5]:
        if st.button(top_tracks['name']):
            if audio_features_df['id'].str.contains(top_tracks['id']).any():
                final_df = audio_features_df[audio_features_df['id']==top_tracks['id']]
                st.dataframe(final_df)
            else:
                audio_feats = requests.get(SPOTIFY_BASE_URL + 'audio-features?ids=' + top_tracks['id'], headers=headers).json()
                audio_features_df = utils.get_audio_features(audio_feats)
                
                track_info = requests.get(SPOTIFY_BASE_URL + 'tracks?ids=' + top_tracks['id'], headers=headers).json()
                track_info_df = utils.get_track_info(track_info)

                final_df = audio_features_df.merge(track_info_df, how='inner', on='id')
                
                st.dataframe(final_df)

# once a button has been clicked, show the results on the page
try:
    st.markdown('---')
    st.write('Recommendations')
    compare, cossim_df, compare_df_sort = utils.create_cossim_df(final_df, playlist_audio_feature_rollup, global_lookup)

    col1, col2, col3,col4 = st.columns([4,3,3,3])
    col1.write(cossim_df['name'].iloc[0])
    col1.image(cossim_df['album_img'].iloc[0])
    col1.audio(cossim_df['preview_url'].iloc[0])

    #col2.markdown([compare_df_sort['name'].iloc[0]](compare_df_sort['link'].iloc[0]))
    col2.image(compare_df_sort['playlist_img'].iloc[0])
    col2.markdown("[link](" + compare_df_sort['link'].iloc[0] + ")")
    col2.write(compare_df_sort['sim'].iloc[0])
    #col2.markdown("[this is an image link]" + col2.image(compare_df_sort['playlist_img'].iloc[0]) + "(" + compare_df_sort['link'].iloc[0] + ")")
    col3.image(compare_df_sort['playlist_img'].iloc[1])
    col3.markdown("[link](" + compare_df_sort['link'].iloc[1] + ")")
    col3.write(compare_df_sort['sim'].iloc[1])
    col4.image(compare_df_sort['playlist_img'].iloc[2])
    col4.markdown("[link](" + compare_df_sort['link'].iloc[2] + ")")
    col4.write(compare_df_sort['sim'].iloc[2])
except:
    pass










# st.markdown('---')

# #st.pyplot(chart)

# scaler = pickle.load(open("model/scaler.pkl", "rb"))
# kmeans = pickle.load(open("model/kmeans.pkl", "rb"))

# X = audio_features_df.drop(columns=['id','duration_ms','update_dttm','time_signature','name','artist','album_img','preview_url'])
# scaled_data = pd.DataFrame(scaler.transform(X[0:1]))
# kmeans.predict(scaled_data)





# KMeans Clustering
# kmeans = KMeans(n_clusters=7)
# kmeans.fit(X_scaled)

# # checking cluster size
# clusters = kmeans.predict(X_scaled)
# pd.Series(clusters).value_counts().sort_index()
# audio_features_df_clustered = audio_features_df.copy()
# audio_features_df_clustered["cluster"] = clusters







# cossim_df['sim'] = cossim_df.apply(lambda x: cosine_similarity(compare0.reshape(1,-1), x.values.reshape(1,-1))[0][0], axis=1)
# cossim_df['id'] = cossim_df_y
# cossim_df = cossim_df.sort_values('sim', ascending=False)
# cossim_df = cossim_df[0:7]

# cossim_df = cossim_df.merge(audio_features_df[['id', 'name', 'artist','album_img','preview_url']], how='inner', on='id')
# cossim_df = cossim_df.drop_duplicates(subset=['name']).reset_index(drop=True)

# col1, col2, col3, col4, col5, col6 = st.columns([2,1,1,1,1,1])
# # for id in kmeans_df['id']:
# #     search = requests.get(SPOTIFY_BASE_URL + 'tracks/' + id , headers=headers)
# #     search = search.json()
# #     kmeans_df['img_url'][kmeans_df['id']==id] = (search['album']['images'][0]['url'])
# #     kmeans_df['prev_url'][kmeans_df['id']==id] = (search['preview_url'])
# #     kmeans_df['songname'][kmeans_df['id']==id] = (search['name'])

# col1.image(cossim_df['album_img'].iloc[0])
# col1.audio(cossim_df['preview_url'].iloc[0])
# col2.image(cossim_df['album_img'].iloc[1])
# col2.audio(cossim_df['preview_url'].iloc[1])
# col2.write(cossim_df['name'].iloc[1])
# col3.image(cossim_df['album_img'].iloc[2])
# col3.audio(cossim_df['preview_url'].iloc[2])
# col4.image(cossim_df['album_img'].iloc[3])
# col4.audio(cossim_df['preview_url'].iloc[3])
# col5.image(cossim_df['album_img'].iloc[4])
# col5.audio(cossim_df['preview_url'].iloc[4])
# col6.image(cossim_df['album_img'].iloc[5])
# if pd.isna(cossim_df['preview_url'][5]) == False:
#     col6.audio(cossim_df['preview_url'].iloc[5])

# st.dataframe(cossim_df)

# for item in search['artists']['items'][0:3]:
#     artist = item['name']
#     artist_id = item['id']
#     listy = []
#     if (st.button(artist)):
#         top_song = requests.get(SPOTIFY_BASE_URL + 'artists/' + artist_id +'/top-tracks?market=US', headers=headers).json()['tracks'][0]
#         audio_feats = requests.get(SPOTIFY_BASE_URL + 'audio-features/' + top_song['id'], headers=headers).json()
#         st.dataframe(pd.DataFrame(audio_feats, index=['id']))
        
        #st.write(top_song['name'])
        #st.dataframe(audio_feats)
        #print(audio_feats)
        
        
        #st.write('works')



#top_song_id = requests.get(SPOTIFY_BASE_URL + 'artists/4dpARuHxo51G3z768sgnrY/top-tracks?market=US', headers=headers).json()['tracks'][0]

#st.write(artist)

# search = requests.get(SPOTIFY_BASE_URL + 'search?q=artist:' + 'Beatles' + '&type=artist', headers=headers)
# search = search.json()

# artist_list = dict()
#artist_img_url = []

# results_len = len(search['artists']['items']) - 1

# if results_len >= 3:
#     for artists in search['artists']['items'][:3]:
#         #print(artists)
#         col1, col2 = st.columns(2)
#         col1.write(artists['name'])
#         col1.image(artists['images'][0]['url'])
#         track_search = requests.get(SPOTIFY_BASE_URL + 'artists/' + artists['id'] + '/top-tracks' + market, headers=headers).json()
#         for t in track_search['tracks'][:1]:
#             print(t['preview_url'])
#             if t['preview_url']:
#                 col2.write('Top Song: ' + t['name'])
#                 col2.audio(t['preview_url'])
#             else:
#                 col2.write('No audio preview available')

# else:
#     for artists in search['artists']['items'][:results_len]:
#         #print(artists)
#         col1, col2 = st.columns(2)
#         col1.write(artists['name'])
#         col1.image(artists['images'][0]['url'])
#         track_search = requests.get(SPOTIFY_BASE_URL + 'artists/' + artists['id'] + '/top-tracks' + market, headers=headers).json()
#         for t in track_search['tracks'][:1]:
#             #print(t['preview_url'])
#             if t['preview_url']:
#                 col2.write('Top Song: ' + t['name'])
#                 col2.audio(t['preview_url'])
#                 col2.write(t['preview_url'])
#         else:
#             col2.write('No audio preview available')

#col1.metric(label='Playlist appearances', value=int(df['# Playlist Appearances'][0]))
#col1.markdown('**' + df['Artist'][0] + " - " + df['Track Name'][0] + '**')
# col1.image(df['img_url'][0])
# if pd.isna(playlist_data_df[df['Track ID'][0]==playlist_data_df['track_id']]['track_preview_url'].iloc[0]) == False:
    # col1.audio(playlist_data_df[df['Track ID'][0]==playlist_data_df['track_id']]['track_preview_url'].iloc[0])



# search_term = st.text_input('Search an artist', 'Led Zeppelin')


# search = requests.get(SPOTIFY_BASE_URL + 'search?q=artist:' + search_term + '&type=artist', headers=headers)
# search = search.json()
# #search


# # search = requests.get(SPOTIFY_BASE_URL + 'search?q=artist:' + 'Beatles' + '&type=artist', headers=headers)
# # search = search.json()

# artist_list = dict()
# #artist_img_url = []

# results_len = len(search['artists']['items']) - 1

# if results_len >= 3:
#     for artists in search['artists']['items'][:3]:
#         #print(artists)
#         col1, col2 = st.columns(2)
#         col1.write(artists['name'])
#         col1.image(artists['images'][0]['url'])
#         track_search = requests.get(SPOTIFY_BASE_URL + 'artists/' + artists['id'] + '/top-tracks' + market, headers=headers).json()
#         for t in track_search['tracks'][:1]:
#             print(t['preview_url'])
#             if t['preview_url']:
#                 col2.write('Top Song: ' + t['name'])
#                 col2.audio(t['preview_url'])
#             else:
#                 col2.write('No audio preview available')

# else:
#     for artists in search['artists']['items'][:results_len]:
#         #print(artists)
#         col1, col2 = st.columns(2)
#         col1.write(artists['name'])
#         col1.image(artists['images'][0]['url'])
#         track_search = requests.get(SPOTIFY_BASE_URL + 'artists/' + artists['id'] + '/top-tracks' + market, headers=headers).json()
#         for t in track_search['tracks'][:1]:
#             #print(t['preview_url'])
#             if t['preview_url']:
#                 col2.write('Top Song: ' + t['name'])
#                 col2.audio(t['preview_url'])
#                 col2.write(t['preview_url'])
#             else:
#                 col2.write('No audio preview available')



# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import silhouette_score
# import seaborn as sns
# import matplotlib.patheffects as PathEffects

# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1.5,
#                 rc={"lines.linewidth": 2.5})
# RS = 123


# def fashion_scatter(x, colors):
#     # choose a color palette with seaborn.
#     num_classes = len(np.unique(colors))
#     palette = np.array(sns.color_palette("hls", num_classes))

#     # create a scatter plot.
#     f = plt.figure(figsize=(8, 8))
#     ax = plt.subplot(aspect='equal')
#     sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
#     plt.xlim(-25, 25)
#     plt.ylim(-25, 25)
#     ax.axis('off')
#     ax.axis('tight')

#     # add the labels for each digit corresponding to the label
#     txts = []

#     for i in range(num_classes):

#         # Position of each label at median of data points.

#         xtext, ytext = np.median(x[colors == i, :], axis=0)
#         txt = ax.text(xtext, ytext, str(i), fontsize=24)
#         txt.set_path_effects([
#             PathEffects.Stroke(linewidth=5, foreground="w"),
#             PathEffects.Normal()])
#         txts.append(txt)

#     return f, ax, sc, txts







# # kmeans = KMeans(n_clusters=3)
# # kmeans.fit(scaled_features)
# # res2['k_means']=kmeans.predict(res2)


# kmeans_df = audio_features_df.drop(columns=['update_dttm', 'time_signature'])
# kmeans_df.id = pd.Categorical(kmeans_df.id)
# kmeans_df['id'] = kmeans_df.id.cat.codes
# kmeans_df_y = kmeans_df.id
# kmeans_df = kmeans_df.drop(columns=['id'])

# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(kmeans_df)


# tsne = TSNE(n_components=2).fit_transform(scaled_features)









# tsne = pd.DataFrame(tsne)
# tsne['duration_ms'] = kmeans_df['duration_ms']

# plt.figure(figsize=(10,10))

# ax = sns.scatterplot(data=tsne, x=0, y=1, 
#                      size='duration_ms', sizes=(50,1000), 
#                      alpha=0.7)

# # display legend without `size` attribute
# h,labs = ax.get_legend_handles_labels()
# ax.legend(h[1:10], labs[1:10], loc='best', ncol=2)












# distortions = []
# K = range(1,20)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k)
#     kmeanModel.fit(scaled_features)
#     distortions.append(kmeanModel.inertia_)

# plt.figure(figsize=(16,8))
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()




# silhouette_coefficients = []

# for k in range(2, 20):
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(scaled_features)
#     score = silhouette_score(scaled_features, kmeans.labels_)
#     silhouette_coefficients.append(score)

# plt.style.use("fivethirtyeight")
# plt.plot(range(2, 20), silhouette_coefficients)
# plt.xticks(range(2, 20))
# plt.xlabel("Number of Clusters")
# plt.ylabel("Silhouette Coefficient")
# plt.show()




# silhouette_coefficients = []
# # Notice you start at 2 clusters for silhouette coefficient
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#     kmeans.fit(scaled_features)
#     score = silhouette_score(scaled_features, kmeans.labels_)
#     silhouette_coefficients.append(score)












# kmeanModel = KMeans(n_clusters=3)
# kmeanModel.fit(res2)
# res2['k_means']=kmeanModel.predict(res2)

# fig, axes = plt.subplots(1, 2, figsize=(16,8))
# #axes[0].scatter(res2[0], res2[1], c=res2['k_means'])
# axes[1].scatter(res2[0], res2[1], c=res2['k_means'], cmap=plt.cm.Set1)
# #axes[0].set_title('Actual', fontsize=18)
# axes[1].set_title('K_Means', fontsize=18)

################################### 
# Pandas Profiling and Streamlit
###################################

#profile = pp.ProfileReport(playlist_audio_feature_rollup,
    #configuration_file="pandas_profiling_minimal.yml" 
    # variables={
    #     "descriptions": {
    #         "danceability":"Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.",
    #         "energy":"Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.",
    #         "key":"The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on",
    #         "loudness":"The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.",
    #         "mode":"Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.",
    #         "speechiness":"Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.",
    #         "acousticness":"A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.",
    #         "instrumentalness":"Predicts whether a track contains no vocals. ""Ooh"" and ""aah"" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly ""vocal"". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.",
    #         "liveness":"Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.",
    #         "valence":"A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).",
    #         "tempo":"The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.",
    #         "duration_ms":"The duration of the track in milliseconds.",
    #         "time_signature":"An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure)."

    #     }
    # }

#)
#profile.to_notebook_iframe()
#st_profile_report(profile)



#########
# density plots matplotlib
#########
# import numpy as np
# import pandas as pd
# import altair as alt
# import seaborn as sns
# import umap.umap_ as umap
# import plotly.express as px
# import matplotlib.pyplot as plt



# feature_names = ['danceability','energy','key','loudness','mode','speechiness','acousticness',
#                 'instrumentalness','liveness','valence','tempo', 'duration_ms', 'country']





# fig = plt.figure(figsize=(20, 15))
# for idx,feat in enumerate(feature_names[:-1]):
#     ax = fig.add_subplot(5,3,idx+1)
#     df_feat.groupby('country')[feat].plot(kind='density', legend=False)
#     plt.title('Density plot for {}'.format(feat))

# plt.tight_layout()
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=10)
# plt.savefig("mypgrah.png")


