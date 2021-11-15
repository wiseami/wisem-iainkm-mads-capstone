import pandas as pd
# import pandas_profiling as pp
# from pandas_profiling import ProfileReport
import streamlit as st
# from streamlit_pandas_profiling import st_profile_report
import requests
import tqdm
import altair as alt
import os
import sys
import json
import numpy as np

### Spotify info
with open('credentials.json') as creds:
    credentials = json.load(creds)

AUTH_URL = 'https://accounts.spotify.com/api/token'

auth_response = requests.post(AUTH_URL, {
    'grant_type': 'client_credentials',
    'client_id': credentials['CLIENT_ID'],
    'client_secret': credentials['CLIENT_SECRET'],
})

auth_response_data = auth_response.json()

access_token = auth_response_data['access_token']

headers = {
    'Authorization': 'Bearer {token}'.format(token=access_token)
}

# only for testing purposes. NEed to remove this later
market = '?market=US'

# base URL of all Spotify API endpoints
BASE_URL = 'https://api.spotify.com/v1/'


### Specify where you're running - mostly in place for working locally vs testing streamlit cloud
if sys.platform == 'win32':
    file_path = os.path.dirname(os.path.abspath(__file__)) + '\\'
    top_pl_df = pd.read_csv(file_path + 'lookups\\global_top_daily_playlists.csv')
    audio_features_df = pd.read_csv(file_path + 'lookups\\track_audio_features.csv')
    playlist_data_df = pd.read_csv(file_path + '\\playlist_data\\2021-11-13.csv')
else:
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    top_pl_df = pd.read_csv(file_path + 'lookups/global_top_daily_playlists.csv')
    audio_features_df = pd.read_csv(file_path + 'lookups/track_audio_features.csv')
    playlist_data_df = pd.read_csv(file_path + 'playlist_data/2021-11-13.csv')

### Join some of the lookups together and drop unneeded columns
merged = playlist_data_df.merge(audio_features_df, how='inner', left_on='track_id', right_on='id')
merged = merged.drop(columns=['market','capture_dttm','track_preview_url','track_duration', 'id', 'track_added_date', 'track_popularity', 'track_number','time_signature', 'track_artist','track_name','track_id'])

grouped = merged.groupby(by=['country'], as_index=False)
res = grouped.agg(['sum', 'count'])
res.columns = list(map('_'.join, res.columns.values))
res = res.reset_index()

### Create Spotify audio features normalized for playlist length
res = res.drop(columns=['danceability_count', 'energy_count', 'key_count', 'loudness_count', 'mode_count', 'speechiness_count', 'acousticness_count', 'instrumentalness_count', 'liveness_count', 'valence_count', 'tempo_count'])
res = res.rename(columns = {'duration_ms_count':'track_count'})
res['duration_m'] = res['duration_ms_sum'] / 1000 / 60
res['danceability'] = res['danceability_sum'] / res['duration_m']
res['energy'] = res['energy_sum'] / res['duration_m']
res['key'] = res['key_sum'] / res['duration_m']
res['mode'] = res['mode_sum'] / res['duration_m']
res['loudness'] = res['loudness_sum'] / res['duration_m']
res['speechiness'] = res['speechiness_sum'] / res['duration_m']
res['acousticness'] = res['acousticness_sum'] / res['duration_m']
res['instrumentalness'] = res['instrumentalness_sum'] / res['duration_m']
res['liveness'] = res['liveness_sum'] / res['duration_m']
res['valence'] = res['valence_sum'] / res['duration_m']
res['tempo'] = res['tempo_sum'] / res['duration_m']

res = res.drop(columns=['danceability_sum', 'energy_sum', 'key_sum', 'loudness_sum', 'mode_sum', 'speechiness_sum', 'acousticness_sum', 'instrumentalness_sum', 'liveness_sum', 'valence_sum', 'tempo_sum', 'duration_ms_sum', 'update_dttm_sum', 'update_dttm_count', 'track_count','duration_m'])

### Start building out Streamlit assets
st.set_page_config(layout="wide")
st.title('Spotify Streamlit')
st.write('this is a test')
st.markdown('---')
st.header('Top 3 Songs Based on number of playlist appearances')
st.write("While the first day of scraping playlists came back with 3,450 total songs, only about half of those were unique. Because of that, we have tons of tracks that show up on multiple playlists. We're looking at a total of 69 daily playlists - 68 country-specific and 1 global - and these songs below show up on multiple different country playlists.")

# st.markdown('---')
df = pd.DataFrame(playlist_data_df.groupby(['track_name', 'track_artist','track_id'])['country'].count().sort_values(ascending=False).reset_index()).head()
df.columns = ['Track Name', 'Artist', 'Track ID', '# Playlist Appearances']
df['img_url'] = np.nan

col1, col2, col3 = st.columns(3)
for id in df['Track ID']:
    search = requests.get(BASE_URL + 'tracks/' + id , headers=headers)
    search = search.json()
    df['img_url'][df['Track ID']==id] = (search['album']['images'][0]['url'])

col1.markdown('**' + df['Artist'][0] + " - " + df['Track Name'][0] + '**')
col1.metric(label='Playlist appearances', value=int(df['# Playlist Appearances'][0]))
col1.image(df['img_url'][0])
if pd.isna(playlist_data_df[df['Track ID'][0]==playlist_data_df['track_id']]['track_preview_url'].iloc[0]) == False:
    col1.audio(playlist_data_df[df['Track ID'][0]==playlist_data_df['track_id']]['track_preview_url'].iloc[0])

col2.markdown('**' + df['Artist'][1] + " - " + df['Track Name'][1] + '**')
col2.metric(label='Playlist appearances', value=int(df['# Playlist Appearances'][1]))
col2.image(df['img_url'][1])

if pd.isna(playlist_data_df[df['Track ID'][1]==playlist_data_df['track_id']]['track_preview_url'].iloc[1]) == False:
    col2.audio(playlist_data_df[df['Track ID'][1]==playlist_data_df['track_id']]['track_preview_url'].iloc[1])

col3.markdown('**' + df['Artist'][2] + " - " + df['Track Name'][2] + '**')
col3.metric(label='Playlist appearances', value=int(df['# Playlist Appearances'][2]))
col3.image(df['img_url'][2])

if pd.isna(playlist_data_df[df['Track ID'][2]==playlist_data_df['track_id']]['track_preview_url'].iloc[2]) == False:
    col3.audio(playlist_data_df[df['Track ID'][2]==playlist_data_df['track_id']]['track_preview_url'].iloc[2])

st.markdown('---')

feature_names_to_show = ['danceability','energy','key','loudness','mode','speechiness','acousticness',
                'instrumentalness','liveness','valence','tempo']

st.write("Let's take a look at the audio features computed and captured by Spotify for these three songs.")
st.table(audio_features_df[0:3][feature_names_to_show])
#st_profile_report(profile)

feature_names = ['danceability','energy','key','loudness','mode','speechiness','acousticness',
                'instrumentalness','liveness','valence','tempo', 'duration_ms', 'country']

df_feat = merged[feature_names]

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

st.write("Knowing we have 69 playlists makes these visuals not-so-easy to consume, but it seemed worth showing the density plots for a couple of audio features across all countries where each line is a country. Definitions on left directly from Spotify's [API documentation.](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features)")

col1, col2 = st.columns([1,2])
col1.markdown('**Danceability** - Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.')
col2.altair_chart(charts[0], use_container_width=True)

col1, col2 = st.columns([1,2])
col1.markdown('**Energy** - Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.')
col2.altair_chart(charts[1], use_container_width=True)

col1, col2 = st.columns([1,2])
col1.markdown('**Loudness** - The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.')
col2.altair_chart(charts[3], use_container_width=True)


st.markdown('---')
st.header('Correlations')

res_correlation = res.corr().stack().reset_index().rename(columns={0: 'correlation', 'level_0': 'variable 1', 'level_1': 'variable 2'})
res_correlation['correlation_label'] = res_correlation['correlation'].map('{:.2f}'.format)

base = alt.Chart(res_correlation).encode(
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

col1, col2 = st.columns(2)
col1.write("Now, let's take country out of the equation and have a closer look at the different individual audio features and how they correlate with one another. In this case, we created an aggregate country-playlist value of each of the individual song audio features and normalized for total duration of the playlist.")
col2.altair_chart(cor_plot + text, use_container_width=True)








# search_term = st.text_input('Search an artist', 'Led Zeppelin')


# search = requests.get(BASE_URL + 'search?q=artist:' + search_term + '&type=artist', headers=headers)
# search = search.json()
# #search


# # search = requests.get(BASE_URL + 'search?q=artist:' + 'Beatles' + '&type=artist', headers=headers)
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
#         track_search = requests.get(BASE_URL + 'artists/' + artists['id'] + '/top-tracks' + market, headers=headers).json()
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
#         track_search = requests.get(BASE_URL + 'artists/' + artists['id'] + '/top-tracks' + market, headers=headers).json()
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
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import silhouette_score




# # kmeans = KMeans(n_clusters=3)
# # kmeans.fit(scaled_features)
# # res2['k_means']=kmeans.predict(res2)


# kmeans_df = merged.drop(columns=['update_dttm'])
# kmeans_df.country = pd.Categorical(kmeans_df.country)
# #kmeans_df['country_code'] = kmeans_df.country.cat.codes
# kmeans_df_y = kmeans_df.country.cat.codes
# kmeans_df = kmeans_df.drop(columns=['country'])

# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(kmeans_df)




# distortions = []
# K = range(1,10)
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

#profile = pp.ProfileReport(res,
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

# df_feat = merged[feature_names]




# fig = plt.figure(figsize=(20, 15))
# for idx,feat in enumerate(feature_names[:-1]):
#     ax = fig.add_subplot(5,3,idx+1)
#     df_feat.groupby('country')[feat].plot(kind='density', legend=False)
#     plt.title('Density plot for {}'.format(feat))

# plt.tight_layout()
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=10)
# plt.savefig("mypgrah.png")


