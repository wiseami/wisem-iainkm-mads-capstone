import pandas as pd
# import pandas_profiling as pp
# from pandas_profiling import ProfileReport
import streamlit as st
# from streamlit_pandas_profiling import st_profile_report
import requests
import tqdm
import altair as alt

top_pl_df = pd.read_csv("C:\\Users\\Mike\\Documents\\GitHub\\coursera\\wisem-iainkm-mads-capstone\\lookups\\global_top_daily_playlists.csv")
audio_features_df = pd.read_csv("C:\\Users\\Mike\\Documents\\GitHub\\coursera\\wisem-iainkm-mads-capstone\\lookups\\track_audio_features.csv")
playlist_data_df = pd.read_csv("C:\\Users\\Mike\\Documents\\GitHub\\coursera\\wisem-iainkm-mads-capstone\\playlist_data\\2021-11-13.csv")

merged = playlist_data_df.merge(audio_features_df, how='inner', left_on='track_id', right_on='id')
#merged[merged['country']=='argentina']
merged = merged.drop(columns=['market','capture_dttm','track_preview_url','track_duration', 'id', 'track_added_date', 'track_popularity', 'track_number','time_signature', 'track_artist','track_name','track_id'])

grouped = merged.groupby(by=['country'], as_index=False)
res = grouped.agg(['sum', 'count'])
res.columns = list(map('_'.join, res.columns.values))
res = res.reset_index()

res = res.drop(columns=['danceability_count', 'energy_count', 'key_count', 'loudness_count', 'mode_count', 'speechiness_count', 'acousticness_count', 'instrumentalness_count', 'liveness_count', 'valence_count', 'tempo_count'])
res = res.rename(columns = {'duration_ms_count':'track_count'})
res['duration_m'] = res['duration_ms_sum'] / 1000 / 60
#res['duration_m_mean'] = res['duration_ms_mean'] / 1000 / 60
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
#res = res.drop(columns=['danceability_mean','energy_mean','key_mean','loudness_mean','mode_mean','speechiness_mean','acousticness_mean','instrumentalness_mean', 'liveness_mean','valence_mean','tempo_mean', 'duration_ms_mean'])
#res

#playlist_data_df[playlist_data_df['country']=='argentina']

################################### 
# Pandas Profiling and Streamlit
###################################

profile = pp.ProfileReport(res,
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

)
#profile.to_notebook_iframe()

st.set_page_config(layout="wide")
st.title('Spotify Streamlit')
st.write('this is a test')
st.markdown('---')
st.header('Top 5 Songs Based on number of playlist appearances')
st.write('While the first day of scraping playlists came back with 3,450 total songs, only about half of those were unique. Because of that, we have tons of tracks that show up on multiple playlists.')
df = pd.DataFrame(playlist_data_df.groupby(['track_name', 'track_artist','track_id'])['country'].count().sort_values(ascending=False).reset_index()).head()
df.columns = ['Track Name', 'Artist', 'Track ID', '# Playlist Appearances']
st.table(df)
#st.dataframe(res)

# st.metric(
#     'Song with most playlist appearances',
#     str(df['Track Name'][0])
#     #int(df['# Playlist Appearances'][0])
# )

st.write(
    "Wow, I didn't realize **" 
    + df['Artist'][0] 
    + "** was so popular across the world! I wonder if their hit song **" 
    + df['Track Name'][0] 
    + "** just came out? Or maybe it has to do with the nature of the music itself. If a 30 second preview is available, you can listen below."
)

if playlist_data_df[df['Track ID'][0]==playlist_data_df['track_id']]['track_preview_url'].iloc[0]:
    st.audio(playlist_data_df[df['Track ID'][0]==playlist_data_df['track_id']]['track_preview_url'].iloc[0])

feature_names_to_show = ['danceability','energy','key','loudness','mode','speechiness','acousticness',
                'instrumentalness','liveness','valence','tempo']

st.write("Let's take a look at the audio features computed and captured by Spotify for that song.")
st.table(audio_features_df[audio_features_df['id'] == df['Track ID'][0]][feature_names_to_show])
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


#st_profile_report(profile)












### this downloads mp3s from the preview
# glob = playlist_data_df.iloc[0:50]
# glob = glob[glob['track_preview_url'].notnull()].reset_index(drop=True)


# for idx, url in enumerate(glob['track_preview_url']):
#     doc = requests.get(url)
#     doc.content
#     with open('audio/{}_{}.mp3'.format(idx, glob.loc[idx]['track_name'].replace('/', '_')), 'wb') as f:
#         f.write(doc.content)











#########
# density plots
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


