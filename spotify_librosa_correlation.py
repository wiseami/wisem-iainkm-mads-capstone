import utils
import altair as alt
import pandas as pd

file_path, audio_features_df, playlist_data_df, global_pl_lookup, pl_w_audio_feats_df, basic_kmeans_inertia, adv_kmeans_inertia = utils.load_data()

audio_feat_corr = audio_features_df.drop(columns=['time_signature','update_dttm','name','artist','album_img','preview_url', 'duration_ms'])
audio_feat_corr = audio_feat_corr.corr().stack().reset_index().rename(columns={0: 'correlation', 'level_0': 'variable 1', 'level_1': 'variable 2'})

audio_feat_corr_ct1 = audio_feat_corr.copy()[(audio_feat_corr['variable 1']!='pl_count') & (audio_feat_corr['variable 1']!='popularity') & (audio_feat_corr['variable 2']!='pl_count') & (audio_feat_corr['variable 2']!='popularity')]
audio_feat_corr_ct1['correlation_label'] = audio_feat_corr_ct1['correlation'].map('{:.2f}'.format)

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

cor_plot1 + text


# Country / Audio Feature Correlation
countries = ['argentina',
       'australia', 'austria', 'belgium', 'bolivia', 'brazil', 'bulgaria',
       'canada', 'chile', 'colombia', 'costa_rica', 'czech_republic',
       'denmark', 'dominican_republic', 'ecuador', 'egypt', 'el_salvador',
       'estonia', 'finland', 'france', 'germany', 'greece',
       'guatemala', 'honduras', 'hong_kong', 'hungary', 'iceland', 'india',
       'indonesia', 'ireland', 'israel', 'italy', 'japan', 'latvia',
       'lithuania', 'luxembourg', 'malaysia', 'mexico', 'morocco',
       'netherlands', 'new_zealand', 'nicaragua', 'norway', 'panama',
       'paraguay', 'peru', 'philippines', 'poland', 'portugal', 'romania',
       'russia', 'saudi_arabia', 'singapore', 'slovakia', 'south_africa',
       'south_korea', 'spain', 'sweden', 'switzerland', 'taiwan', 'thailand',
       'turkey', 'uae', 'ukraine', 'united_kingdom', 'united_states',
       'uruguay', 'vietnam']

audio_feat_corr = pl_w_audio_feats_df.drop(columns=['duration_ms','basic_kmeans_cluster','adv_kmeans_cluster','pl_count','popularity'])
audio_feat_corr = audio_feat_corr.join(pd.get_dummies(audio_feat_corr['country']))
audio_feat_corr = audio_feat_corr.corr().stack().reset_index().rename(columns={0: 'correlation', 'level_0': 'variable 1', 'level_1': 'variable 2'})

audio_feat_corr_ct1 = audio_feat_corr.copy()[(audio_feat_corr['variable 1'].isin(countries)) & (~audio_feat_corr['variable 2'].isin(countries))]
audio_feat_corr_ct1['correlation_label'] = audio_feat_corr_ct1['correlation'].map('{:.2f}'.format)
        
# Correlation matrix
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

cor_plot1 + text