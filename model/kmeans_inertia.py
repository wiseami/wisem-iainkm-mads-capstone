import utils
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cluster import KMeans
import pandas as pd
import pickle
from os.path import exists

# load necessary data using function
file_path, top_pl_df, audio_features_df, playlist_data_df, global_lookup, pl_w_audio_feats_df, kmeans_inertia = utils.load_data()

X = audio_features_df.drop(columns=['id','duration_ms','update_dttm','time_signature','name','artist','album_img','preview_url'])
scaler = StandardScaler().fit(X)
data_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(data_scaled)
# dump scaler to pickle
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

k_scores = utils.kmeans_k_tuning(X_scaled, 2, 16)
k_scores.to_csv('model/kmeans_inertia.csv', index=False)

kmeans = KMeans(n_clusters=6) # 6 clusters was best as of 11/22/21
kmeans.fit(X_scaled)
# dump kmeans to pickle
pickle.dump(kmeans, open("model/kmeans.pkl", "wb"))

# checking cluster size
clusters = kmeans.predict(X_scaled)
pd.Series(clusters).value_counts().sort_index()
audio_features_df_clustered = audio_features_df.copy()
audio_features_df_clustered["cluster"] = clusters

