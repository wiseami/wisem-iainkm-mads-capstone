import utils
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cluster import KMeans
import pandas as pd
import pickle

# load necessary data using function
file_path, top_pl_df, audio_features_df, playlist_data_df, global_lookup, pl_w_audio_feats_df, kmeans_inertia = utils.load_data()

X_scaled = utils.kmeans_prepro_X_scaled(audio_features_df)
k_scores = utils.kmeans_k_tuning(X_scaled, 2, 16)

k_scores.to_csv('C:\\Users\\Mike\\Documents\\GitHub\\coursera\\wisem-iainkm-mads-capstone\\model\\kmeans_inertia.csv', index=False)




kmeans = KMeans(n_clusters=7, random_state=99)
kmeans.fit(X_scaled)

kmeans = pickle.load(open("model/kmeans.pkl", "rb"))
# checking cluster size
clusters = kmeans.predict(X_scaled)
pd.Series(clusters).value_counts().sort_index()
audio_features_df_clustered = audio_features_df.copy()
audio_features_df_clustered["cluster"] = clusters

pickle.dump(kmeans, open("kmeans.pkl", "wb"))





X = audio_features_df.drop(columns=['id','duration_ms','update_dttm','time_signature','name','artist','album_img','preview_url'])
scaler = StandardScaler().fit(X)
data_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(data_scaled)

pickle.dump(scaler, open("scaler.pkl", "wb"))