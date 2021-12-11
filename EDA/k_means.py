import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import utils

full_audio_feats = pd.read_csv('lookups/full_audio_features.csv')

X = full_audio_feats.drop(columns=['track_id','duration_ms','update_dttm','time_signature','name','artist','album_img','preview_url', 'popularity'])
if 'basic_kmeans_cluster' in X.columns:
    X.drop(columns=['basic_kmeans_cluster'], inplace=True)
if 'advanced_kmeans_cluster' in X.columns:
    X.drop(columns=['advanced_kmeans_cluster'], inplace=True)

## Basic KMeans Clustering - Using only Spotify features
X_small = X.drop(columns=['chroma', 'chroma_cens', 'mff', 'spectral_centroid', 'spectral_bandwidth', 'spectral_contrast', 'spectral_flatness', 'Spectral_Rolloff', 'poly_features', 'tonnetz', 'ZCR', 'onset_strength', 'pitch', 'magnitude', 'tempo'])
basic_scaler = StandardScaler().fit(X_small)
data_scaled = basic_scaler.transform(X_small)
X_scaled = pd.DataFrame(data_scaled)

basic_kmeans = KMeans(n_clusters=10, n_init=30, max_iter=500)
basic_kmeans.fit(X_scaled)

pca = PCA(n_components=.95)
df = pca.fit_transform(X_scaled)

label = basic_kmeans.fit_predict(df)
u_labels = np.unique(label)

centroids = basic_kmeans.cluster_centers_

for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1], label = i)
plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()

basic_k_scores = utils.kmeans_k_tuning(X_scaled, 2, 16)

### Loop for plotting PCA-reduced data
for i in np.arange(1, 13, 1):
    basic_kmeans = KMeans(n_clusters=i, n_init=30, max_iter=500)
    basic_kmeans.fit(X_scaled)

    pca = PCA(2)
    df = pca.fit_transform(X_scaled)

    label = basic_kmeans.fit_predict(df)
    u_labels = np.unique(label)

    centroids = basic_kmeans.cluster_centers_

    for i in u_labels:
        plt.scatter(df[label == i , 0] , df[label == i , 1], label = i)
    plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
    plt.legend()
    plt.show()

## Advanced KMeans Clustering - Using Spotify and Librosa features
X_adv = X[X['spectral_contrast'].notna()]
adv_scaler = StandardScaler().fit(X_adv)
adv_data_scaled = adv_scaler.transform(X_adv)
X_adv_scaled = pd.DataFrame(adv_data_scaled)

adv_kmeans = KMeans(n_clusters=11, n_init=30, max_iter=500)
adv_kmeans.fit(X_adv_scaled)

pca = PCA(2)
df = pca.fit_transform(X_adv_scaled)

label = adv_kmeans.fit_predict(df)
u_labels = np.unique(label)

centroids = adv_kmeans.cluster_centers_

for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1], label = i)
plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()

adv_k_scores = utils.kmeans_k_tuning(X_adv_scaled, 2, 16)

### Loop for plotting PCA-reduced data
X_adv = X[X['spectral_contrast'].notna()]
adv_scaler = StandardScaler().fit(X_adv)
adv_data_scaled = adv_scaler.transform(X_adv)
X_adv_scaled = pd.DataFrame(adv_data_scaled)

for i in np.arange(1, 20, 1):
    adv_kmeans = KMeans(n_clusters=i, n_init=30, max_iter=500)
    adv_kmeans.fit(X_adv_scaled)

    pca = PCA(2)
    df = pca.fit_transform(X_adv_scaled)

    label = adv_kmeans.fit_predict(df)
    u_labels = np.unique(label)

    centroids = adv_kmeans.cluster_centers_

    for i in u_labels:
        plt.scatter(df[label == i , 0] , df[label == i , 1], label = i)
    plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
    plt.legend()
    plt.show()