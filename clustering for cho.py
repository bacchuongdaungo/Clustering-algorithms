#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#load dataset cho.txt
cho_data = pd.read_csv("E:\data mining\cho.txt", sep='\t', header=None)


# In[3]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score
import numpy as np


# In[4]:


# Extracting features and true labels
features_cho = cho_data.iloc[:, 2:]  # gene expression values
true_labels_cho = cho_data.iloc[:, 1]  # ground truth clusters


# In[5]:


# Standardizing the features
scaler = StandardScaler()
features_cho_scaled = scaler.fit_transform(features_cho)


# In[9]:


from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
import matplotlib.pyplot as plt

# Applying K-means clustering
silhouette_scores_dropped = []
K_range = range(2, 11)

for K in K_range:
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans_labels = kmeans.fit_predict(features_cho_scaled)
    score = silhouette_score(features_cho_scaled, kmeans_labels)
    silhouette_scores_dropped.append(score)

# Find the new optimal number of clusters
optimal_K_dropped = K_range[silhouette_scores_dropped.index(max(silhouette_scores_dropped))]
optimal_score_dropped = max(silhouette_scores_dropped)

# Clustering with the new optimal number of clusters
kmeans_optimal_dropped = KMeans(n_clusters=optimal_K_dropped, random_state=42)
kmeans_optimal_labels_dropped = kmeans_optimal_dropped.fit_predict(features_cho_scaled)

# The optimal number of clustering for K-Means is 3 with the silhouette score of 0.24623786590509342

#### #### #### ####
# Applying DBSCAN clustering
# A range of epsilon values to try. We'll start with a range that's common for scaled data.
eps_range = np.arange(0.1, 2.0, 0.1)

# The minimum number of samples required for a core point. Starting with a common choice.
min_samples_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Placeholder for the best silhouette score found
best_silhouette_score = -1

# Placeholder for the best parameters
best_eps = None
best_min_samples = None

for eps in eps_range:
    for min_samples in min_samples_range:
        # Apply DBSCAN with the current parameters
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(features_cho_scaled)

        # We only calculate silhouette score for results with more than 1 cluster and less than n-1 clusters
        if len(set(dbscan_labels)) > 1 and len(set(dbscan_labels)) < len(features_cho_scaled) - 1:
            score = silhouette_score(features_cho_scaled, dbscan_labels)
            if score > best_silhouette_score:
                best_silhouette_score = score
                best_eps = eps
                best_min_samples = min_samples

#best_eps: 1.9000000000000001, best_min_samples: 7, best_silhouette_score: 0.027112825529119606
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
dbscan_labels = dbscan.fit_predict(features_cho_scaled)

#### #### #### ####
# Applying Spectral Clustering
silhouette_scores_dropped_spectral = []
K_range = range(2, 11)

for K in K_range:
    spectral = SpectralClustering(n_clusters=K, random_state=42, affinity='nearest_neighbors')
    spectral_labels = spectral.fit_predict(features_cho_scaled)
    score_spectral = silhouette_score(features_cho_scaled, spectral_labels)
    silhouette_scores_dropped_spectral.append(score_spectral)

# Find the new optimal number of clusters
optimal_K_dropped_spectral = K_range[silhouette_scores_dropped_spectral.index(max(silhouette_scores_dropped_spectral))]
optimal_score_dropped_spectral = max(silhouette_scores_dropped_spectral)

# Clustering with the new optimal number of clusters
spectral = SpectralClustering(n_clusters=optimal_K_dropped_spectral, random_state=42, affinity='nearest_neighbors')
spectral_labels = spectral.fit_predict(features_cho_scaled)

# Calculate silhouette score for Spectral Clustering
spectral_silhouette_score = silhouette_score(features_cho_scaled, spectral_labels)


# In[10]:


from sklearn.metrics import adjusted_rand_score

# Calculate Adjusted Rand Index (ARI) for each clustering result
ari_kmeans = adjusted_rand_score(true_labels_cho.values, kmeans_optimal_labels_dropped)
ari_dbscan = adjusted_rand_score(true_labels_cho.values, dbscan_labels)
ari_spectral = adjusted_rand_score(true_labels_cho.values, spectral_labels)

ari_scores = {
    "KMeans ARI": ari_kmeans,
    "DBSCAN ARI": ari_dbscan,
    "Spectral ARI": ari_spectral
}

ari_scores


# In[11]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Apply PCA and reduce the data to two dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(features_cho_scaled)

# Plotting function
def plot_clusters(X_pca, labels, algorithm_name):
    plt.figure(figsize=(16, 8))
    plt.title(f"Clusters from {algorithm_name} by PCA")
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    legend = plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.gca().add_artist(legend)
    plt.show()

# Visualize the clustering results of K-means, DBSCAN, and Spectral clustering algorithms by PCA
plot_clusters(X_pca, kmeans_optimal_labels_dropped, 'K-Means')
plot_clusters(X_pca, dbscan_labels, 'DBSCAN')
plot_clusters(X_pca, spectral_labels, 'Spectral Clustering')


# In[ ]:





# In[ ]:




