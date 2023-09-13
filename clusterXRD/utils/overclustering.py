import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances



def calculate_intra_cluster_distances(pca_transformed_scaled_features,non_amorphous_features,cluster_number,kmeans):
    # for the origin cluster, calculate distances from all points to the centroid
    cluster_features = pca_transformed_scaled_features[non_amorphous_features['Cluster labels'] == cluster_number] #scaled, PCA
    point_point_distances = euclidean_distances(cluster_features).flatten()
    point_point_distances = point_point_distances[np.nonzero(point_point_distances)]
    return point_point_distances
    

def calculate_inter_cluster_centroid_distances(kmeans,cluster_number):
    # get cluster centroids
    centroids = kmeans.cluster_centers_ # scaled, PCA

    # calculate pairwise distances
    pairwise_centroid_distances = euclidean_distances(centroids) # scaled, PCA

    return pairwise_centroid_distances[cluster_number,:]