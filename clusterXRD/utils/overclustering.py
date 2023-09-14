import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances



def calculate_intra_cluster_distances(pca_transformed_scaled_features,non_amorphous_features,cluster_number):
    '''
    Calculate pairwise histogram feature distances for all histograms in a given cluster.

    Args:
        pca_transformed_scaled_features (numpy array): array of PCA scaled feature values
        non_amorphous_features (pandas DataFrame): dataframe with cluster labels from clustering
        cluster_number (int): the cluster number of interest, whose histogram distances are to be calculated
    
    Returns:
        numpy array: 1D array of distances between histogram pairs within the cluster
    '''
    # for the origin cluster, calculate distances from all points to the centroid
    cluster_features = pca_transformed_scaled_features[non_amorphous_features['Cluster labels'] == cluster_number] #scaled, PCA
    point_point_distances = euclidean_distances(cluster_features).flatten()
    point_point_distances = point_point_distances[np.nonzero(point_point_distances)]
    return point_point_distances
    

def calculate_inter_cluster_centroid_distances(kmeans,cluster_number):
    '''
    Calculate pairwise distances of a specified cluster centroid to all other centroids in feature space.

    Args:
        kmeans (class sklearn.cluster._kmeans.KMeans): KMeans object after clustering histograms
        cluster_number (int): the cluster number of interest, whose distances to all other centroids are to be calculated
    
    Returns:
        numpy array: 1D array of distances to all other centroids
    '''
    # get cluster centroids
    centroids = kmeans.cluster_centers_ # scaled, PCA

    # calculate pairwise distances
    pairwise_centroid_distances = euclidean_distances(centroids) # scaled, PCA

    return pairwise_centroid_distances[cluster_number,:]