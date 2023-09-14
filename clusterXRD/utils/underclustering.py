from sklearn.metrics.pairwise import euclidean_distances

def get_point_point_distances(comparison_pairs,pca_scaled_cluster_features):
    '''
    Calculate pairwise histogram feature distances.

    Args:
        comparison_pairs (numpy array): indices of histograms whose feature distances need to be caluclated
        pca_scaled_cluster_features (numpy array): the PCA scaled feature values
    
    Returns:
        numpy array: 1D array of feature distances for each histogram pair specified in comparison pairs
    '''
    
    pair_pca_scaled_feature_distances = []

    for index_pair in comparison_pairs:
        index_1, index_2 = index_pair[0], index_pair[1]

        if index_1 >= index_2: continue # only look at upper diagonal index pairs to avoid duplicates

        histogram_1_features = pca_scaled_cluster_features[index_1]
        histogram_2_features = pca_scaled_cluster_features[index_2]

        distance = euclidean_distances([histogram_1_features],[histogram_2_features])[0][0]

        pair_pca_scaled_feature_distances += [distance]
    
    return pair_pca_scaled_feature_distances