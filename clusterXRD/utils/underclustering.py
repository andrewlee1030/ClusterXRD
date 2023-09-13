import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import percentileofscore


def get_pattern_pairs_and_distances(similarity_matrix,similarity_min,similarity_max,pca_scaled_cluster_features):

    pattern_names = similarity_matrix.columns
    similarity_matrix_numerical = similarity_matrix.to_numpy()


    pairs_to_compare = np.argwhere((similarity_matrix_numerical > similarity_min) & (similarity_matrix_numerical < similarity_max)) # all pair indices of patterns that have low similarity (as defined by the low_similarity_threshold)
    if len(pairs_to_compare) == 0: return [], []
    index_to_other_points_distance_percentiles = dict() # store calculated distances so we don't need to calculate them repeatedly
    index_to_other_points_distances = dict() # store calculated distances so we don't need to calculate them repeatedly

    pairs_to_compare_distances = []
    pattern_1_names = []
    pattern_2_names = []

    for index_pair in pairs_to_compare:
        index_1, index_2 = index_pair[0], index_pair[1]

        if index_1 >= index_2: continue # only look at upper diagonal index pairs to avoid duplicates

        pattern_1_name, pattern_2_name = pattern_names[index_1], pattern_names[index_2]

        pattern_1_names += [pattern_1_name]
        pattern_2_names += [pattern_2_name]


        pattern_1_features = pca_scaled_cluster_features[index_1]
        pattern_2_features = pca_scaled_cluster_features[index_2]

        # calculate distance percentiles from index 1 to 2

        try:
            distance_percentiles = index_to_other_points_distance_percentiles[index_1]
            distances = index_to_other_points_distances[index_1]
        except:
            distances = euclidean_distances(pca_scaled_cluster_features,pattern_1_features.reshape(1,len(pattern_1_features))).flatten() # distance of pattern 1 to all other patterns
            index_to_other_points_distances[index_1] = distances
            distance_percentiles = percentileofscore(distances,distances)
            index_to_other_points_distance_percentiles[index_1] = distance_percentiles

        pattern_1_to_2_distance_percentile = distance_percentiles[index_2]
        pattern_1_to_2_distance = distances[index_2]

        # calculate distance percentiles from index 2 to 1

        try:
            distance_percentiles = index_to_other_points_distance_percentiles[index_2]
            distance = index_to_other_points_distances[index_2]
        except:
            distances = euclidean_distances(pca_scaled_cluster_features,pattern_2_features.reshape(1,len(pattern_2_features))).flatten()
            index_to_other_points_distances[index_2] = distances
            distance_percentiles = percentileofscore(distances,distances)
            index_to_other_points_distance_percentiles[index_2] = distance_percentiles

        pattern_2_to_1_distance_percentile = distance_percentiles[index_1]
        pattern_2_to_1_distance = distances[index_1]

        #larger_percentile = np.max([pattern_1_to_2_distance_percentile,pattern_2_to_1_distance_percentile])
        larger_distance = np.max([pattern_1_to_2_distance,pattern_2_to_1_distance])
        
        pairs_to_compare_distances += [larger_distance]

    pairs_to_compare_distances = np.array(pairs_to_compare_distances)
    pattern_pairs = np.array(list(zip(pattern_1_names,pattern_2_names,pairs_to_compare_distances)))

    return pattern_pairs[:,0:2], pattern_pairs[:,2].astype(float)




def get_point_point_distances(comparison_pairs,pca_scaled_cluster_features):
    pair_pca_scaled_feature_distances = []

    for index_pair in comparison_pairs:
        index_1, index_2 = index_pair[0], index_pair[1]

        if index_1 >= index_2: continue # only look at upper diagonal index pairs to avoid duplicates

        pattern_1_features = pca_scaled_cluster_features[index_1]
        pattern_2_features = pca_scaled_cluster_features[index_2]

        distance = euclidean_distances([pattern_1_features],[pattern_2_features])[0][0]

        pair_pca_scaled_feature_distances += [distance]
    
    return pair_pca_scaled_feature_distances