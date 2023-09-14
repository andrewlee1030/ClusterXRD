import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import shutil
import matplotlib.pyplot as plt

def clustering_convergence_check(post_clustering_dir,
                                 cluster_dir): 
    '''
    Checks whether the number of additional/fewer clusters has converged. If so, subsequent rounds of clustering are halted.

    Args:
        post_clustering_dir (str): path to directory with post clustering analysis files
        cluster_dir (str): path to directory with the cluster log

    Returns:
        None: writes to local log file if convergence is reached
    '''

    log_a = open(f'{cluster_dir}/cluster_log.txt','a')
    log_r = open(f'{cluster_dir}/cluster_log.txt','r').read()

    # check for convergence from under/over-clustering

    try:
        underclustering = pd.read_csv(f'{post_clustering_dir}/underclustering_statistics.csv')
    except:
        if "Convergence" not in log_r: 
            log_a.write('Clustering convergence check - Could not find underclustering statistics \n')
            return None

    underclustering_number = underclustering['Additional clusters needed'][0]

    if underclustering_number != 0: delta_cluster_number = underclustering_number
    else:
        try:
            overclustering = pd.read_csv(f'{post_clustering_dir}/overclustering_statistics.csv')
        except:
            if "Convergence" not in log_r: 
                log_a.write('Clustering convergence check - Could not find overclustering statistics \n')
                return None

        overclustering_number = overclustering['Additional clusters needed'][0]

        if overclustering_number == 0 and underclustering_number == 0: # clustering is converged on an optimal number of clusters
            delta_cluster_number = 0 # no more reclustering is required
        elif overclustering_number != 0 and underclustering_number == 0:
            delta_cluster_number = overclustering_number

    
    # check for convergence from oscillation after the delta cluster number is determined

    try: 
        log = open(f'{cluster_dir}/cluster_log.txt','r').readlines()
    except: 
        return None
    
    try: 
        n_clustering_clusters = []

        for line in log:
            if 'Clustering' in line: 
                n_clustering_clusters += [int(re.search('[\S*] - (\d*)',line).group(1))]
            else: 
                return None

        previous_n_clusters = n_clustering_clusters[-1] 
        previous_previous_n_clusters = n_clustering_clusters[-2]

        delta = previous_n_clusters - previous_previous_n_clusters
        if delta == delta_cluster_number * -1:
            if np.abs(delta) == 1: # if oscillations are +1/-1 clusters
                log_a.write('(!) N Clusters Oscillation Detected (!) \n')
                log_a.write(f'Optimal Cluster count is between {previous_n_clusters} and {previous_previous_n_clusters} clusters')
                log_a.write('** Convergence Reached ** \n')
                return None
            else:
                delta_cluster_number = int(delta_cluster_number / 2) # takes the floor of the average since int(1.5) = 1 if delta == 3
                log_a.write('(!) N Clusters Oscillation Detected (!) \n')
                log_a.write(f'Optimal Cluster count is between {previous_n_clusters} and {previous_previous_n_clusters} clusters')
                log_a.write(f'Trying {previous_n_clusters + delta_cluster_number} clusters next.')

    except: # when there hasn't been enough iterations to check for oscillations
        pass 

    if delta_cluster_number == 0: 
        log_a.write('** Convergence Reached ** \n')
    else: 
        np.savetxt(f'{post_clustering_dir}/reclustering_input',[delta_cluster_number])

def get_similarity_matrix(cluster_features,
                          cluster_dir,
                          split_histograms_dir):
    '''
    Calculates pairwise similarities between histograms from each cluster.

    Args:
        cluster_features (pandas DataFrame): features values with cluster labels for each histogram
        cluster_dir (str): path to files for this clustering iteration 
        split_histograms_dir (str): path to background separated histograms

    Returns:
        None: writes similarity matrix to a local file
    '''

    try:
        unique_clusters = list(set(cluster_features['Cluster labels']))
    except:
        return None


    if os.path.isdir(f'{cluster_dir}/similarity_scores'): shutil.rmtree(f'{cluster_dir}/similarity_scores')
    os.makedirs(f'{cluster_dir}/similarity_scores',exist_ok=True)

    for cluster in unique_clusters: # loop through clusters
        if cluster == -1.0: continue # skip amorphous
        
        this_cluster_features = cluster_features[cluster_features['Cluster labels'] == cluster]
        
        cluster_plot_names = this_cluster_features['name']
        xrd_data_list = [np.genfromtxt(f'{split_histograms_dir}/{name}_crystalline') for name in cluster_plot_names]


        xrd_data = pd.DataFrame(data = np.vstack([xrd_data_list[i] for i in range(len(xrd_data_list))])).T


        # get cosine similarities
        cosine_similarities = cosine_similarity(xrd_data.T) # this transpose is important for the cosine similarity function

        pd.DataFrame(data=cosine_similarities).set_axis(cluster_plot_names,axis=1).set_axis(cluster_plot_names,axis=0).to_csv(f'{cluster_dir}/similarity_scores/{int(cluster)}_cosine.csv')
 
