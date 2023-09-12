import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import shutil
import matplotlib.pyplot as plt

def clustering_convergence_check(post_clustering_dir,cluster_dir): # may not need this
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

def get_similarity_stats(similarities,prefix):

    try:

        mean = np.mean(similarities)
        median = np.median(similarities)
        std = np.std(similarities)
        minimum = np.min(similarities)
        pctile_25 = np.quantile(similarities,0.25)

        return {f'mean_{prefix}':mean, 
                f'median_{prefix}':median,
                f'std_{prefix}':std, 
                f'min_{prefix}':minimum, 
                f'pctile_25_{prefix}':pctile_25}
    except:
        return dict() # return empty dictionary when there are no phases in a cluster

def get_similarity_matrix(cluster_features,wafer_dir,xrd_data_dir):
    
    try:
        unique_clusters = list(set(cluster_features['Cluster labels']))
    except:
        return None


    if os.path.isdir(f'{wafer_dir}/similarity_scores'): shutil.rmtree(f'{wafer_dir}/similarity_scores')
    os.makedirs(f'{wafer_dir}/similarity_scores',exist_ok=True)

    for cluster in unique_clusters: # loop through clusters
        if cluster == -1.0: continue # skip amorphous
        
        this_cluster_features = cluster_features[cluster_features['Cluster labels'] == cluster]
        
        cluster_plot_names = this_cluster_features['name']
        xrd_data_list = [np.genfromtxt(f'{xrd_data_dir}/{name}_crystalline') for name in cluster_plot_names]


        xrd_data = pd.DataFrame(data = np.vstack([xrd_data_list[i] for i in range(len(xrd_data_list))])).T


        # get cosine similarities
        cosine_similarities = cosine_similarity(xrd_data.T) # this transpose is important for the cosine similarity function

        pd.DataFrame(data=cosine_similarities).set_axis(cluster_plot_names,axis=1).set_axis(cluster_plot_names,axis=0).to_csv(f'{wafer_dir}/similarity_scores/{int(cluster)}_cosine.csv')
 
def plot_histogram(feature_set,title,across_all_wafers,has_crystalline, fig_dir = '.'):
    similarity_metrics = has_crystalline[feature_set].to_numpy().flatten()
    similarity_metrics_no_nan = similarity_metrics[~np.isnan(similarity_metrics)]
    if across_all_wafers: 
        multiplier = 1.5
        nbins=5
    else: 
        multiplier = 1
        nbins=50
    counts,bins = np.histogram(similarity_metrics_no_nan,bins=nbins)
    center = (bins[:-1] + bins[1:])/2
    width = multiplier * (bins[1] - bins[0])
    plt.bar(center, counts, align='center', width=width)
    plt.xlabel('Similarity Coefficient / Score')
    plt.xlim([-0.1,1.0])
    plt.ylabel('Number of Clusters')
    if across_all_wafers: title += ' across all wafers'
    plt.title(title)
    plt.savefig(f'{fig_dir}/figs/{title}.png',dpi=300)
    plt.close()

def plot_histogram_per_wafer(similarities,title,fig_dir = None,nbins=20):
    counts,bins = np.histogram(similarities,bins=nbins)
    center = (bins[:-1] + bins[1:])/2
    width = (bins[1] - bins[0])
    plt.bar(center, counts, align='center', width=width)
    plt.xlabel('Similarity Coefficient / Score')
    plt.xlim([-0.1,1.0])
    plt.ylabel('Count')
    plt.title(title)
    if fig_dir == None: plt.show()
    else:
        plt.savefig(f'{fig_dir}/similarity_scores/{title}.png',dpi=300)
        plt.close()