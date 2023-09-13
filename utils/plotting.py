import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .peak_processing import *
from .imgs_to_movie import *


def plot_xrd_with_peaks(histogram_name,raw_data_dir = '.', split_data_dir = 'split_histograms', peak_data_dir = 'peaks', save_dir = 'plots',normalize_value = 1,save=True):

    '''
    Plots splot XRD histograms with peak location.

    Args:
        histogram_name (str): name of the histogram
        raw_data_dir (str), default '.': path to raw histogram intensities
        split_data_dir (str), default 'split_histograms':  path to split histogram intensities
        peak_data_dir (str), default 'peaks': path to peak data
        save_dir (str), default 'plots': path to save plots into
        normalize_value (float), default 1: value to divide histogram intensities by for normalization purposes
        save (bool), default True: boolean for whether or not to save plots

    Returns:
        None: plots XRD histograms and saves them (if save==True)

    '''

    try:
        peak_data = pd.read_csv(f'{peak_data_dir}/{histogram_name}_peaks.csv')
        background_data = np.genfromtxt(f'{split_data_dir}/{histogram_name}_background')/normalize_value
        crystalline_data = np.genfromtxt(f'{split_data_dir}/{histogram_name}_crystalline')/normalize_value
        raw_xrd_data = pd.read_csv(f'{raw_data_dir}/{histogram_name}_1D.csv',header=None)[1]/normalize_value
    except:
        raise Exception('Cannot read data.')

    plt.title(histogram_name)
    plt.xlabel('Index')
    plt.ylabel('Intensity')

    # plot raw XRD data
    plt.plot(raw_xrd_data,'k',label='Raw XRD',linewidth=0.75,alpha=0.6)
    if save: plt.savefig(f'{save_dir}/{histogram_name}_plot_RAW.png',dpi=300,bbox_inches='tight')

    # plot background and crystalline raw data
    plt.plot(background_data,label='background',alpha=0.7,linewidth=1)
    plt.plot(crystalline_data,label='crystalline',alpha=0.7,linewidth=1)
    
    try:
        # plot peak data
        plt.scatter(peak_data['Peak Index'],crystalline_data[peak_data['Peak Index'].to_numpy()],s=90,color='g',marker='*',zorder=3,label='Peaks')
        plt.scatter(peak_data['Peak Start Index'],crystalline_data[peak_data['Peak Start Index'].to_numpy()],s=10,color='b',marker=9,zorder=3,label='Start')
        plt.scatter(peak_data['Peak End Index'],crystalline_data[peak_data['Peak End Index'].to_numpy()],s=10,color='r',marker=8,zorder=3,label='End')
    except:
        pass

    plt.legend()


    if save: 
        plt.savefig(f'{save_dir}/{histogram_name}_plot.png',dpi=300,bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_kmeans_plots(kmeans, wafer_dir, peak_plot_dir, non_amorphous_names, gifs, gif_dir, save_raw_plot=False):
    '''
    Moves histogram plots (split histograms with peaks) around based on clustering outcomes, generates gifs of clusters

    Args:
        kmeans (class sklearn.cluster._kmeans.KMeans): KMeans object after clustering histograms
        wafer_dir (str): path for this set of histograms
        peak_plot_dir (str): path to generated histogram plots
        non_amorphous_names (numpy array): list of non amorphous histogram names
        gifs (bool): if True, generates histogram gifs for each cluster
        gifs_dir (str): directory to save gifs into
        save_raw_plot (bool): if True, also moves raw histogram plots to cluster directories
    '''
    unique_kmeans_labels = list(set(kmeans.labels_))
    for unique_label in unique_kmeans_labels:

        # prepare dir for visualizing this cluster class
        label_dir = f'{wafer_dir}/{unique_label}'
        os.mkdir(label_dir)

        matching_indices = np.argwhere(kmeans.labels_ == unique_label).flatten()
        matching_prefixes = non_amorphous_names[matching_indices]
        
        # copy plots to label dir
        for prefix in matching_prefixes:
            try:
                shutil.copy(f'{peak_plot_dir}/{prefix}_plot.png',f'{label_dir}/{prefix}_plot.png')
                if save_raw_plot == True: shutil.copy(f'{peak_plot_dir}/{prefix}_plot_RAW.png',f'{label_dir}/{prefix}_plot_RAW.png')  
            except:
                pass # will skip over amorphous plots

        if gifs == True: make_gif(label_dir,f'label_{unique_label}',gif_dir) 

def overclustering_plots(intra_cluster_distances, cluster, n_under_cutoff, inter_cluster_distances,post_clustering_dir):
    # generate plots - this take a while so it's possible to turn this off
    fig, ax = plt.subplots()
    ax.hist(intra_cluster_distances, label='Point-point',alpha=0.6,color='green')

    ax.set_xlabel('Distance')
    ax.set_ylabel('Count (point-point)')
    plt.title(f'Cluster {cluster} | n other clusters within cutoff: {n_under_cutoff}')
    
    # also plot distance from this cluster's centroid to all other cluster centroids
    ax2 = ax.twinx()
    ax2.hist(inter_cluster_distances,label='Centroid-centroid',alpha=0.6,color='red')
    ax2.set_ylabel('Count (centroid-centroid)',color='red')

    # get legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    plt.savefig(f'{post_clustering_dir}/{cluster}_overclustering_plot.png',dpi=300)
    plt.close()

def underclustering_plots(similar_pair_pca_scaled_feature_distances,dissimilar_pair_pca_scaled_feature_distances,cluster,percent_dissimilar_over_threshold,post_clustering_dir):
    # generate distance plots - will take extra time, can turn off
    plt.figure()
    plt.hist(similar_pair_pca_scaled_feature_distances,label='Similar Pairs',alpha=0.5,bins=20,density=True)
    plt.hist(dissimilar_pair_pca_scaled_feature_distances,label='Dissimilar Pairs',alpha=0.5,bins=20,density=True)

    plt.xlabel('PCA Scaled Feature Distance')
    plt.ylabel('Probability Distributions')
    plt.title(f'Cluster: {cluster} | Pct over threshold: {percent_dissimilar_over_threshold}')
    plt.savefig(f'{self.post_clustering_dir}/{cluster}_underclustering_plot.png',dpi=300)
    plt.close()