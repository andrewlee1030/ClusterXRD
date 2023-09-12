import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .peak_processing import *
from .imgs_to_movie import *
import shutil
import os


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

def save_kmeans_plots(kmeans, wafer_dir, peak_plot_dir, non_amorphous_names, gifs, gif_dir):

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
                shutil.copy(f'{peak_plot_dir}/{prefix}_plot_RAW.png',f'{label_dir}/{prefix}_plot_RAW.png')  
            except:
                pass # will skip over amorphous plots

        if gifs == True: make_gif(label_dir,f'label_{unique_label}',gif_dir) 