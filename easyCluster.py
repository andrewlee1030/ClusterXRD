import numpy as np
import os
from .utils import *
import glob
import multiprocessing
import pickle
from sklearn import preprocessing
import shutil
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm



class ezcluster():
    '''
    Should be initialized once per wafer/set of XRD histograms to be analyzed.
    '''
    def __init__(self,k_clusters=4,input_dir=None,filename_suffix='_1D.csv',features=None,split_histogram_dir=None,peak_dir=None,plot_dir=None,features_dir=None):
        '''
        Defines required information for clustering

        Args:
            input_dir (str): directory of raw histograms
        '''
        
        self.k_clusters = k_clusters
        self.input_dir = input_dir

        # get histogram paths
        self.histogram_paths = glob.glob(f'{self.input_dir}/*.csv')

        # get histogram file names
        self.filename_suffix = filename_suffix
        self.histogram_filenames = np.array([x.split('/')[-1].replace(self.filename_suffix,'') for x in self.histogram_paths])

        # load raw XRD data, raise exception if there are invalid files
        try:
            self.histograms = np.array([pd.read_csv(path,header=None)[1] for path in self.histogram_paths])
            self.q_data = np.array([pd.read_csv(path,header=None)[0] for path in self.histogram_paths])
        except:
            raise Exception(f'Something wrong with opening histograms in {self.input_dir}')

        # check for uniform array length for histogram data
        histogram_lengths = np.array([len(x) for x in self.histograms])
        unique_histogram_lengths = set(histogram_lengths)
        if len(unique_histogram_lengths) != 1: raise Exception(f'ERROR: multiple histogram lengths detected for {self.input_dir} ({unique_histogram_lengths})')

        # initialize features to be used

        if features == None: self.features = feature_names
        else: self.features = features

        # initialize precalculated directories
        if split_histogram_dir != None:
            self.split_histogram_dir = split_histogram_dir
        else:
            self.split_histogram_dir = None
        if peak_dir != None:
            self.peak_dir = peak_dir
        else:
            self.peak_dir = None
        if plot_dir != None:
            self.plot_dir = plot_dir
        else:
            self.plot_dir = None
        if features_dir != None:
            self.features_dir = features_dir
        else:
            self.features_dir = None

        print(f'Initialized clustering class for {self.input_dir}')

    def find_peaks(self,histogram_intensities,histogram_filename,save_dir='peaks',stubby_peak_removal=True): 
        '''
        Identifies peak starts, locations, and ends for a given histogram intensity array. DO NOT run directly, use the parallelized wrapper instead

        Args:
            histogram_intensities (numpy array): array of histogram intensities
            histogram_filename (str): filename for the histogram data from which peaks are identified
            save_dir (str), default 'Peaks': directory in which peak information will be saved, if None then nothing will be saved
            stubby_peak_removal (bool), default 'True': if True, will remove small/stubby peaks
        Returns:
            pandas dataframe: peak starts, locations, and ends
    
        '''
        n_datapoints = len(histogram_intensities)

        roll_index_lengths = [1,round(n_datapoints/1000),round(n_datapoints/200)] # maximum roll index length cannot be longer than half the width of the skinniest peak you wish to detect (measured in index units)

        peak_indices_list = []
        peak_starts_list = []
        peak_ends_list = []
        valley_indices_list = []

        for roll_index_length in roll_index_lengths:
            A = histogram_intensities 
            B = np.roll(histogram_intensities,roll_index_length) 
            C = A - B # change from previous point
            D = np.roll(C,-roll_index_length) # change to the next point

            E = C > 0 
            F = D > 0 
            G = C < 0 
            H = D < 0 
            I = C == 0 
            J = D == 0
            
            peak_indices_list += [np.argwhere(E & H).flatten()]
            peak_starts_list += [np.argwhere(I & F).flatten()]
            peak_ends_list += [np.argwhere(G & J).flatten()]
            valley_indices_list += [np.argwhere(G & F).flatten()]
        
        all_peak_indices = reduce(np.intersect1d,peak_indices_list) # find intersection between all peak indices lists
        all_peak_starts = peak_starts_list[0]
        all_peak_ends = peak_ends_list[0]
        all_valley_indices = valley_indices_list[0]

        # add all valleys to starts and ends since one valley = one start + one end

        all_peak_starts = np.sort(np.concatenate([all_peak_starts,all_valley_indices]))
        all_peak_ends = np.sort(np.concatenate([all_peak_ends,all_valley_indices]))
        
        max_number_peaks = len(all_peak_indices)

        peak_start_indices = np.zeros(max_number_peaks,dtype=int)
        peak_indices = np.zeros(max_number_peaks,dtype=int)
        peak_end_indices = np.zeros(max_number_peaks,dtype=int)
        
        peak_group_number = 0
        
        for i in all_peak_indices: # loop through each peak and find suitable corresponding start and end indices
            
            potential_starts = np.flip(all_peak_starts[all_peak_starts < i]) # need to flip because potential starts are evaluated from high values to low!
            potential_ends = all_peak_ends[all_peak_ends > i]

            start = 0  #initialize default value in case no corresponding start can be found
            end = 0
            
            try: start = potential_starts[0]
            except: pass

            try: end = potential_ends[0]
            except: pass

            peak_start_indices[peak_group_number] = start
            peak_indices[peak_group_number] = i
            peak_end_indices[peak_group_number] = end
            peak_group_number += 1
        
        df = pd.DataFrame(data={'Peak Start Index': peak_start_indices,
                                'Peak Index': peak_indices,
                                'Peak End Index': peak_end_indices},dtype=int)

        df = df[~(df == 0).any(axis=1)] # gets rid of any incomplete / fake peaks

        if stubby_peak_removal == True: df = remove_stubby_peaks(histogram_intensities,df)

        if save_dir: df.to_csv(f'{self.peak_dir}/{histogram_filename}_peaks.csv')

        return df

    def find_peaks_parallelized_wrapper(self,save_dir='peaks'):
        '''
        Wrapper function to run the 'find_peaks' function in parallel for all histograms within a wafer


        '''
        
        if self.peak_dir != None: return None # if True, then peaks have already been identified

        # initialize directory for storing peaks
        self.peak_dir = f'{self.input_dir}/{save_dir}' # full path
        os.makedirs(self.peak_dir,exist_ok=True)

        n_parallel_cpus = multiprocessing.cpu_count() - 2 # reserve 2 cores
        p = multiprocessing.Pool(n_parallel_cpus)

        multiprocess_plotting_inputs = []

        for i in range(len(self.histograms)):
            histogram_intensities = self.histograms[i]
            histogram_filename = self.histogram_filenames[i]
            multiprocess_plotting_inputs += [[histogram_intensities,histogram_filename]]
        
        p.starmap(self.find_peaks, multiprocess_plotting_inputs)
        print(f'Finished finding peaks for {self.input_dir}')

    def split_histogram(self,smooth_q_background_setting, save_dir = 'split_histograms', save=True):
        
        if self.split_histogram_dir != None: return None # if True, then histograms have already been split

        
        histograms = self.histograms
        self.split_histogram_dir = f'{self.input_dir}/{save_dir}' # full path
        os.makedirs(self.split_histogram_dir,exist_ok=True)

        background,fastq,au = separate_background(histograms, plot=False,
                                                    threshold = 50,
                                                    smooth_q = 1,
                                                    smooth_neighbor_background = 0,
                                                    smooth_q_background = smooth_q_background_setting,
                                                    bg_smooth_post = 15)
        for i in range(len(histograms)):
            fname = self.histogram_filenames[i]
            histogram_data = histograms[i]
            background_data = background[0][i]
            offset = get_best_offset(histogram_data, background_data)

            background_data -= offset
            crystalline_data = histogram_data - background_data # crystalline data derived from raw - background

            crystalline_data[crystalline_data < 0] = 0  # crystalline data derived from raw - background

            background[0][i] = background_data
            fastq[0][i] = crystalline_data

            if save == True:
                np.savetxt(f'{self.split_histogram_dir}/{fname}_background',background_data)
                np.savetxt(f'{self.split_histogram_dir}/{fname}_crystalline',crystalline_data)

        print(f'Finished splitting histograms for {self.input_dir}')
        return background,fastq

    def plot_split_data(self,save_dir='plots'):
        
        if self.plot_dir != None: return None # if True, then plots have already been generated

        n_parallel_cpus = multiprocessing.cpu_count() - 2 # reserve 2 cores
        p = multiprocessing.Pool(n_parallel_cpus)

        multiprocess_plotting_inputs = []
        
        # make directory for plots
        self.plot_dir = f'{self.input_dir}/{save_dir}'
        os.makedirs(self.plot_dir,exist_ok=True)

        for filename in self.histogram_filenames:
            multiprocess_plotting_inputs += [[filename,self.input_dir, self.split_histogram_dir, self.peak_dir, self.plot_dir]]

        p.starmap(plot_xrd_with_peaks, multiprocess_plotting_inputs)
        print(f'Finished plotting histograms for {self.input_dir}')

    def calculate_features(self):

        if self.features_dir != None: return None # if True, then features have already been calculated


        self.features_dir = f'{self.input_dir}/features'
        os.makedirs(self.features_dir,exist_ok=True)

        peak_feats = [] # list of dictionaries
        feat_stats = []


        for i in range(len(self.histogram_filenames)):
            # try:
            filename = self.histogram_filenames[i]
            raw_intensities = self.histograms[i]
            q_data = self.q_data[i]
            peak_data = pd.read_csv(f'{self.peak_dir}/{filename}_peaks.csv', index_col=0)
            background_data = np.genfromtxt(f'{self.split_histogram_dir}/{filename}_background')
            crystal_data = np.genfromtxt(f'{self.split_histogram_dir}/{filename}_crystalline')

            max_peak_intensity_this_wafer = np.max(crystal_data)

            q_width = np.max(q_data) - np.min(q_data) # need to normalize all widths by the q width! - units of inverse angstroms

            # first check if amorphous
            amorphous_status = is_amorphous(crystal_data,background_data,peak_data,max_peak_intensity_this_wafer)
            
            # set up data for feature generation
            peaks = [crystal_data[row['Peak Start Index']:row['Peak End Index']+1] for i,row in peak_data.iterrows() ]
            peak_intensities = crystal_data[peak_data['Peak Index']]
            peak_base_widths = get_peak_base_widths(peak_data,crystal_data,q_width)
            fwhms = get_fwhm_v2(peaks,crystal_data, q_width) # this will replace the fwhm above
            hms = peak_intensities / 2.0
            peak_start_end_intensities = get_peak_start_end_intensities(crystal_data, peak_data)

        ################## ALL FEATURES ARE BELOW ##################

            # number of peaks features
            peak_bin_sizes = np.array([0.10,0.20,0.50,0.80]) * np.max(peak_intensities)
            n_peaks_per_bin_size = get_n_peaks_per_bin_size(peak_intensities,peak_bin_sizes)
            n_peaks = np.sum(n_peaks_per_bin_size)

            # peak intensity features
            i_pct_25, i_pct_75, i_mu, i_max = get_derived_features(peak_intensities)
            
            # peak width features
            fwhm_pct_25, fwhm_pct_75, fwhm_mu, fwhm_max = get_derived_features(fwhms)
            bw_pct_25, bw_pct_75, bw_mu, bw_max = get_derived_features(peak_base_widths)

            # peak area features
            pct_area_under_raw_data = get_percentage_area_under_raw_data(raw_intensities) # defined in peak_processing.py from codes_for_import
            
            # floating peak features
            n_floating_peaks = get_n_floating_peaks(crystal_data, peak_data,min_threshold = 10)
            pse_pct_25, pse_pct_75, pse_mu, pse_max = get_derived_features(peak_start_end_intensities)

            final_feature_dict = {'name':filename, 
                    'n_peaks': n_peaks,
                    'I_max': i_max, 
                    'I_mu': i_mu, 
                    'I_25th_percentile': i_pct_25, 
                    'I_75th_percentile': i_pct_75,
                    'fwhm_25th_percentile': fwhm_pct_25, 
                    'fwhm_75th_percentile': fwhm_pct_75,
                    'fwhm_max': fwhm_max, 
                    'fwhm_mu': fwhm_mu, 
                    'amorphous': amorphous_status, 
                    'bw_max': bw_max,
                    'bw_mu':bw_mu,
                    'bw_25th_percentile': bw_pct_25, 
                    'bw_75th_percentile': bw_pct_75,
                    'n_floating_peaks': n_floating_peaks,
                    'pse_max': pse_max,
                    'pse_mu': pse_mu,
                    'pse_25th_percentile': pse_pct_25, 
                    'pse_75th_percentile': pse_pct_75, 
                    'pct_area_under_raw_data': pct_area_under_raw_data} # v2

            # add features for number of peaks by size
            for i in range(len(n_peaks_per_bin_size)):
                if i == 0: feature_name = f'n_peaks_size_{i}'
                else: feature_name = f'n_peaks_size_{i}'
                final_feature_dict[feature_name] = n_peaks_per_bin_size[i]

            feat_stats.append( final_feature_dict )
            peak_feats.append( {'name':filename, 'peak intensities':peak_intensities, 'fwhms': fwhms, 'hms':hms} ) 
        
            # except:
            #     print(f'Error with filename: {filename}, this pattern has {len(peak_data)} peaks.')
            #     pass
        
        feat_stats_df = pd.DataFrame(feat_stats)
        feat_stats_df.to_csv(f'{self.features_dir}/features_original.csv',index=False)
        print(f'Finished calculating features for {self.input_dir}')

    def scale_features(self):

        features_original = pd.read_csv(f'{self.features_dir}/features_original.csv')
        non_amorphous_features_original = features_original.query('amorphous == False')
        # filter for features that we actually want to use for clustering
        non_amorphous_features_filtered = non_amorphous_features_original[self.features]
        if len(non_amorphous_features_filtered) == 0: return None
        
        # apply min-max scaling to non amorphous patterns
        scaler = preprocessing.MinMaxScaler() # min-max scaler
        scaler.fit(non_amorphous_features_filtered)
        X_scaled = scaler.transform(non_amorphous_features_filtered)

        # save scaled features to csv
        X_scaled = pd.DataFrame(data=X_scaled,columns=self.features)
        X_scaled.to_csv(f'{self.features_dir}/features_scaled.csv',index=False)

        # save scaler object
        with open(f'{self.features_dir}/scaler.obj','wb+') as h:
            pickle.dump(scaler,h)
            h.close()

        # perform PCA on scaled features
        n_pca_components = 4
        if len(X_scaled) < n_pca_components: n_pca_components = len(X_scaled)
        pca = PCA(n_components=n_pca_components)
        pca.fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        np.savetxt(f'{self.features_dir}/features_pca_scaled', X_pca)

        # save pca object
        with open(f'{self.features_dir}/pca.obj','wb+') as f:
            pickle.dump(pca,f)
            f.close()
        
        print(f'Finished scaling features + PCA for {self.input_dir}')

    def perform_clustering(self,use_pca=True,gifs=True,max_iter=300,n_init=30,tol=1e4,clustering_name=0):
        self.cluster_dir = f'{self.input_dir}/clusters'
        
        ###### for the initial run only
        if clustering_name == 0:

            # clear previous clustering runs
            try: shutil.rmtree(self.cluster_dir)
            except: pass
            os.makedirs(self.cluster_dir,exist_ok=True)
            
            # initialize some information
            log = open(f'{self.cluster_dir}/cluster_log.txt','a')
            if self.k_clusters == 0:
                log.write('Having 0 clusters is not allowed - changing to 1 cluster \n')
                self.k_clusters = 1
        
        else: ###### for subsequent runs only
            log = open(f'{self.cluster_dir}/cluster_log.txt','a')

        ###### prepare directory for clustering results
        self.cluster_results_dir = f'{self.cluster_dir}/cluster_results_{clustering_name}'
        os.makedirs(self.cluster_results_dir,exist_ok=True)

        if gifs:
            # prepare directory for gifs
            self.gif_dir = f'{self.cluster_results_dir}/gifs'
            os.mkdir(self.gif_dir)

        ###### read in original wafer features
        original_feature_df = pd.read_csv(f'{self.features_dir}/features_original.csv')
        pattern_names = original_feature_df['name']

        ###### identify amorphous samples and copy plots to cluster folder
        is_amorphous = np.array(original_feature_df['amorphous'] == True)
        feature_df_amorphous = original_feature_df[is_amorphous]

        self.amorphous_dir = f'{self.cluster_results_dir}/amorphous'
        os.makedirs(self.amorphous_dir,exist_ok=True)

        for filename in feature_df_amorphous['name']: shutil.copy(f'{self.plot_dir}/{filename}_plot.png',f'{self.amorphous_dir}/{filename}_plot.png')

        ###### generate gifs for amorphous xrd patterns
        if gifs == True: make_gif(self.amorphous_dir,'amorphous',self.gif_dir)

        ###### prepare cluster labels
        cluster_labels = np.zeros(len(original_feature_df)) - 2

        ###### save amorphous cluster labels
        cluster_labels[np.nonzero(is_amorphous)] = -1

        ###### read in features (PCA or scaled)

        try:
            if use_pca == True:
                X_clustering = np.genfromtxt(f'{self.features_dir}/features_pca_scaled')
            else:
                X_clustering = pd.read_csv(f'{self.features_dir}/features_scaled.csv')
        except: # in case there are no crystalline histograms
            log.write(f'No crystalline histograms found. \n')
            log.close()
            return None

        if len(X_clustering) < self.k_clusters: self.k_clusters = len(X_clustering)
        kmeans = KMeans(n_clusters=self.k_clusters,n_init=n_init,max_iter=max_iter,tol=tol).fit(X_clustering)
        cluster_labels[np.nonzero(~is_amorphous)] = kmeans.labels_
    
        ###### save plots for non amorphous xrd patterns
        save_kmeans_plots(kmeans, self.cluster_results_dir, self.plot_dir, pattern_names[~is_amorphous].to_numpy(), gifs,self.gif_dir)

        ###### Pickle Clustering object

        with open(f'{self.cluster_results_dir}/kmeans.obj','wb+') as g:
            pickle.dump(kmeans,g)
            g.close()

        assert(-2 not in cluster_labels)
        original_feature_df['Cluster labels'] = cluster_labels
        original_feature_df.to_csv(f'{self.features_dir}/features_original_w_labels.csv',index=False)

        log.write(f'Clustering - {self.k_clusters} clusters \n')
        log.close()
        self.previous_clustering_name = clustering_name
        self.clustering_name = clustering_name + 1
        print(f'Finished clustering {self.previous_clustering_name}')

    def get_cluster_similarities(self):
        log_a = open(f'{self.cluster_dir}/cluster_log.txt','a')
        log_r = open(f'{self.cluster_dir}/cluster_log.txt','r').read()
        
        try:
            wafer_features = pd.read_csv(f'{self.features_dir}/features_original_w_labels.csv')
        except:
            if "Convergence" not in log_r: log_a.write('Get Cluster Similarities - There may not be any crystalline patterns in this wafer \n')
            return None
        
        try:
            get_similarity_matrix(wafer_features,self.cluster_results_dir,self.split_histogram_dir) # function is from clustering_analysis.py import
        except:
            log_a.write('Get Cluster Similarities - There may not be any crystalline patterns in this wafer \n')
            return None

        print(f'Finished calculating similarities.')

    def underclustering_analysis(self,plots=False):
        additional_clusters_needed = 0
        log_a = open(f'{self.cluster_dir}/cluster_log.txt','a')
        log_r = open(f'{self.cluster_dir}/cluster_log.txt','r').read()


        similarity_scores_dir = f'{self.cluster_results_dir}/similarity_scores'

        if not os.path.isfile(f'{self.features_dir}/features_original_w_labels.csv'): 
            if 'Convergence' not in log_r: 
                log_a.write('Underclustering analysis - no crystalline samples found \n')
                return None
        else:
            features = pd.read_csv(f'{self.features_dir}/features_original_w_labels.csv')

        pca_path = f'{self.features_dir}/pca.obj'
        scaler_path = f'{self.features_dir}/scaler.obj'
        pca = pickle.load(open(pca_path,'rb'))
        scaler = pickle.load(open(scaler_path,'rb'))

        # create directory for analysis outputs
        self.post_clustering_dir = f'{self.cluster_results_dir}/post_clustering_analysis'
        if not os.path.isdir(self.post_clustering_dir): 
            os.makedirs(self.post_clustering_dir,exist_ok=True)

        similarity_score_file_paths = glob.glob(f'{similarity_scores_dir}/*cosine.csv')

        for similarity_score_file_path in similarity_score_file_paths:
            similarity_matrix = pd.read_csv(similarity_score_file_path).iloc[:,1:]

            this_cluster_number = int(similarity_score_file_path.split('/')[-1].split('_')[0])

            cluster_features = features.query(f'`Cluster labels` == {this_cluster_number}')

            n_patterns_in_cluster = len(cluster_features)
            if n_patterns_in_cluster < 15: continue # having too few patterns in a cluster makes this function too unreliable 
            n_possible_comparisons_in_cluster = n_patterns_in_cluster * (n_patterns_in_cluster-1) / 2 # number of unique comparisons hence the division by 2

            scaled_cluster_features = scaler.transform(cluster_features[feature_names]) # do NOT do fit_transform
            pca_scaled_cluster_features = pca.transform(scaled_cluster_features) # do NOT do fit_transform

            similarity_min = 0
            similarity_max = 0.70

            similarity_matrix = pd.read_csv(similarity_score_file_path).iloc[:,1:]
            similarity_matrix_numerical = similarity_matrix.to_numpy()

            dissimilar_pairs_to_compare = np.argwhere((similarity_matrix_numerical > similarity_min) & (similarity_matrix_numerical < similarity_max))
            similar_pairs_to_compare = np.argwhere((similarity_matrix_numerical >= similarity_max)) 
            if len(dissimilar_pairs_to_compare) < 0.1 * n_possible_comparisons_in_cluster: continue # don't bother splitting clusters with a small number of dissimilar patterns

            similar_pair_pca_scaled_feature_distances = get_point_point_distances(similar_pairs_to_compare,pca_scaled_cluster_features)

            dissimilar_pair_pca_scaled_feature_distances = get_point_point_distances(dissimilar_pairs_to_compare,pca_scaled_cluster_features)

            similar_median = np.median(similar_pair_pca_scaled_feature_distances)
            similar_std = np.std(similar_pair_pca_scaled_feature_distances)

            max_similar_distance_threshold = similar_median + similar_std

            n_dissimilar_over_threshold = np.sum(dissimilar_pair_pca_scaled_feature_distances > max_similar_distance_threshold)
            percent_dissimilar_over_threshold = n_dissimilar_over_threshold / len(dissimilar_pair_pca_scaled_feature_distances)
            if percent_dissimilar_over_threshold > 0.40: additional_clusters_needed += 1

            if plots == True:
                # generate distance plots - will take extra time, can turn off
                plt.figure()
                plt.hist(similar_pair_pca_scaled_feature_distances,label='Similar Pairs',alpha=0.5,bins=20,density=True)
                plt.hist(dissimilar_pair_pca_scaled_feature_distances,label='Dissimilar Pairs',alpha=0.5,bins=20,density=True)

                plt.xlabel('PCA Scaled Feature Distance')
                plt.ylabel('Probability Distributions')
                plt.title(f'Cluster: {this_cluster_number} | Pct over threshold: {percent_dissimilar_over_threshold}')
                plt.savefig(f'{self.post_clustering_dir}/{this_cluster_number}_underclustering_plot.png',dpi=300)
                plt.close()
            

        underclustering_data = pd.DataFrame(data = {'Current cluster count': [len(similarity_score_file_paths)],
                                                    'Additional clusters needed': [additional_clusters_needed]})
            
        underclustering_data.to_csv(f'{self.post_clustering_dir}/underclustering_statistics.csv',index=False)
        
        log_a.write(f'Underclustering analysis - need {additional_clusters_needed} more clusters \n')
        log_a.close()
        print(f'Finished underclustering analysis.')

    def overclustering_analysis(self,plots=False): # clustering name needs to be the most recent run (highest number)

        log_a = open(f'{self.cluster_dir}/cluster_log.txt','a')
        log_r = open(f'{self.cluster_dir}/cluster_log.txt','r').read()

        try:
            # get PCA scaled features
            pca_transformed_scaled_features = np.genfromtxt(f'{self.features_dir}/features_pca_scaled')
            if pca_transformed_scaled_features.shape == (): 
                if 'Convergence' not in log_r: 
                    log_a.write('Overclustering analysis - PCA features are not in a 2D array \n')
                    return None
        except:
            if 'Convergence' not in log_r: 
                log_a.write('Overclustering analysis - there may be no crystalline patterns in this wafer \n')
                return None

        # create directory for analysis outputs
        if not os.path.isdir(self.post_clustering_dir): return None # if the directory doesn't exist, then underclustering has not been done yet

        underclustering = pd.read_csv(f'{self.post_clustering_dir}/underclustering_statistics.csv')

        if underclustering['Additional clusters needed'][0] != 0: 
            if 'Convergence' not in log_r: log_a.write('Overclustering analysis - underclustering has not finished \n')
            return None # don't try to overcluster before underclustering finishes
        
        # get original features with clustering labels
        non_amorphous_features = pd.read_csv(f'{self.features_dir}/features_original_w_labels.csv').query('`Cluster labels` != -1')

        # load kmeans object
        f = open(f'{self.cluster_results_dir}/kmeans.obj','rb')
        kmeans = pickle.load(f)

        all_clusters = range(len(kmeans.cluster_centers_))
        n_clusters_to_reduce = 0

        if len(all_clusters) > 2: # overclustering analysis only works when there are a minimum number of clusters
            for cluster in all_clusters:
                intra_cluster_distances = calculate_intra_cluster_distances(pca_transformed_scaled_features,non_amorphous_features,cluster,kmeans)
                inter_cluster_distances = np.delete(calculate_inter_cluster_centroid_distances(kmeans,cluster),[cluster])

                if len(intra_cluster_distances) == 0: continue

                # cutoff = np.quantile(intra_cluster_distances,0.95)
                cutoff = 0.75 * np.max(intra_cluster_distances)

                n_under_cutoff = np.sum(inter_cluster_distances < cutoff)
                if n_under_cutoff > (len(all_clusters)-1)/2: n_clusters_to_reduce += 1
                
                if plots == True:
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

                    plt.savefig(f'{self.post_clustering_dir}/{cluster}_overclustering_plot.png',dpi=300)
                    plt.close()
        else:
            cutoff = np.nan
            n_under_cutoff = np.nan

        

        # save number of clusters to reduce and other data


        overclustering_data = pd.DataFrame(data = {'Current cluster count': [len(all_clusters)],
                                                    'Additional clusters needed': np.array([n_clusters_to_reduce])*-1, # need to multiply by -1 because these are number of clusters to be subtracted
                                                    })
        
        overclustering_data.to_csv(f'{self.post_clustering_dir}/overclustering_statistics.csv',index=False)
        log_a.write(f'Overclustering analysis - need {n_clusters_to_reduce} fewer clusters \n')
        log_a.close()
        print('Finished overclustering analysis.')

    def perform_reclustering(self):

        log_a = open(f'{self.cluster_dir}/cluster_log.txt','a')
        log_r = open(f'{self.cluster_dir}/cluster_log.txt','r').read()

        try:
            additional_n_clusters = int(np.genfromtxt(f'{self.post_clustering_dir}/reclustering_input'))
            current_n_clusters = pd.read_csv(f'{self.post_clustering_dir}/underclustering_statistics.csv')['Current cluster count'][0]
        except:
            if 'Convergence' not in log_r: 
                log_a.write('Over/under-clustering contradiction, cannot recluster. \n')
            return None
        if additional_n_clusters == 0:
            log_a.write('Convergence already reached. \n')
            placeholder_dir = f'{self.cluster_dir}/cluster_results_{self.clustering_name}'
            if os.path.isdir(placeholder_dir): shutil.rmtree(placeholder_dir)
            os.mkdir(placeholder_dir) # make an empty placehoder folder 
            return None

        self.k_clusters = current_n_clusters + additional_n_clusters
        self.perform_clustering(self,clustering_name = self.clustering_name)


if __name__ == '__main__':

    # initialize required values

    clus = ezcluster() # initialize for a SINGLE wafer, should parallelize when possible

    # split histograms (automatically done over all histograms)
    
    array_length = clus.histograms.shape[1]
    smooth_q_background_setting = 20 * (array_length-1)/800 # smooth q value of 20 is calibrated to an array of length 800
    clus.split_histogram(smooth_q_background_setting)
    
    # find peaks (parallelized over all histograms)
    clus.find_peaks_parallelized_wrapper()

    # plot patterns (parallelized over all histograms)
    clus.plot_split_data(save_dir='plots')

    # calculate features
    clus.calculate_features()

    # scale features
    clus.scale_features()

    # initial clustering
    clus.perform_clustering()

    # for i in tqdm(range(10),desc='Clustering Iterations'): # do four iterations underclustering analysis and reclustering
    #     get_cluster_similarities(self,plots)
    #     underclustering_analysis(wafers,previous_clustering_name,plots)
    #     overclustering_analysis(wafers,previous_clustering_name,plots)
    #     clustering_convergence_check(wafers,previous_clustering_name)  
    #     perform_reclustering(wafers,previous_clustering_name,clustering_name,use_pca,generate_gif,generate_tsne)
    #     previous_clustering_name = clustering_name