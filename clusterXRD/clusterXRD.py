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


class clusterXRD():
    '''
    Should be initialized once per wafer/set of XRD histograms to be analyzed. Contains individual functions that can be used to carry out clustering
    '''
    def __init__(self,
                 k_clusters=4,
                 input_dir=None,
                 filename_suffix='_1D.csv',
                 features=None,
                 split_histogram_dir=None,
                 peak_dir=None,
                 plot_dir=None,
                 features_dir=None):
        '''
        Initializes data and variables for clustering on a given histogram.

        Args:
            k_clusters (int): number of clusters for first iteration of clustering
            input_dir (str): directory of raw histograms
            filename_suffix (str): common filename suffix for all raw histogram data files
            features ()
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

    def find_peaks_parallelized_wrapper(self,
                                        save_dir='peaks'):
        '''
        Wrapper function to run the 'find_peaks' function in parallel for all histograms within a wafer. Identifies peak starts, locations, and ends for histograms.

        Args:
            save_dir (str): directory name for peak information to be saved to
        
        Returns:
            None: writes peaks to local files
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
            multiprocess_plotting_inputs += [[histogram_intensities,histogram_filename,self.peak_dir]]
        
        p.starmap(find_peaks, multiprocess_plotting_inputs)
        print(f'Finished finding peaks for {self.input_dir}')

    def split_histograms(self,
                        smooth_q_background_setting,
                        save_dir = 'split_histograms',
                        save=True):
        
        '''
        Splits raw histograms into the crystalline and background components

        Args:
            smooth_q_background_setting (float): hyperparameter for background separation
            save_dir (str), default 'split_histograms' : directory name to save split histogram files into

        Returns:
            None: saves split histogram data into local files
        
        '''

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

    def plot_split_data(self,
                        save_dir='plots'):
        '''
        Plot the split XRD histograms with peak information.

        Args:
            save_dir (str), default 'plots': directory name for plots to be saved into

        Returns:
            None: this function calls the plotting function which saves plots locally
        '''
        
        if self.plot_dir != None: return None # if True, then plots have already been generated

        n_parallel_cpus = multiprocessing.cpu_count() - 2 # reserve 2 cores
        p = multiprocessing.Pool(n_parallel_cpus)

        multiprocess_plotting_inputs = []
        
        # make directory for plots
        self.plot_dir = f'{self.input_dir}/{save_dir}'
        os.makedirs(self.plot_dir,exist_ok=True)

        # set up multiprocessing inputs
        for filename in self.histogram_filenames:
            multiprocess_plotting_inputs += [[filename,self.input_dir, self.split_histogram_dir, self.peak_dir, self.plot_dir, self.filename_suffix]]

        # execute multiprocessing
        p.starmap(plot_xrd_with_peaks, multiprocess_plotting_inputs)
        print(f'Finished plotting histograms for {self.input_dir}')

    def calculate_features(self):
        '''
        Calculates features to represent histograms for clustering.
        '''

        if self.features_dir != None: return None # if True, then features have already been calculated

        self.features_dir = f'{self.input_dir}/features'
        os.makedirs(self.features_dir,exist_ok=True)

        peak_feats = [] # list of dictionaries
        feat_stats = []

        for i in range(len(self.histogram_filenames)): # for each histogram
            try:
                # load histogram data
                filename = self.histogram_filenames[i]
                raw_intensities = self.histograms[i]
                q_data = self.q_data[i]
                peak_data = pd.read_csv(f'{self.peak_dir}/{filename}_peaks.csv', index_col=0)
                background_data = np.genfromtxt(f'{self.split_histogram_dir}/{filename}_background')
                crystal_data = np.genfromtxt(f'{self.split_histogram_dir}/{filename}_crystalline')

                # calculate some properties
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

            ################## FEATURES ARE GENERATED BELOW ##################

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


                # construct dict for writing feature csv
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
        
            except:
                print(f'Error with filename: {filename}, this pattern has {len(peak_data)} peaks.')
                pass
        
        # write all histogram features into csv
        feat_stats_df = pd.DataFrame(feat_stats)
        feat_stats_df.to_csv(f'{self.features_dir}/features_original.csv',index=False)
        print(f'Finished calculating features for {self.input_dir}')

    def scale_features(self):
        '''
        Scales features with min-max scaler and applies PCA
        '''
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

    def perform_clustering(self,
                           use_pca=True,
                           gifs=True,
                           max_iter=300,
                           tol=1e4,
                           clustering_name=0):
        '''
        Performs clustering on histograms.

        Args:
            use_pca (bool), default True: use PCA-transformed features
            gifs (bool), default True: save gifs of clustered plots
            max_iter (int), default 300: maximum number of iterations for the k-means algorithm
            tol (float), default 1e4: Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
            clustering_name (int), default 0: nth iteration of clustering
        '''
        
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

        ###### perform clustering
        if len(X_clustering) < self.k_clusters: self.k_clusters = len(X_clustering)
        kmeans = KMeans(n_clusters=self.k_clusters,max_iter=max_iter,tol=tol,n_init=1).fit(X_clustering)
        cluster_labels[np.nonzero(~is_amorphous)] = kmeans.labels_
    
        ###### save plots for non amorphous xrd patterns
        save_kmeans_plots(kmeans, self.cluster_results_dir, self.plot_dir, pattern_names[~is_amorphous].to_numpy(), gifs,self.gif_dir)

        ###### Pickle Clustering object

        with open(f'{self.cluster_results_dir}/kmeans.obj','wb+') as g:
            pickle.dump(kmeans,g)
            g.close()

        ###### Save cluster labels with feature information
        assert(-2 not in cluster_labels)
        original_feature_df['Cluster labels'] = cluster_labels
        original_feature_df.to_csv(f'{self.features_dir}/features_original_w_labels.csv',index=False)

        ###### Write results to log file, set up variables for next round of clustering
        log.write(f'Clustering - {self.k_clusters} clusters \n')
        log.close()
        self.previous_clustering_name = clustering_name
        self.clustering_name = clustering_name + 1
        print(f'Finished clustering {self.previous_clustering_name}')

    def get_cluster_similarities(self):
        '''
        Calculates intra-cluster histogram similarities. Similarity matrices are written locally. 
        '''
        
        log_a, log_r = log_loader(self.cluster_dir)
        
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

    def underclustering_analysis(self,
                                 plots=False):
        '''
        Determines how many more clusters are needed for the next round of clustering.
        '''
        
        additional_clusters_needed = 0

        log_a, log_r = log_loader(self.cluster_dir)

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

            cluster = int(similarity_score_file_path.split('/')[-1].split('_')[0])

            cluster_features = features.query(f'`Cluster labels` == {cluster}')

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
                underclustering_plots(similar_pair_pca_scaled_feature_distances,dissimilar_pair_pca_scaled_feature_distances,cluster,percent_dissimilar_over_threshold,self.post_clustering_dir)
            

        underclustering_data = pd.DataFrame(data = {'Current cluster count': [len(similarity_score_file_paths)],
                                                    'Additional clusters needed': [additional_clusters_needed]})
            
        underclustering_data.to_csv(f'{self.post_clustering_dir}/underclustering_statistics.csv',index=False)
        
        log_a.write(f'Underclustering analysis - need {additional_clusters_needed} more clusters \n')
        log_a.close()
        print(f'Finished underclustering analysis.')

    def overclustering_analysis(self,
                                plots=False,
                                min_distance_frac=0.75): # clustering name needs to be the most recent run (highest number)
        '''
        Determines how many fewer clustes are needed for the next round of clustering.
        '''
        
        log_a, log_r = log_loader(self.cluster_dir)

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
                intra_cluster_distances = calculate_intra_cluster_distances(pca_transformed_scaled_features,non_amorphous_features,cluster)
                inter_cluster_distances = np.delete(calculate_inter_cluster_centroid_distances(kmeans,cluster),[cluster])

                if len(intra_cluster_distances) == 0: continue

                cutoff = min_distance_frac * np.max(intra_cluster_distances)

                n_under_cutoff = np.sum(inter_cluster_distances < cutoff)
                if n_under_cutoff > (len(all_clusters)-1)/2: n_clusters_to_reduce += 1
                
                if plots == True:
                    overclustering_plots(intra_cluster_distances,cluster,n_under_cutoff,inter_cluster_distances,self.post_clustering_dir)
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
        '''
        Reads in the new number of clusters needed and executes next round of clustering.
        '''

        log_a, log_r = log_loader(self.cluster_dir)

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

def ezCluster(k_clusters,
              input_dir,
              filename_suffix,
              n_clustering_rounds):
    '''
    Call this to initiate the full clustering loop for a single wafer/set of histograms to cluster.

    Args:
        k_clusters (int): the number of clusters in the intial round of clustering
        input_dir (str): path to the XRD histograms for clustering
        filename_suffix (str): common suffix at the end of each histogram filename
        n_clustering_rounds (int): max number of clustering rounds to perform
    
    Returns:
        None: Will place clusters in a directory within the input_dir
    '''
    
    clus = clusterXRD(k_clusters=k_clusters,input_dir=input_dir,filename_suffix=filename_suffix)
    
    array_length = clus.histograms.shape[1]
    smooth_q_background_setting = 20 * (array_length-1)/800 # smooth q value of 20 is calibrated to an array of length 800
    
    clus.split_histograms(smooth_q_background_setting)

    clus.find_peaks_parallelized_wrapper()
    
    clus.plot_split_data(save_dir='plots')

    clus.calculate_features()

    clus.scale_features()

    clus.perform_clustering()

    for i in range(n_clustering_rounds):

        clus.get_cluster_similarities()

        clus.underclustering_analysis()

        clus.overclustering_analysis()

        clustering_convergence_check(clus.post_clustering_dir,clus.cluster_dir)

        clus.perform_reclustering()