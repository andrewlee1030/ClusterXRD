# ClusterXRD

ClusterXRD is a tool designed to cluster visually-similar XRD histograms and filter out amorphous ones. The goal is to help the user quickly identify crystalline (single or multi phase) histograms.

This tool is designed for use with XRD histograms that have high amounts of noise, peak shifting, and artefacts. Minimal to no pre-processing should be required to use this tool. Furthermore, the code here works off of common dependencies, so installation should be straightforward. Functions are parallelized where possible, meaning this tool can be feasibly applied on large amounts of histograms.

### How this tool works

This tool performs the following tasks in order:

  1. Split raw histogram intensities into crystalline and background components
  2. Identify peaks within the crystalline components
  3. Generate features based on hisogram intensities and identified peaks
  4. Cluster histograms (user specifies # of clusters for initial round of clustering)
  5. Assess how many additional/fewer clusters are needed
  6. Repeat steps 4 and 5 until either there are no additional/fewer clusters OR the maximum number of iterations has been reached 

Once the tool has finished running, there will be additional directories alongside the original raw histograms. These additional directories

### Installation
Run the following command using Python 3.11.0 or higher

    python setup.py install


### Test the installation

Run the following commands from this directory

    cd tests
    tar -xvf cluster_test_files/sample_xrd_wafer.tar.gz -C cluster_test_files
    python test.py

### Basic usage

Import the module, define required variables, and call the ezCluster function.

    from ClusterXRD import *

    k_clusters = 4 # number of intial clusters
    input_dir = 'path/to/your/histograms'
    filename_suffix = '.csv' # trailing string that is shared among all histogram filenames
    n_clustering_rounds = 10 # max number of clustering iterations

    ezCluster(k_clusters,input_dir,filename_suffix,n_clustering_rounds)


### Advanced usage

You can adjust clustering behavior on a more granular level by modifying the functions defined in the **clusterXRD** class within **clusterXRD.py**.

The only requirements are for the following functions to be called in order:

 1. Initialize the clusterXRD class
 2. split_histograms
 3. find_peaks / find_peaks_parallelized_wrapper
 4. plot_split_data
 5. calculate_features
 6. scale_features
 7. perform_clustering
 8. get_cluster_similarities
 9. underclustering_analysis
 10. overclustering_analysis
 11. clustering_convergence_check
 12. repeat steps 7 through 11 in order as many times as needed



