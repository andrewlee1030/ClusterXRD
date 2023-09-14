# ClusterXRD

ClusterXRD is a tool designed to cluster visually-similar XRD histograms and filter out amorphous ones. The goal is to help the user quickly identify crystalline (single or multi phase) histograms.

This tool is designed for use with XRD histograms that have high amounts of noise, peak shifting, and artefacts. Minimal to no pre-processing is required to use this tool. 

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

### Quick start

Import the module, define required variables, and call the ezCluster function.

    from ClusterXRD import *

    k_clusters = 4 # number of intial clusters
    input_dir = 'path/to/your/histograms'
    filename_suffix = '.csv'
    n_clustering_rounds = 10 # max number of clustering loops to get optimal number of clusters

    ezCluster(k_clusters,input_dir,filename_suffix,n_clustering_rounds)


### More advanced use cases

You can adjust how the clustering tool works on a more granular level by adjusting each function defined in the **clusterXRD** class within **clusterXRD.py**.

