import sys
sys.path.append('../')
from ClusterXRD import *

if __name__ ==  '__main__':
    input_dir = 'cluster_test_files/cluster_1'

    ezCluster(4,input_dir,'_bkgdSub_1D.csv',5)