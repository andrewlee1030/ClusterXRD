def log_loader(cluster_dir):
    '''
    Loads an appendable and read-only version of a cluster log file.

    Args:
        cluster_dir (str): path to clusters directory
    
    Returns:
        _io.TextIOWrapper: appendable object of clusters log file
        _io.TextIOWrapper: read-only object of clusters log file
    '''
    
    log_a = open(f'{cluster_dir}/cluster_log.txt','a')
    log_r = open(f'{cluster_dir}/cluster_log.txt','r').read()

    return log_a, log_r