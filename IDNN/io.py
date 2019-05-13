import numpy as np
from time import time

def print_dist2csv(fname, dist):
    idx = np.triu_indices(dist.shape[0], k=1)
    dist_file = np.vstack(( idx[0] + 1 ,  idx[1] + 1 , dist[idx])).T
    tic = time()
    np.savetxt(fname, dist_file, fmt='%d %d %f')
    print('Done in : ' + str( time() - tic)  + ' seconds')