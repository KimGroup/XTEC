import sys 
import numpy as np

def Build_Markov_Matrix(data_inds, L_scale=1,kernel_type='local', unit_cell_shape=None, uniform_similarity = False,zero_cutoff=1e-2):
    """ Builds the Adjacency matrix (without using multiprocessors) 
    The parameters are:
        data_inds (num_data,dim): contain the  hkl indices (dim=3 or dim=2) of the preprocessed data
        L_scale: lengthscale for local correlations
        Kernel_type= 'local' / 'periodic' : indicating whether correlations are local, or periodic (with unit_cell_shape) 
        unit_cell_shape: an array that gives the size of unit cell (integer lengths in unit of number of pixels)
                          (only needed for periodic kernel)
        uniform_similarity= True/False: if True, sets all nonzero elements of Markov matrix (above zero_cutoff) to 1 (with 
                                     a normalization factor)
        zero_cutoff =1e-2 sets the cutoff to select nonzero elements of Markov matrix
        
        
    Returns Markov_matrix.
    
    To implement label_smoothing with periodic kernel, the following snippet shows how:
        data_inds = threshold.ind_high_std_dev   # hkl indices of the thresholded data from class Threshold_Background
        unit_cell_shape = np.array([20,20])      # In this case, a 2D lattice with size of BZ in units of pixels 
        L_scale = .2                             
        kernel_type = 'periodic'
        uniform_similarity = True
        Markov_matrix = Build_Markov_Matrix(data_inds, L_scale,kernel_type, unit_cell_shape, uniform_similarity)
        clusterGMM.RunEM(True, Markov_matrix,1)  # will introduce 1 Markov_matrix between E and M steps to diffuse cluster probabilities.

    """    
    print("\n\tBuilding Adjacency Matrix, 1cpu, ...")
    import time
    start_time = time.time()
    Markov_data = []
    row_all     = []
    col_all     = []

    for i in range(data_inds.shape[0]):
       if kernel_type == 'local':
          Markov_vec = np.exp(-np.sum((data_inds[i]-data_inds)**2/L_scale**2,axis=1)) # Marcov_vec[j] = exp[-Sum_{x=1,2,3}(qx_i-qx_j)^2/L] 
       elif kernel_type == 'periodic':
          Markov_vec =  np.exp(-np.sum(np.sin(np.pi/unit_cell_shape*(data_inds[i]-data_inds))**2/L_scale**2,axis=1)) 
                           # Marcov_vec[j] =exp[-Sum_{x=1,2,3}sin[k_x(qx_i-qx_j)]^2/L] where 2k_x=2pi/Lx is R basis vec, shape=(num_data)
       else:
          print("Error: Invalid kernel type")
          sys.exit()
        
       row_ind = np.where(Markov_vec > zero_cutoff)      # a cutoff to mask zeros
       col_ind = np.full(row_ind[0].shape,i)      
       if uniform_similarity is False: 
          Markov_data.append(Markov_vec[row_ind])
       else:
          Markov_data.append(np.ones(Markov_vec[row_ind].shape))     # sets 1 to all non-zero elements of Markov vec
       row_all.append(row_ind[0])
       col_all.append(col_ind)

    Markov_data = np.concatenate( Markov_data, axis=0 )
    row_all     = np.concatenate( row_all, axis=0 )
    col_all     = np.concatenate( col_all, axis=0 )

    from scipy.sparse import csr_matrix
    Markov_matrix = csr_matrix((Markov_data, (row_all, col_all)),shape=(data_inds.shape[0],data_inds.shape[0]))

    from sklearn.preprocessing import normalize
    Markov_matrix = normalize(Markov_matrix, norm='l1', axis=1)

    print("\tFinished Building Adjacency Matrix in", time.time() - start_time, "s \n")
   
    return Markov_matrix

# ....................................................................................................................

# below is a faster version for building local kernel 
def Build_Graph_for_Label_Smoothing_Fast_Local(data_inds, full_data_shape, neighbors, parallel=False):
   

   print("\n\tBuilding Adjacency Matrix (Fast, Local)...")
   import time
   start_time = time.time()

   #Markov_data = []
   row_all     = []
   col_all     = []


   
   # PARALLEL IMPLEMENTATION pooling over neighbors
   # N.B. the overhead for pooling is large since it must copy all relevant data and programs (thanks global interpreter lock)
   # So only use the parallel flag if the overhead is small relative to the cost of the unparallelized version (the overhead should scale linearly w/ the data size while the cost should scale quadratically)
   if parallel is True:
      import multiprocessing as mp
      from functools import partial

      # Divvy up the neighbors
      start  = 0
      nprocs = mp.cpu_count() - 2 # Leave 2 procs free
      min_num_per_set = int(np.floor(len(neighbors)/nprocs))
      all_subsets = []
      for i in range(nprocs):
          if i < (len(neighbors)-min_num_per_set*nprocs):
              step = min_num_per_set + 1
          else:
              step = min_num_per_set
          subset = neighbors[start:start+step]
          all_subsets.append(subset)
          start = start+step

      start_time_pool = time.time()         
      Get_Neighbors_Parallel_Bound = partial(Get_Neighbors, full_data_shape, data_inds)
      pool = mp.Pool(nprocs)
      all_neighbors = pool.map(Get_Neighbors_Parallel_Bound,all_subsets)
      print("Pooling Time:", time.time() - start_time_pool)

      start_time_rowcol_all = time.time()
      for neighbor_out in all_neighbors:
         col_all.extend(neighbor_out[0])
         row_all.extend(neighbor_out[1])
      print("Building row/col all:",time.time() - start_time_rowcol_all)

   else:
      col_all, row_all = Get_Neighbors(full_data_shape, data_inds, neighbors)

   # Could add to Markov_data above in order to include distance
   # Currently all neighbors are weighted equally
   Markov_data = np.ones(len(row_all))

   from scipy.sparse import csr_matrix
   Markov_matrix = csr_matrix((Markov_data, (row_all, col_all)),shape=(data_inds.shape[0],data_inds.shape[0]))
   
   from sklearn.preprocessing import normalize
   Markov_matrix = normalize(Markov_matrix, norm='l1', axis=1)

   print("\tFinished Building Adjacency Matrix in", time.time() - start_time, "s \n")
   
   return Markov_matrix


def Get_Neighbors(full_data_shape_, data_inds_, neighbors):
   data_ind_local = np.arange(data_inds_.shape[0])
   padding = max(abs(np.array(neighbors).max()),abs(np.array(neighbors).min()))
   full_data_shape_ = tuple([el+2*padding for el in full_data_shape_])
   data_inds_ += padding # pad data inds to handle case where neighbor is outside dataset
   raveled_ind  = np.ravel_multi_index(data_inds_.transpose(), full_data_shape_)
   hash_raveled = set(raveled_ind) # hash all raveled thresholded global indices
   map_global_to_local = dict(zip(raveled_ind,data_ind_local))
   
   ind_col_local_subset = []
   ind_row_local_subset = []
   
   for neighbor in neighbors:
      data_shift = data_inds_ + neighbor
      raveled_ind_shifted = np.ravel_multi_index(data_shift.transpose(), full_data_shape_)
      ind_lookup_mask = np.array([ind in hash_raveled for ind in raveled_ind_shifted])
      ind_col_global = raveled_ind[ind_lookup_mask]
      ind_row_global = raveled_ind_shifted[ind_lookup_mask]
      ind_col_local  = [map_global_to_local[ind] for ind in ind_col_global]
      ind_row_local  = [map_global_to_local[ind] for ind in ind_row_global]
      ind_col_local_subset.extend(ind_col_local)
      ind_row_local_subset.extend(ind_row_local)

   data_inds_ -= padding
   
   return ind_col_local_subset, ind_row_local_subset


# ........................................................................................................................
