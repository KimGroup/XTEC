import sys
import numpy as np

# ##############################################################################################################################    
r"""
class GMM performes Step-wise EM (cf. Liang and Klein 2009) to cluster data(num_sam, num_T). 
An example to cluster with GMM without label smoothing:

    num_clusters = 2
    clusterGMM=GMM(data, num_clusters)  # Sets stage for clustering data in to 2 clusters. data.shape=(num_sam, num_T)
    # clusterGMM = GMM(data,num_clusters,"diagonal",1,0.7,1e-5,50,500,False) #These are default values for other parameters
    clusterGMM.RunEM()                  # to run EM algorithm for clustering data without label smoothing
    
    print(clusterGMM.num_per_cluster)
    cluster_assignments = clusterGMM.cluster_assignments  #cluster label in range(num_clusters) of each sample. dim=num_sam.
    clusterGMM.Plot_Cluster_Results_traj(Temp,False)    # to plot the cluster trajectory mean and variance. 
                                                        # False plots only GMM cluster results, 
                                                        # True plots all the data trajectories color coded by the clustering assignment  
    
Example to cluster with label smoothing and periodic kernel (see GMM_kernel.py for constructing Markov matrix):

    # Generate Markov matrix to be used between E and M step to diffuse cluster probabilities with neighbouring data
    
    data_inds = threshold.ind_high_std_dev   # hkl indices of the thresholded data from class Threshold_Background (see Preprocessing.py)
    L_scale = .2                             # scale for the correlations                
    kernel_type = 'periodic'   
    unit_cell_shape = np.array([20,20])      # In this case, a 2D lattice with size of BZ in units of pixels (needed only for periodic kernel) 
    uniform_similarity = True  # sets all nonzero elements of Markov matrix (above cutoff 1e-2) to 1 (with the normalization factor)
    Markov_matrix = Build_Markov_Matrix(data_inds, L_scale,kernel_type,unit_cell_shape,uniform_similarity)
    
    # GMM clustering with label smoothing
    num_clusters = 2
    clusterGMM=GMM(data, num_clusters)  # Sets parameters for clustering data in to 2 clusters. data.shape=(num_sam, num_T)
    clusterGMM.RunEM(True, Markov_matrix,1)  # will introduce Markov_matrix (1 times) between E and M steps. 


    print(clusterGMM.num_per_cluster)
    cluster_assignments = clusterGMM.cluster_assignments
    clusterGMM.Plot_Cluster_Results_traj(Temp,False)

.....................................................................    
    
The parameters of GMM are:

data        : Assume the data is handed in so that the first axis corresponds to the number of samples (num_sam), and the second 
              to num of temperatures (num_T)
num_clusters: number of clusters (K)
cov_type    : = "diagonal" or "full", whether to keep only diagonal elements or retain full cov matrix. 
batch_num   : number of batches. Algorithm has a "batch" phase, followed by a "full" phase that operates on the entire dataset.
alpha       : decay exponent of i'th step-size eta_i = (i + 2 )^{- alpha}, 0.5 < alpha < 1.
tol         : tolerance for the convergence of loglikelihood.
max_batch_epoch: max number of batch iterations.
max_full_epoch : max number of iterations on full dataset.
verbose     : True/False on whether to print loglikelihood at each iteration.


RunEM(label_smoothing_flag,Markov_matrix,smoothing_iterations) - performs stepwise EM (cf. Liang and Klein 2009) 

Parameters of RunEM are: 

label_smoothing_flag   : if True implements label smoothing between E and M step.
Markov_matrix          : Adjacency matrix for label smoothing (when smoothing_flag=True). Use GMM_kernels.py to 
                            construct Markov matrix with local or periodic kernel.
smoothing_iterations   : number of times Markov_matrix is applied to cluster prob between E and M step.
   
Key attributes after GMM clustering:

       self.cluster[k].mean: cluster mean trajectory (dim=num_T) of the k-th cluster
       self.cluster[k].cov : cluster covariance [dim=num_T for diagonal, (num_T,num_T) for full] of the k-th cluster
       self.cluster_assignments : cluster assignment k in range(num_clusters), of each sample. dim=num_sam
       self.num_per_cluster : number of samples in each cluster.
       self.cluster_probs   : cluster probablity of data points, dim=(num_clusters, num_data) 
       self.cluster_weights : mixing weights of each cluster, dim=(num_clusters)
       
   """ 

class GMM:
   def __init__(self, data, cluster_num, cov_type = "diagonal", batch_num = 1, alpha = 0.7, tol = 1e-5, max_batch_epoch = 50, max_full_epoch = 500,verbose = False):
      self.cluster         = []
      self.cluster_num     = cluster_num              
      self.cov_type        = cov_type                 
      if self.cov_type != "full" and self.cov_type != "diagonal":
         print("Error: cov_type must be full or diagonal"); sys.exit()
      self.verbose         = verbose                  
      self.mixing_weights  = (1.0/self.cluster_num)*np.ones(self.cluster_num)    # starting with equal mixing weights: w_k=1/K
      self.batch_num       = batch_num                
      self.tol             = tol                     
      self.max_batch_epoch = max_batch_epoch
      self.max_full_epoch  = max_full_epoch
      self.epoch           = 0
   
      # For setting up evenly distributed batch sizes for stepwise EM 
      self.batch_base_num = int(np.floor(data.shape[0]/self.batch_num))   # divide data (num_data) in to batch_num batches
      self.batch_mod_num  = data.shape[0]%self.batch_num                  # the remaining data after dividing into batches 
      if self.batch_base_num ==0: self.batch_num = self.batch_mod_num     # handle case when given batch_num > sample_num

      self.alpha          = alpha       # decay exponent for i'th step-size eta_i = (i + 2 )^{- alpha}, 0.5 < alpha <1
      self.data_rand      = data.copy() # make deep copy for easy randomization
      self.data           = data.copy() 

      self.cluster_probs = None          

      for k in range(self.cluster_num):
          new_cluster = Cluster_Gaussian(data)   #  mean_guess and cov from Gaussian with mean and cov of the data. 
          self.cluster.append(new_cluster)       # list of K Cluster_Gaussian initially with diff means (mean_guess), and same cov 

      if self.cov_type == "diagonal":            # keep only diagonal elements of cov, discard the off-diag terms 
         for k in range(self.cluster_num):
            self.cluster[k].cov = np.diag(self.cluster[k].cov)

      self.means = np.array( [ self.cluster[k].mean for k in range(self.cluster_num)] )  # cluster means, shape=(cluster_num,num_T)
      self.covs  = np.array( [ self.cluster[k].cov  for k in range(self.cluster_num)] )  # cluster covs

   # ...................................................................................................    

   def RunEM(self, label_smoothing_flag = False, Markov_matrix = None, smoothing_iterations = 1):
      loglikelihood_diff =  1e6
      loglikelihood_new  = -1e6

      while loglikelihood_diff > self.tol and self.epoch < self.max_batch_epoch:
         loglikelihood_old = loglikelihood_new
         if self.batch_num != 1:              # stepwise EM applies only when batch_num>1, standard EM needs no data randomization 
            np.random.shuffle(self.data_rand) # randomize the (deep copied) data

         left_bound  = 0    # batch data is the shuffled data[left_bound:right_bound]
         right_bound = 0
         for batch_val in range(self.batch_num):
            right_bound += self.batch_base_num + (1 if batch_val<self.batch_mod_num else 0)  
            data_batch = self.data_rand[left_bound:right_bound]     #first batch_mod_num batches accomodate the batch_mod data            
            left_bound  = right_bound
            self.E_Step(data_batch)
            self.M_Step(data_batch)          

         loglikelihood_new  = self.LogLikelihood(self.data)                  
         loglikelihood_diff = np.abs(loglikelihood_new - loglikelihood_old)
         if self.verbose is True:
            print("Batch Log-likelihood:",loglikelihood_new)
         self.epoch += 1                 # number of steps for stepwise EM to converge 

      if label_smoothing_flag == True:
         self.Markov_matrix = Markov_matrix
          
      # Finish on all data (guaranteed one "full" pass)
      loglikelihood_diff = 1e6
      loglikelihood_new  = -1e6
      batch_epoch = self.epoch     
      self.converged = None
      while loglikelihood_diff > self.tol and self.epoch < batch_epoch + self.max_full_epoch + 1:
         loglikelihood_old = loglikelihood_new
         self.E_Step(self.data)
         if label_smoothing_flag == True:
            self.Smooth_Labels(smoothing_iterations)
         self.M_Step(self.data)
         loglikelihood_new  = self.LogLikelihood(self.data)
         loglikelihood_diff = np.abs(loglikelihood_new - loglikelihood_old) # label smoothing can decrease the log-likelihood
         self.loglikelihood = loglikelihood_new
         if self.verbose is True:
            print("Full  Log-likelihood:",self.loglikelihood)
         self.epoch += 1
   
         if self.epoch == batch_epoch + self.max_full_epoch + 1:
            self.converged = False
         if loglikelihood_diff <= self.tol:
            self.converged = True
      
      # Now the clustering is complete 
      # update self.cluster[k].mean and self.cluster[k].cov
      for k in range(self.cluster_num):
         self.cluster[k].mean = self.means[k]
         self.cluster[k].cov  = self.covs[k]

      self.cluster_assignments     = np.argmax(self.cluster_probs,axis=0)   # cluster assignment of each sample,  dim=(num_sam) 
    
      self.num_per_cluster         = [ np.sum(self.cluster_assignments == k)\
                                      for k in range(self.cluster_num)]     # number of samples within each cluster
      self.discovery_cluster_ind   = np.argmax(np.array([0. if self.num_per_cluster[k] < 2 else np.std(self.means[k])\
                                                         for k in range(self.cluster_num)]))

   # ...................... define functions needed to RunEM............................................  

   def E_Step(self, data_):                                   # data_.shape=(num_data, num_T), num_data is sample num in the batch
      log_gaussian = self.LogGaussian(data_)                  # log[N(data|means(k),cov(k))], dim=(num_cluster, num_data)
      self.cluster_probs = self.mixing_weights[:,np.newaxis]*self.Logp2p(log_gaussian)  #  cluster_prob, Ck
      self.cluster_probs /= np.sum(self.cluster_probs,axis=0)[np.newaxis,:]      # Ck = w_k*N(data|means(k),cov(k))/[sum_k w_k*N(data|..)]
      
   def M_Step(self, data_):
      self.mixing_weights = np.mean(self.cluster_probs, axis=1)  # w_k=<Ck> averaged over data, dim=(num_clusters)
      weight_mask = (self.mixing_weights == 0)
      self.mixing_weights[weight_mask] = 1e-10            # Handles potential division by 0 below
      self.means_new = 1.0/(data_.shape[0]*self.mixing_weights[:,np.newaxis])*(self.cluster_probs@data_) 
                                                                                        # new means, dim=(num_clusters, num_T)
      
      if self.cov_type == "full": # Often convergence issues with this
         self.covs_new   = 1.0/(data_.shape[0]*self.mixing_weights[:,np.newaxis,np.newaxis])*\
            np.array([ data_no_mean.T@(self.cluster_probs[k][:,np.newaxis]*data_no_mean) \
                      for k in range(self.cluster_num) for data_no_mean in [data_ - self.means_new[k][np.newaxis,:]] ])
      elif self.cov_type == "diagonal":
         self.covs_new   = 1.0/(data_.shape[0]*self.mixing_weights[:,np.newaxis])*\
            np.array([ np.einsum('ij,ij->i',data_no_mean.T,(self.cluster_probs[k][:,np.newaxis]*data_no_mean).T) \
                      for k in range(self.cluster_num) for data_no_mean in [data_ - self.means_new[k][np.newaxis,:]] ])
      
      # Step-wise EM update
      eta_k      = (self.epoch + 2)**(-self.alpha)            # stepsize
      self.means = (1-eta_k)*self.means + eta_k*self.means_new
      self.covs  = (1-eta_k)*self.covs  + eta_k*self.covs_new    
                                                                 
    
   def LogGaussian(self, data_):
      # map data matrix with shape (num_sam, dim) to cluster log probabilities w/ dim (num_clusters, num_sam)
      log_gaussian = None
      if self.cov_type == "full":
         log_gaussian = np.array([ -0.5*self.cluster_num*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(self.covs[k])) \
                                  - 0.5*np.einsum('ij,ij->i', np.dot(data_no_mean, np.linalg.inv(self.covs[k])), data_no_mean) \
                                  for k in range(self.cluster_num) for data_no_mean in [data_ - self.means[k][np.newaxis,:]] ])
      elif self.cov_type == "diagonal":
         log_gaussian = np.array([ -0.5*self.cluster_num*np.log(2*np.pi) - 0.5*np.log(np.prod(self.covs[k]))\
                                  - 0.5*np.einsum('ij,ij->i', data_no_mean*(1.0/self.covs[k])[np.newaxis,:],data_no_mean)\
                                  for k in range(self.cluster_num) for data_no_mean in [data_ - self.means[k][np.newaxis,:]] ])
      return log_gaussian    # array of log[N(data|means(k),cov(k))],   shape=(num_cluster, num_sam)

   def Logp2p(self, log_p): # convert log(P_k) to P_k/(\sum_k P_k): Log-sum-exp trick for avoiding numerical overflow
      max_logp = np.max(log_p,axis=0)[np.newaxis,:]            # log_p shape= (num_cluster,num_sam)
      return np.exp(log_p - max_logp - np.log(np.sum(np.exp(log_p-max_logp),axis=0) ))  #returns P_k/(\sum_k P_k)

   def LogLikelihood(self,data_):    # return Sum_data{Log[Sum_k w_k N(data|means(k),cov(k))]}
      log_gaussian = self.LogGaussian(data_) #  log[N(data|means(k),cov(k))],  shape=(num_cluster, num_sam)
      return np.sum(self.LogSumTrick(np.log(self.mixing_weights[:,np.newaxis]) + log_gaussian))  

  
   def LogSumTrick(self, log_p): # log_p.shape=(num_cluster,num_sam). Another log-sum trick, used for evaluating log-likelihood
      max_logp = np.max(log_p,axis=0)
      return np.log(np.sum(np.exp(log_p - max_logp[np.newaxis,:]),axis=0)) + max_logp # returns log[sum_k P_k], shape=(num_sam)
 

   def LogLikelihood_lowerbound(self,data_):
      log_gaussian = self.LogGaussian(data_)
      return np.sum(self.cluster_probs*log_gaussian + self.xlogy(self.cluster_probs,self.mixing_weights[:,np.newaxis])\
                    - self.xlogy(self.cluster_probs,self.cluster_probs))
   
        
   def xlogy(self, x, y):
      mask = x > 0.0
      return np.where(mask, x*np.log(y, where=mask), 0.0)

   def Smooth_Labels(self, it_num = 1):
      for i in range(it_num):
         self.cluster_probs = (self.Markov_matrix @ self.cluster_probs.transpose()).transpose()   # "diffused" cluster prob
            
   # ....................... define functions for Plotting results ...........................................
   """ 
   Following functions plot the clustering results. 
   
   self.Plot_Cluster_Results_traj(x_train, traj_flag,data_means): 
       Plots the trajectories and cluster means +- 1*variance.  
       x_train    : Temperatures, dim=(num_T)
       traj_flag  : if True plots the trajectories  color coded by the clustering label.
       data_means : if not None, will shift the cluster_means. dim=(num_data), gives the value each traj needs to be offset. 
                    useful to put back actual data mean, which might have been subtracted before clustering.   

       Plotting for only less than 7 clusters currently enabled.  
   """

   def Plot_Cluster_Results_traj(self, x_train, traj_flag = True,data_means = None):
      import matplotlib.pyplot as plt
      if self.cluster_num > 7:
         print('Error: Plotting for only less than 7 clusters currently enabled')
      else:
         color_list = ['red', 'blue', 'green', 'purple', 'yellow', 'orange', 'pink']

         if traj_flag == True:
            plt.figure()
            for i in range(self.data.transpose().shape[1]):
               plt.plot(x_train,self.data.transpose()[:,i],color=color_list[self.cluster_assignments[i]])
 
            for i in range(self.cluster_num):
               plt.plot(x_train,self.cluster[i].mean,'k--', lw=2);
            return plt
         plt.figure()
         
         std_dev_num = 1
         if data_means is None:
            self.traj_means = [self.cluster[k].mean for k in range(self.cluster_num)]
         else:
            mean_shift = 1.0/(self.data.shape[0]*self.mixing_weights)*(self.cluster_probs@data_means)
            self.traj_means = [self.cluster[k].mean + mean_shift[k] for k in range(self.cluster_num)]
 
         for i in range(self.cluster_num):
            plt.plot(x_train,self.traj_means[i],color=color_list[i], lw=2);
            plt.gca().fill_between(x_train, self.traj_means[i]-(std_dev_num*self.cluster[i].cov)**0.5, self.traj_means[i]\
                                   +(std_dev_num*self.cluster[i].cov)**0.5 , color=color_list[i],alpha=0.4)
         return plt
   """.................................................................................................... 
   
   Plot_Cluster_kspace_2D_slice(threshold,figsize_,cluster_list,data_ind,slice_ind,axis_) :Plots the 2D image slice, with each (thresholded)
   pixel color coded by the clustering label, or colored grey if not clustered. 
   Example to plot 2D clustering slice for say pixels near Bragg peaks which have been isolated and stored as 
   Bragg_data, dim=(num_Bragg_data,num_T) gives intensities, and Bragg_ind, dim=(num_Bragg_data,2) stores the (h,k) indices of Bragg_data
   
        num_clusters = 3
        clusterGMM = GMM(Bragg_data,num_clusters)    
        clusterGMM.RunEM()
        
        # Now plot the '0' (Red) and '1'(Blue) clusters for the Bragg peak pixels whose h,k indices are given by Bragg_ind  
        clusterGMM.Plot_Cluster_kspace_2D_slice(threshold,figsize_=(20,20),cluster_list=[0,1],Bragg_ind)
        # this plots a figure of size (20,20). 
        
        # to plot separately (for later customization with labels, tick marks etc...)
        plt.figure(figsize=(10,10))
        plt.imshow(clusterGMM.plot_image,origin='lower',cmap=clusterGMM.plot_cmap,norm=clusterGMM.plot_norm)

       
   the input parameters are:
       threshold      : class Threshold_Background from Preprocessing
       figsize_       : size of image. If None, will not plot the image
       cluster_list   : list of cluster numbers 0,1,2 etc. to be plotted. The remaining clusters if any will show as grey.
       data_ind       : (h,k,l) or (h,k) indices of the clustered data. shape=(num_data,2) or (num_data,3)
       slice_ind      : if 3D, the index along which to take the data slice, plotting_matrix.take(slice_ind, axis=axis_)
       axis_          : axis_ of slice_ind, if data is 3D 
   
   Output attributes for ploting with imshow(),
       self.plot_image : 2D image to be plotted 
       self.plot_cmap  : color map for the clustering, matching the same color scheme of the cluster trajectories
       self.plot_norm  : the norm to be used in imshow() so that cmap matches the cluster assignment integers 
   """        
            
   def Plot_Cluster_kspace_2D_slice(self,threshold,figsize_=None, cluster_list=None,data_ind=None, slice_ind=None, axis_=None):

                              
       import matplotlib.pyplot as plt
       from matplotlib import colors
       
       if cluster_list is None:
           cluster_list=range(self.cluster_num)
      
       if data_ind is None:
            data_ind=threshold.ind_thresholded
       
       plotting_matrix = threshold.thresholded.copy()   # 2D or 3D matrix to be assigned cluster labels for each pixels 
       
       data_shape=threshold.data_shape_orig[1:]  # (num_l, num_k, num_h)
       
       
       for k in cluster_list:
           cluster_mask = (self.cluster_assignments == k)
           cluster_ind  = data_ind[cluster_mask]
           if len(data_shape)==2: 
               plotting_matrix[cluster_ind[:,0],cluster_ind[:,1]] = k+2
           elif len(data_shape)==3:
               plotting_matrix[cluster_ind[:,0],cluster_ind[:,1],cluster_ind[:,2]] = k+2
            
       color_list = ['white', 'lightgrey', 'red', 'blue', 'green', 'purple', 'yellow', 'orange', 'pink']
       cluster_cmap = colors.ListedColormap(color_list)
    
       bounds = []
       for i in range(len(color_list)+1):
           bounds.append(i-0.5)
       norm = colors.BoundaryNorm(bounds,cluster_cmap.N)
       
       if len(data_shape) == 2:
           self.plot_image=plotting_matrix
       elif len(data_shape) == 3:
           self.plot_image=plotting_matrix.take(slice_ind, axis=axis_)

        
       self.plot_norm=norm
       self.plot_cmap=cluster_cmap 
       
       if figsize_ is not None:
           plt.figure(figsize=figsize_)
           plt.imshow(self.plot_image,origin='lower',cmap=cluster_cmap,norm=norm)
           return plt
            
            
            
   """ ...............................................................................................................         
   
   Plot_Cluster_Results_kspace_3D(threshold): Plots the thresholded data in kspace (3D), with each pixel color coded by 
                                              their cluster assignment
       threshold : class Threshold_Background from Preprocessing
   """        

   def Plot_Cluster_Results_kspace_3D(self, threshold):
      import matplotlib.pyplot as plt
      from matplotlib import colors

      data_shape = threshold.data_shape_orig[1:]
      Ql_cell = np.arange(data_shape[0])/(data_shape[0]-1)
      Qk_cell = np.arange(data_shape[1])/(data_shape[1]-1)
      Qh_cell = np.arange(data_shape[2])/(data_shape[2]-1)
   
      X,Y,Z = np.meshgrid(Ql_cell,Qk_cell,Qh_cell, indexing='ij')
      X=np.reshape(X,np.prod(data_shape))
      Y=np.reshape(Y,np.prod(data_shape))
      Z=np.reshape(Z,np.prod(data_shape))
   
      low_std_dev_cluster = np.zeros(data_shape).astype(bool)
      if len(data_shape) == 2:
         low_std_dev_cluster[threshold.ind_low_std_dev[:,0],threshold.ind_low_std_dev[:,1]] = True
      elif len(data_shape) == 3:
         low_std_dev_cluster[threshold.ind_low_std_dev[:,0],threshold.ind_low_std_dev[:,1],threshold.ind_low_std_dev[:,2]] = True

      masks = [low_std_dev_cluster.reshape(np.prod(data_shape))]

      for k in range(self.cluster_num):
         cluster_mask = (self.cluster_assignments == k)
         cluster_ind  = threshold.ind_high_std_dev[cluster_mask]
         cluster_kspace_mask = np.zeros(data_shape).astype(bool)
         cluster_kspace_mask[cluster_ind[:,0],cluster_ind[:,1],cluster_ind[:,2]] = True
         masks.append(cluster_kspace_mask.reshape(np.prod(data_shape)))

      color_list = ['white', 'lightgrey', 'red', 'blue', 'green', 'purple', 'yellow', 'orange', 'pink']
      
      RGBs = [ colors.to_rgb(color) for color in color_list ]

      import matplotlib.pyplot as plt
      from mpl_toolkits.mplot3d import Axes3D
      fig = plt.figure(dpi=600)
      ax = fig.add_subplot(111, projection='3d')

 
      for mask_num, mask in enumerate(masks):
         if mask_num > 0:
            alpha = 0.3
            sval  = 20
            rgba_mask = np.append(RGBs[mask_num+1],alpha)[np.newaxis,:]
            #ax.scatter(X[mask],Y[mask],Z[mask],c=rgba_mask,marker=".",s=sval,edgecolors='none')
            ax.scatter((5*X[mask]-5),(5*Y[mask]-5),(5*Z[mask]-5),c=rgba_mask,marker=".",s=sval,edgecolors='none')

      ax.view_init(azim=50)
      #ax.grid(False)
      ax.set_xlabel('$Q_l$',fontsize=15)
      ax.set_ylabel('$Q_k$',fontsize=15)
      ax.set_zlabel('$Q_h$',fontsize=15)
      plt.tight_layout()

      #plt.savefig('/Users/jvenderley/Desktop/cluster_unit_cell_grid.png')
            
            
# ################################################################################################
"""
Cluster_Gaussian  - to be used in the class GMM : will randomly pick mean (dim= num_T) from a 
normal dist with mean and covariance  of the data=(num_sam, num_T).
num_sam are the number of samples = thresholded data points, and num_T is the number of temperatures. 
Important attributes:
    self.mean: mean guess from normal dist with sample mean (mean_init) and sample covariance, dim=num_T.
    self.cov : sample covariance (diagonal), dim = num_T.
    
"""

class Cluster_Gaussian:
   def __init__(self, data):
      self.data_dim = data.shape[1]                 # data.shape= (num_sam, num_T), data_dim=num_T
      self.mean = np.zeros((1,self.data_dim)); self.cov = np.zeros((self.data_dim,self.data_dim))

      # Initialize w/ randomly drawn means and reasonable (spherical) cov estimated from data
      mean_init = np.mean(data,axis=0)
      cov_init  = np.diag(np.var(data,axis=0))
      mean_guess = np.random.multivariate_normal(mean_init, cov_init)
      
      self.mean = mean_guess
      self.cov  = cov_init

   def pdf(self,pt):                  # returns Gaussian prob dist at pt: N(pt|mean, cov)
      return (1.0/((2*np.pi)**self.data_dim * np.linalg.det(self.cov))**0.5)*np.exp(-0.5*np.transpose(pt-self.mean)@np.linalg.inv(self.cov)@(pt-self.mean))

    