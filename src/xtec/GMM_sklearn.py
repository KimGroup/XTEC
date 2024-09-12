import sys

import numpy as np
from sklearn.mixture import GaussianMixture


class GMM(object):
    """Implementation of the Gaussian Mixture Model with no label smoothing.

    Uses sklearn implementation of GMM 
    (https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) 
    to cluster data(num_data, num_T).
    
    

    Attributes
    ----------
    cluster[k].mean : array-like
        Cluster mean trajectory (dim=num_T) of the k-th cluster
    cluster[k].cov : array-like
        Cluster covariance of the k-th cluster
        [dim=num_T for diagonal, (num_T,num_T) for full]
    cluster_assignments : array-like
        Cluster assignment k in range(num_clusters), of each sample.
        dim=num_data
    num_per_cluster : array-like
        Number of samples in each cluster
    num_per_cluster : array-like
        Number of samples in each cluster
    cluster_probs : array-like
        Cluster probablity of data points, dim=(num_clusters, num_data)
    cluster_weights : array-like
        Mixing weights of each cluster, dim=(num_clusters)

    Examples
    --------
        1) An example to cluster with GMM:

        from xtec.GMM_sklearn import GMM

        num_clusters = 2
        clusterGMM=GMM(data, num_clusters)
        clusterGMM.RunEM()

        print(clusterGMM.num_per_cluster)
        cluster_assignments = clusterGMM.cluster_assignments
        clusterGMM.Plot_Cluster_Results_traj(Temp, False)
        # False plots only GMM cluster results,
        # True plots all the data trajectories color coded by the clustering
        # assignment

        

    """

    def __init__(
        self,
        data,
        cluster_num,
        cov_type="diag", 
        n_init=1, 
        tol=1e-4,
        reg_covar=1e-06,
        max_iter=100,
        init_params='kmeans', 
        weights_init=None, 
        means_init=None, 
        precisions_init=None, 
        random_state=None, 
        warm_start=False, 
        verbose=0, 
        verbose_interval=10,
        color_list=None
        
    ):
        """Initialize GMM parameters

        Parameters (mostly follows scikit-learn implementation 
        ----------
        data : array-like
            Input data. The first axis corresponds to the number of samples
            (num_data), and the second to num of temperatures (num_T)
        cluster_num : int
            Number of GMM clusters
        cov_type : "diagonal"| "full", optional
            Setting to choose whether to keep only diagonal elements or
            retain the full covariance matrix, by default "diagonal"
        
        Remaining parameters are same as in scikit learn......
        
        tol: The convergence threshold.
        reg_covar: float
            Non-negative regularization added to the diagonal of covariance. Allows to 
            assure that the covariance matrices areall positive.
        n_init : number of initializations to run. The best results are kept.
        init_params: {‘kmeans’, ‘k-means++’, ‘random’, ‘random_from_data’}, default=’kmeans’
                     The method used to initialize the weights, the means and the precisions. 
                     String must be one of:

                    ‘kmeans’ : responsibilities are initialized using kmeans.

                    ‘k-means++’ : use the k-means++ method to initialize.

                    ‘random’ : responsibilities are initialized randomly.

                    ‘random_from_data’ : initial means are randomly selected data points.
        
        
        """
        if(color_list):
            self.color_list=color_list
        else:
            self.color_list=["red",
                             "blue",
                             "green",
                             "purple",
                             "yellow",
                             "orange",
                             "pink",
                            ]
        
        self.data=data
        self.cluster_num=cluster_num
        self.cov_type=cov_type 
        self.tol=tol
        self.n_init=n_init
        self.random_state=random_state 
        
        self.GaussianMixture= GaussianMixture(
            n_components=cluster_num,
            covariance_type=cov_type,
            tol=tol, 
            reg_covar=reg_covar, 
            max_iter=max_iter, 
            n_init=n_init, 
            init_params=init_params, 
            weights_init=weights_init, 
            means_init=means_init, 
            precisions_init=precisions_init,
            random_state=random_state, 
            warm_start=warm_start,
            verbose=verbose, 
            verbose_interval=verbose_interval
        )
        
        
    def RunEM(
        self,
        label_smoothing_flag=False):
        """Performs sklearn GMM fit.

        Parameters
        ----------
        label_smoothing_flag : False
            If True returns error (no label smoothing)
        """
        
        if (label_smoothing_flag):
            print('No label smoothing in GMM sklearn, use xtec.GMM instead')
            return None
        
        self.GaussianMixture.fit(self.data)
        
        self.cluster_probs = self.GaussianMixture.predict_proba(self.data).T
        self.cluster_assignments = self.GaussianMixture.predict(self.data)
        self.means = self.GaussianMixture.means_
        self.covs = self.GaussianMixture.covariances_
        
        
        # number of samples within each cluster
        self.num_per_cluster = [
            np.sum(self.cluster_assignments == k)
            for k in range(self.cluster_num)
        ]
        
        self.cluster=[]
        for k in range(self.cluster_num):
            cluster_mean=self.means[k]
            cluster_cov=self.covs[k]
            self.cluster.append(Cluster_Gaussian(cluster_mean,cluster_cov))
    def Plot_Cluster_Results_traj(
        self, x_train, traj_flag=False, data_means=None
    ):
        """Plots the trajectories and cluster means ± 1*std.

        Parameters
        ----------
        x_train : array-like
            Temperatures, dim=(num_T)
        traj_flag : bool, optional
            If True plots the trajectories  color coded by the
            clustering label, by default False
        data_means : [type], optional
            The value each traj needs to be offset, which is useful to
            put back actual data mean, which might have been subtracted
            before clustering, by default None
        """
        import matplotlib.pyplot as plt

        if self.cluster_num > len(self.color_list):
            print("Error: cluster num larger than color list")
        else:
            color_list = self.color_list
            if traj_flag is True:
                plt.figure()
                for i in range(self.data.transpose().shape[1]):
                    plt.plot(
                        x_train,
                        self.data.transpose()[:, i],
                        color=color_list[self.cluster_assignments[i]],
                    )

                for i in range(self.cluster_num):
                    plt.plot(x_train, self.cluster[i].mean, "k--", lw=2)
                return plt
            plt.figure()

            std_dev_num = 1
            if data_means is None:
                self.traj_means = [
                    self.cluster[k].mean for k in range(self.cluster_num)
                ]
            else:
                mean_shift = (
                    1.0
                    / (self.data.shape[0] * self.mixing_weights)
                    * (self.cluster_probs @ data_means)
                )
                self.traj_means = [
                    self.cluster[k].mean + mean_shift[k]
                    for k in range(self.cluster_num)
                ]

            for i in range(self.cluster_num):
                plt.plot(
                    x_train, self.traj_means[i], color=color_list[i], lw=2
                )
                plt.gca().fill_between(
                    x_train,
                    self.traj_means[i]
                    - (std_dev_num * self.cluster[i].cov) ** 0.5,
                    self.traj_means[i]
                    + (std_dev_num * self.cluster[i].cov) ** 0.5,
                    color=color_list[i],
                    alpha=0.4,
                )
            return

    def Plot_Cluster_kspace_2D_slice(
        self,
        threshold,
        figsize_=None,
        data_ind=None,
        slice_ind=None,
        axis_=None,
        cluster_assignments=None,
        cluster_list=None,
    ):
        """Plots the 2D image slice color coded by the clustering label.

        Each (thresholded) pixel color is coded by the clustering label,
        or colored grey if not clustered.

        Parameters
        ----------
        threshold :
            class Threshold_Background from Preprocessing
        figsize_ : tuple, optional,
            size of image, by default None
        data_ind : ndarray, optional
            (H,K,L) or (H,K) indices of the clustered data,
            by default None. shape=(num_data,2) or (num_data,3),
        slice_ind : int, optional
            If 3D, the index along which to take the data slice,
            by default None
        axis_ : int, optional
            Axis_ of slice_ind, by default None. If data is 3D, set 0
            for L plane, 1 for K plane and 2 for H plane,
        cluster_assignments : ndarray, optional
            cluster assignments in range(cluster_num) of the data,
            shape=(num_data), by default None.
        cluster_list : array like, optional, by default None.
            List of cluster numbers 0,1,2 etc. to be plotted.
            The remaining clusters if any will show as grey. If None,
            all clusters are plotted.
        """

        import matplotlib.pyplot as plt
        from matplotlib import colors

        if cluster_list is None:
            cluster_list = range(self.cluster_num)

        if data_ind is None:
            data_ind = threshold.ind_thresholded

        plotting_matrix = (
            threshold.thresholded.copy()
        )  # 2D or 3D matrix to be assigned cluster labels for each pixels

        data_shape = threshold.data_shape_orig[1:]  # (num_l, num_k, num_h)

        if cluster_assignments is None:
            cluster_assignments = self.cluster_assignments

        for k in cluster_list:
            cluster_mask = cluster_assignments == k
            cluster_ind = data_ind[cluster_mask]
            if len(data_shape) == 2:
                plotting_matrix[cluster_ind[:, 0], cluster_ind[:, 1]] = k + 2
            elif len(data_shape) == 3:
                plotting_matrix[
                    cluster_ind[:, 0], cluster_ind[:, 1], cluster_ind[:, 2]
                ] = (k + 2)

        color_list = ["white","lightgrey"]+self.color_list
        
        cluster_cmap = colors.ListedColormap(color_list)

        bounds = []
        for i in range(len(color_list) + 1):
            bounds.append(i - 0.5)
        norm = colors.BoundaryNorm(bounds, cluster_cmap.N)

        if len(data_shape) == 2:
            self.plot_image = plotting_matrix
        elif len(data_shape) == 3:
            self.plot_image = plotting_matrix.take(slice_ind, axis=axis_)

        self.plot_norm = norm
        self.plot_cmap = cluster_cmap

        if figsize_ is not None:
            plt.figure(figsize=figsize_)
            plt.imshow(
                self.plot_image, origin="lower", cmap=cluster_cmap, norm=norm
            )

    def Plot_Cluster_Results_kspace_3D(self, threshold):

        """Plots the 3D thresholded data color coded by their cluster
        assignments.

        Parameters
        ----------
        threshold : class Threshold_Background from Preprocessing
        """

        import matplotlib.pyplot as plt
        from matplotlib import colors

        data_shape = threshold.data_shape_orig[1:]
        Ql_cell = np.arange(data_shape[0]) / (data_shape[0] - 1)
        Qk_cell = np.arange(data_shape[1]) / (data_shape[1] - 1)
        Qh_cell = np.arange(data_shape[2]) / (data_shape[2] - 1)

        X, Y, Z = np.meshgrid(Ql_cell, Qk_cell, Qh_cell, indexing="ij")
        X = np.reshape(X, np.prod(data_shape))
        Y = np.reshape(Y, np.prod(data_shape))
        Z = np.reshape(Z, np.prod(data_shape))

        low_std_dev_cluster = np.zeros(data_shape).astype(bool)
        if len(data_shape) == 2:
            low_std_dev_cluster[
                threshold.ind_low_std_dev[:, 0],
                threshold.ind_low_std_dev[:, 1],
            ] = True
        elif len(data_shape) == 3:
            low_std_dev_cluster[
                threshold.ind_low_std_dev[:, 0],
                threshold.ind_low_std_dev[:, 1],
                threshold.ind_low_std_dev[:, 2],
            ] = True

        masks = [low_std_dev_cluster.reshape(np.prod(data_shape))]

        for k in range(self.cluster_num):
            cluster_mask = self.cluster_assignments == k
            cluster_ind = threshold.ind_high_std_dev[cluster_mask]
            cluster_kspace_mask = np.zeros(data_shape).astype(bool)
            cluster_kspace_mask[
                cluster_ind[:, 0], cluster_ind[:, 1], cluster_ind[:, 2]
            ] = True
            masks.append(cluster_kspace_mask.reshape(np.prod(data_shape)))

        color_list = ["white","lightgrey"]+self.color_list
        

        RGBs = [colors.to_rgb(color) for color in color_list]

        import matplotlib.pyplot as plt

        fig = plt.figure(dpi=600)
        ax = fig.add_subplot(111, projection="3d")

        for mask_num, mask in enumerate(masks):
            if mask_num > 0:
                alpha = 0.3
                sval = 20
                rgba_mask = np.append(RGBs[mask_num + 1], alpha)[np.newaxis, :]
                # ax.scatter(X[mask],Y[mask],Z[mask],c=rgba_mask,marker=".",s=sval,edgecolors='none')
                ax.scatter(
                    (5 * X[mask] - 5),
                    (5 * Y[mask] - 5),
                    (5 * Z[mask] - 5),
                    c=rgba_mask,
                    marker=".",
                    s=sval,
                    edgecolors="none",
                )

        ax.view_init(azim=50)
        # ax.grid(False)
        ax.set_xlabel("$Q_l$", fontsize=15)
        ax.set_ylabel("$Q_k$", fontsize=15)
        ax.set_zlabel("$Q_h$", fontsize=15)
        plt.tight_layout()

    def Get_pixel_labels(self, Peak_avg):
        """Assigns the cluster labels of each peak averaged
        trajectory to all the pixels that belong to a peak.

        Parameters
        ----------
        Peak_avg : class Peak_averaging from Preprocessing
            Contains peak averages and pixels in each peak.

        Returns
        -------
        Data_ind : array like, shape=(num_data,3)
            hkl indices of all the data
        Pixel_assignments : array like, shape=(num_data)
            cluster assignments in range(cluster_num) of
            all pixels in the data
        """

        Peak_avg_data = Peak_avg.peak_avg_data
        # shape=(num_temperatures, num_thresholded data)
        Peak_avg_ind_list = Peak_avg.peak_avg_ind_list  #

        Data_ind = []
        Pixel_assignments = []

        for i in range(Peak_avg_data.shape[1]):
            Data_ind.append(Peak_avg_ind_list[i])
            for j in range(Peak_avg_ind_list[i].shape[0]):
                Pixel_assignments.append(self.cluster_assignments[i])

        self.Data_ind = np.vstack(Data_ind)

        self.Pixel_assignments = np.vstack(Pixel_assignments).flatten()

class Cluster_Gaussian(object):
    """
    Attributes
    ----------
    mean : array-like
        Mean of gaussian, dim=num_T
    cov : array-like
        Sample covariance (diagonal), dim = num_T.
    """

    def __init__(self, mean,cov):
        self.mean = mean
        self.cov=np.diag(cov) if cov.ndim == 2 else cov
        