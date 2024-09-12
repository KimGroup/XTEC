import sys

import numpy as np


class GMM(object):
    """Implementation of the Gaussian Mixture Model.

    Performes Step-wise EM (cf. Liang and Klein 2009) to cluster
    data(num_data, num_T).

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
        1) An example to cluster with GMM without label smoothing:

        from xtec.GMM import GMM

        num_clusters = 2
        clusterGMM=GMM(data, num_clusters)
        clusterGMM.RunEM()

        print(clusterGMM.num_per_cluster)
        cluster_assignments = clusterGMM.cluster_assignments
        clusterGMM.Plot_Cluster_Results_traj(Temp, False)
        # False plots only GMM cluster results,
        # True plots all the data trajectories color coded by the clustering
        # assignment

        2) Example to cluster with label smoothing and periodic kernel
            (see GMM_kernel for constructing Markov matrix):

        # Generate Markov matrix to be used between E and M step to
        # diffuse cluster probabilities with neighbouring data

        from xtec.GMM import GMM
        from xtec.GMM import GMM_kernels

        data_inds = threshold.ind_high_std_dev
        L_scale = 0.2
        kernel_type = 'periodic'
        unit_cell_shape = np.array([20,20,20])
        uniform_similarity = True
        Markov_matrix = Build_Markov_Matrix(data_inds, L_scale,
                                            kernel_type,
                                            unit_cell_shape,
                                            uniform_similarity)

        # GMM clustering with label smoothing
        num_clusters = 2
        clusterGMM=GMM(data, num_clusters)
        smoothing_iterations=1
        clusterGMM.RunEM(True, Markov_matrix, smoothing_iterations)

        print(clusterGMM.num_per_cluster)
        cluster_assignments = clusterGMM.cluster_assignments
        clusterGMM.Plot_Cluster_Results_traj(Temp,False)

    """

    def __init__(
        self,
        data,
        cluster_num,
        cov_type="diagonal",
        batch_num=1,
        alpha=0.7,
        tol=1e-5,
        max_batch_epoch=50,
        max_full_epoch=500,
        verbose=False,
        color_list=None
    ):
        """Initialize GMM parameters

        Parameters
        ----------
        data : array-like
            Input data. The first axis corresponds to the number of samples
            (num_data), and the second to num of temperatures (num_T)
        cluster_num : int
            Number of GMM clusters
        cov_type : "diagonal"| "full", optional
            Setting to choose whether to keep only diagonal elements or
            retain the full covariance matrix, by default "diagonal"
        batch_num : int, optional
            Number of batches, by default 1
        alpha : float, optional
            decay exponent of step-size 0.5 < alpha < 1, by default 0.7
        tol : float, optional
            Tolerance for the convergence of loglikelihood, by default 1e-5
        max_batch_epoch : int, optional
            Maximum number of batch iterations, by default 50
        max_full_epoch : int, optional
            Maximum number of iterations on full dataset, by default 500
        verbose : bool, optional
            If True, print loglikelihood at each iteration, by default False
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
        self.cluster = []
        self.cluster_num = cluster_num
        self.cov_type = cov_type
        if self.cov_type != "full" and self.cov_type != "diagonal":
            print("Error: cov_type must be full or diagonal")
            sys.exit()
        self.verbose = verbose
        # starting with equal mixing weights: w_k=1/K
        self.mixing_weights = (1.0 / self.cluster_num) * np.ones(
            self.cluster_num
        )
        self.batch_num = batch_num
        self.tol = tol
        self.max_batch_epoch = max_batch_epoch
        self.max_full_epoch = max_full_epoch
        self.epoch = 0

        # For setting up evenly distributed batch sizes for stepwise EM
        # divide data (num_data) in to batch_num batches
        self.batch_base_num = int(np.floor(data.shape[0] / self.batch_num))
        # the remaining data after dividing into batches
        self.batch_mod_num = data.shape[0] % self.batch_num
        # handle case when given batch_num > sample_num
        if self.batch_base_num == 0:
            self.batch_num = self.batch_mod_num

        # decay exponent for i'th step-size eta_i = (i + 2 )^{- alpha},
        # 0.5 < alpha <1
        self.alpha = alpha
        # make deep copy for easy randomization
        self.data_rand = data.copy()
        self.data = data.copy()

        self.cluster_probs = None

        for k in range(self.cluster_num):
            #  mean_guess and cov from Gaussian with mean and cov of the data.
            new_cluster = Cluster_Gaussian(data)
            # list of K Cluster_Gaussian initially with diff means
            # (mean_guess), and same cov
            self.cluster.append(new_cluster)

        # keep only diagonal elements of cov, discard the off-diag terms
        if self.cov_type == "diagonal":
            for k in range(self.cluster_num):
                self.cluster[k].cov = np.diag(self.cluster[k].cov)

        # cluster means, shape=(cluster_num,num_T)
        self.means = np.array(
            [self.cluster[k].mean for k in range(self.cluster_num)]
        )
        # cluster covs
        self.covs = np.array(
            [self.cluster[k].cov for k in range(self.cluster_num)]
        )

    def RunEM(
        self,
        label_smoothing_flag=False,
        Markov_matrix=None,
        smoothing_iterations=1,
    ):
        """Performs stepwise EM (cf. Liang and Klein 2009)

        Parameters
        ----------
        label_smoothing_flag : bool, optional
            If True implements label smoothing between E and M step,
            by default False
        Markov_matrix : class scipy.sparse.csr.csr_matrix, optional
            Adjacency matrix for label smoothing (when smoothing_flag=True),
            by default None. Use GMM_kernels to construct Markov
            matrix with local or periodic kernel
        smoothing_iterations : int, optional
            Number of times Markov_matrix is applied to cluster prob
            between E and M step, by default 1
        """
        loglikelihood_diff = 1e6
        loglikelihood_new = -1e6

        while (
            loglikelihood_diff > self.tol and self.epoch < self.max_batch_epoch
        ):
            loglikelihood_old = loglikelihood_new
            # stepwise EM applies only when batch_num>1, standard EM
            # needs no data randomization
            if self.batch_num != 1:
                # randomize the (deep copied) data
                np.random.shuffle(self.data_rand)

            # batch data is the shuffled data[left_bound:right_bound]
            left_bound = 0
            right_bound = 0
            for batch_val in range(self.batch_num):
                right_bound += self.batch_base_num + (
                    1 if batch_val < self.batch_mod_num else 0
                )
                # first batch_mod_num batches accomodate the batch_mod data
                data_batch = self.data_rand[left_bound:right_bound]
                left_bound = right_bound
                self.E_Step(data_batch)
                self.M_Step(data_batch)

            loglikelihood_new = self.LogLikelihood(self.data)
            loglikelihood_diff = np.abs(loglikelihood_new - loglikelihood_old)
            if self.verbose is True:
                print("Batch Log-likelihood:", loglikelihood_new)
            self.epoch += 1  # number of steps for stepwise EM to converge

        if label_smoothing_flag is True:
            self.Markov_matrix = Markov_matrix

        # Finish on all data (guaranteed one "full" pass)
        loglikelihood_diff = 1e6
        loglikelihood_new = -1e6
        batch_epoch = self.epoch
        self.converged = None
        while (
            loglikelihood_diff > self.tol
            and self.epoch < batch_epoch + self.max_full_epoch + 1
        ):
            loglikelihood_old = loglikelihood_new
            self.E_Step(self.data)
            if label_smoothing_flag is True:
                self.Smooth_Labels(smoothing_iterations)
            self.M_Step(self.data)
            loglikelihood_new = self.LogLikelihood(self.data)
            # label smoothing can decrease the log-likelihood
            loglikelihood_diff = np.abs(loglikelihood_new - loglikelihood_old)
            self.loglikelihood = loglikelihood_new
            if self.verbose is True:
                print("Full  Log-likelihood:", self.loglikelihood)
            self.epoch += 1

            if self.epoch == batch_epoch + self.max_full_epoch + 1:
                self.converged = False
            if loglikelihood_diff <= self.tol:
                self.converged = True

        # Now the clustering is complete
        # update self.cluster[k].mean and self.cluster[k].cov
        for k in range(self.cluster_num):
            self.cluster[k].mean = self.means[k]
            self.cluster[k].cov = self.covs[k]

        # cluster assignment of each sample,  dim=(num_data)
        self.cluster_assignments = np.argmax(self.cluster_probs, axis=0)

        # number of samples within each cluster
        self.num_per_cluster = [
            np.sum(self.cluster_assignments == k)
            for k in range(self.cluster_num)
        ]
        self.discovery_cluster_ind = np.argmax(
            np.array(
                [
                    0.0
                    if self.num_per_cluster[k] < 2
                    else np.std(self.means[k])
                    for k in range(self.cluster_num)
                ]
            )
        )

    def E_Step(self, data_):
        """Expectation step, evaluates the cluster probabilities
        Parameters
        ----------
        data_ : array-like
            Data matrix with shape (num_data, num_T)
        """

        # log[N(data|means(k),cov(k))], dim=(num_cluster, num_data)
        log_gaussian = self.LogGaussian(data_)

        #  cluster_prob, Ck
        self.cluster_probs = self.mixing_weights[:, np.newaxis] * self.Logp2p(
            log_gaussian
        )
        self.cluster_probs /= np.sum(self.cluster_probs, axis=0)[
            np.newaxis, :
        ]  # Ck = w_k*N(data|means(k),cov(k))/[sum_k w_k*N(data|..)]

    def M_Step(self, data_):
        """Maximization step, evaluates the cluster means, variance, and mixing
        weights

        Parameters
        ----------
        data_ : array-like
            Data matrix with shape (num_data, num_T)
        """
        # w_k=<Ck> averaged over data, dim=(num_clusters)
        self.mixing_weights = np.mean(self.cluster_probs, axis=1)
        weight_mask = self.mixing_weights == 0
        # Handles potential division by 0 below
        self.mixing_weights[weight_mask] = 1e-10
        # new means, dim=(num_clusters, num_T)
        self.means_new = (
            1.0
            / (data_.shape[0] * self.mixing_weights[:, np.newaxis])
            * (self.cluster_probs @ data_)
        )

        if self.cov_type == "full":  # Often convergence issues with this
            self.covs_new = (
                1.0
                / (
                    data_.shape[0]
                    * self.mixing_weights[:, np.newaxis, np.newaxis]
                )
                * np.array(
                    [
                        data_no_mean.T
                        @ (self.cluster_probs[k][:, np.newaxis] * data_no_mean)
                        for k in range(self.cluster_num)
                        for data_no_mean in [
                            data_ - self.means_new[k][np.newaxis, :]
                        ]
                    ]
                )
            )
        elif self.cov_type == "diagonal":
            self.covs_new = (
                1.0
                / (data_.shape[0] * self.mixing_weights[:, np.newaxis])
                * np.array(
                    [
                        np.einsum(
                            "ij,ij->i",
                            data_no_mean.T,
                            (
                                self.cluster_probs[k][:, np.newaxis]
                                * data_no_mean
                            ).T,
                        )
                        for k in range(self.cluster_num)
                        for data_no_mean in [
                            data_ - self.means_new[k][np.newaxis, :]
                        ]
                    ]
                )
            )

        # Step-wise EM update
        eta_k = (self.epoch + 2) ** (-self.alpha)  # stepsize
        self.means = (1 - eta_k) * self.means + eta_k * self.means_new
        self.covs = (1 - eta_k) * self.covs + eta_k * self.covs_new

    def LogGaussian(self, data_):
        """Map data matrix to cluster log probabilities

        Parameters
        ----------
        data_ : array-like
            Data matrix with shape (num_data, dim)

        Returns
        -------
        array-like
            Log probabilities with shape (num_clusters, num_data)
        """
        log_gaussian = None
        if self.cov_type == "full":
            log_gaussian = np.array(
                [
                    -0.5 * self.cluster_num * np.log(2 * np.pi)
                    - 0.5 * np.log(np.linalg.det(self.covs[k]))
                    - 0.5
                    * np.einsum(
                        "ij,ij->i",
                        np.dot(data_no_mean, np.linalg.inv(self.covs[k])),
                        data_no_mean,
                    )
                    for k in range(self.cluster_num)
                    for data_no_mean in [data_ - self.means[k][np.newaxis, :]]
                ]
            )
        elif self.cov_type == "diagonal":
            log_gaussian = np.array(
                [
                    -0.5 * self.cluster_num * np.log(2 * np.pi)
                    - 0.5 * np.log(np.prod(self.covs[k]))
                    - 0.5
                    * np.einsum(
                        "ij,ij->i",
                        data_no_mean * (1.0 / self.covs[k])[np.newaxis, :],
                        data_no_mean,
                    )
                    for k in range(self.cluster_num)
                    for data_no_mean in [data_ - self.means[k][np.newaxis, :]]
                ]
            )
        return log_gaussian  # array of log[N(data|means(k),cov(k))]

    def Logp2p(self, log_p):
        """Convert log(P_k) to P_k/(sum_k P_k):

        Uses a Log-sum-exp trick for avoiding numerical overflow
        """
        # log_p shape= (num_cluster,num_data)
        max_logp = np.max(log_p, axis=0)[np.newaxis, :]
        # returns P_k/(\sum_k P_k)
        return np.exp(
            log_p - max_logp - np.log(np.sum(np.exp(log_p - max_logp), axis=0))
        )

    def LogLikelihood(self, data_):
        """Return Sum_data{Log[Sum_k w_k N(data|means(k),cov(k))]}"""
        # log[N(data|means(k),cov(k))],  shape=(num_cluster, num_data)
        log_gaussian = self.LogGaussian(data_)
        return np.sum(
            self.LogSumTrick(
                np.log(self.mixing_weights[:, np.newaxis]) + log_gaussian
            )
        )

    def LogSumTrick(self, log_p):
        """Evaluate log-likelihood using another log-sum trick"""
        max_logp = np.max(log_p, axis=0)
        # returns log[sum_k P_k], shape=(num_data)
        return (
            np.log(np.sum(np.exp(log_p - max_logp[np.newaxis, :]), axis=0))
            + max_logp
        )

    def LogLikelihood_lowerbound(self, data_):
        log_gaussian = self.LogGaussian(data_)
        return np.sum(
            self.cluster_probs * log_gaussian
            + self.xlogy(
                self.cluster_probs, self.mixing_weights[:, np.newaxis]
            )
            - self.xlogy(self.cluster_probs, self.cluster_probs)
        )

    def xlogy(self, x, y):
        mask = x > 0.0
        return np.where(mask, x * np.log(y, where=mask), 0.0)

    def Smooth_Labels(self, it_num=1):
        for i in range(it_num):
            # "diffused" cluster prob
            self.cluster_probs = (
                self.Markov_matrix @ self.cluster_probs.transpose()
            ).transpose()

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
            print("Error: more clusters than cluster color labels")
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
    """Randomly picks mean from a nomrmal distribution.

    Attributes
    ----------
    mean : array-like
        Mean guess from normal dist with sample mean (mean_init) and
        sample covariance, dim=num_T.
    cov : array-like
        Sample covariance (diagonal), dim = num_T.
    """

    def __init__(self, data):
        self.data_dim = data.shape[1]
        # data.shape= (num_data, num_T), data_dim=num_T
        self.mean = np.zeros((1, self.data_dim))
        self.cov = np.zeros((self.data_dim, self.data_dim))

        # Initialize w/ randomly drawn means and reasonable (spherical)
        # cov estimated from data
        mean_init = np.mean(data, axis=0)
        cov_init = np.diag(np.var(data, axis=0))
        mean_guess = np.random.multivariate_normal(mean_init, cov_init)

        self.mean = mean_guess
        self.cov = cov_init

    def pdf(self, pt):
        """Returns Gaussian prob dist at pt: N(pt|mean, cov)"""
        return (
            1.0
            / ((2 * np.pi) ** self.data_dim * np.linalg.det(self.cov)) ** 0.5
        ) * np.exp(
            -0.5
            * np.transpose(pt - self.mean)
            @ np.linalg.inv(self.cov)
            @ (pt - self.mean)
        )


class GMM_kernels(object):
    """Builds the Adjacency matrix for label smoothing"""

    def Build_Markov_Matrix(
        data_inds,
        L_scale=1,
        kernel_type="local",
        unit_cell_shape=None,
        uniform_similarity=True,
        zero_cutoff=1e-2,
    ):
        """returns the Adjacency (Markov) matrix to be used for
        label smoothing

        Parameters
        ----------
        data_inds : array-like
            HKL indices (dim=3 or dim=2) of the preprocessed data
        L_scale : int, optional
            Length scale (in units of number of pixels) for local
            correlations, by default 1
        kernel_type : 'local' | 'periodic', optional
            Use local or periodic correlations, by default "local"
        unit_cell_shape : array-like, optional
            Number of pixels defining the unit cell, by default None
        uniform_similarity : bool, optional
            if True, sets all nonzero elements of Markov matrix
            (above zero_cutoff) to 1 (with a normalization factor),
            by default True
        zero_cutoff : float, optional
            Cutoff to select nonzero elements of Markov matrix,
            by default 1e-2

        Returns
        -------
        Markov matrix: class scipy.sparse.csr.csr_matrix,
        """

        print("\n\tBuilding Adjacency Matrix,  ...")
        import time

        start_time = time.time()
        Markov_data = []
        row_all = []
        col_all = []

        for i in range(data_inds.shape[0]):
            if kernel_type == "local":
                # Marcov_vec[j] = exp[-Sum_{x=1,2,3}(qx_i-qx_j)^2/L]
                Markov_vec = np.exp(
                    -np.sum(
                        (data_inds[i] - data_inds) ** 2 / L_scale ** 2, axis=1
                    )
                )
            elif kernel_type == "periodic":
                Markov_vec = np.exp(
                    -np.sum(
                        np.sin(
                            np.pi
                            / unit_cell_shape
                            * (data_inds[i] - data_inds)
                        )
                        ** 2
                        / L_scale ** 2,
                        axis=1,
                    )
                )
                # Marcov_vec[j] = exp[-Sum_{x=1,2,3}sin[k_x(qx_i-qx_j)]^2/L]
                # where 2k_x=2pi/Lx is R basis vec, shape=(num_data)
            else:
                print("Error: Invalid kernel type")
                sys.exit()

            row_ind = np.where(
                Markov_vec > zero_cutoff
            )  # a cutoff to mask zeros
            col_ind = np.full(row_ind[0].shape, i)
            if uniform_similarity is False:
                Markov_data.append(Markov_vec[row_ind])
            else:
                Markov_data.append(
                    np.ones(Markov_vec[row_ind].shape)
                )  # sets 1 to all non-zero elements of Markov vec
            row_all.append(row_ind[0])
            col_all.append(col_ind)

        Markov_data = np.concatenate(Markov_data, axis=0)
        row_all = np.concatenate(row_all, axis=0)
        col_all = np.concatenate(col_all, axis=0)

        from scipy.sparse import csr_matrix

        Markov_matrix = csr_matrix(
            (Markov_data, (row_all, col_all)),
            shape=(data_inds.shape[0], data_inds.shape[0]),
        )

        from sklearn.preprocessing import normalize

        Markov_matrix = normalize(Markov_matrix, norm="l1", axis=1)

        print(
            "\tFinished Building Adjacency Matrix in",
            time.time() - start_time,
            "s \n",
        )

        return Markov_matrix
