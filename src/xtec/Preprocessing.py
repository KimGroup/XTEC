import numpy as np
from scipy import ndimage
from scipy.special import xlogy

"""
This module performs the preprocessing steps to be used before
clustering. Here, the raw data (2D slice or 3D) is processed to remove
zeros (Mask_Zeros), the low intensity background noise
(Threshold_Background), and remove low variance temperature
trajectories (Get_High_Variance).

Written by Jordan Venderley, and modifications by
Krishnanand Mallayya.

A snippet to applying thresholding is,

    from xtec.Preprocessing import Mask_Zeros
    from xtec.Preprocessing import Threshold_Background
   
    I=np.array(data)

    # remove points where I=0 for all T (mean_T =0)
    masked    = Mask_Zeros(I, 'zero_mean')

    # remove background from maksed zero data
    threshold = Threshold_Background(masked, None, 'KL')

    num_thresholded  = threshold.data_thresholded.shape[1]
    if num_thresholded  == 0:
        print('Thresholding fail')
    else:
        print('Num data thresholded=', num_thresholded)
    threshold.plot_cutoff()

    # The processed data, shape=(num_T, num_thresholded_data)
    data_post_thresh = threshold.data_thresholded
    data_post_thresh_no_mean = threshold.Rescale_traj(
        data_post_thresh)   # = data_post_thresh/<data_post_thresh>_T - 1
    # get (rescaled) data with std_dev (in T) > 0.3
    threshold.Get_High_Variance(data_post_thresh_no_mean, 0.3)

    plt.plot(Tlist, threshold.data_high_std_dev)
    plt.xlabel('T')
    plt.ylabel('Rescaled $I_q(T)$')
"""


class Mask_Zeros(object):
    """Removes zero values from the data.

    Attributes
    ----------
    data_non_zero : array-like
        Masked data points. shape=(num_T, num_data_non_zero)
    ind_cols_nonzero : array-like
        HKL indices of the masked data. shape=(num_data_non_zero, 3)
    data_shape : tuple
        Shape of input data.
    """

    def __init__(self, data, mask_type="zero_mean", check4NaN=False):
        """Initialize data and mask types.

        Parameters
        ----------
        data : array-like. Shape=(num_T, num_L, num_K, num_H) 
            Intensity (I) in 2D or 3D as a function of temperature (T).
        mask_type : 'zero_mean' or 'any_zeros', optional
            Type of mask used to remove data, by default 'zero_mean'.
            When set to 'zero_mean', I=0 for all T (mean_T=0), but when set to
            'any_zeros', I=0 for any T.
        check4NaN : bool, optional
            True if NaNs are to be removed from the data, by default False
        """
        self.data_shape = data.shape
        if mask_type == "zero_mean":
            zero_mask = (
                np.mean(data, axis=0) <= 0
            )  # mean over Temp: returns True if mean<=0
        elif mask_type == "any_zeros":
            zero_mask = (data <= 0).any(
                axis=0
            )  # True if any one Temp gives data<=0
        if check4NaN is True:
            NaN_mask = np.isnan(data).any(
                axis=0
            )  # Ture if any one Temp data is NaN
            zero_mask = np.logical_or(zero_mask, NaN_mask)

        self.zero_mask = zero_mask
        self.ind_cols_with_zeros = np.array(
            np.where(zero_mask)
        ).transpose()  # hkl indices where zero mask  = true
        self.ind_cols_nonzeros = np.array(
            np.where(~zero_mask)
        ).transpose()  # indices where zero mask = False
        self.data_with_zeros = data[:, zero_mask]  # zero data at each T
        self.data_nonzero = data[:, ~zero_mask]  # non zero data at each T


class Threshold_Background(object):
    """Removes low intensity background noise.

    Attributes
    ----------
    data_thresholded : array-like, shape=(num_T,num_data_threshed)
        Thresholded data. 
    ind_thresholded : array-like, shape=(num_data_threshed,3)
        HKL indices of the thresholded data
    thresholded : array-like, shape=(num_T, num_L, num_K, num_H)
        Values set to 0 or 1 depending on whether the corresponding pixel is
        thresholded.
    data_high_std_dev : array-like, shape=(num_T, num_high_std_data)
        High variance trajectories.
    ind_high_std_dev : array-like, shape=(num_high_std_data, 3)
        HKL indices of high std trajectories
    data_low_std_dev : array-like, shape=(num_T, num_low_std_data)
        Low variance trajectories
    ind_low_std_dev : array-like, shape=(num_low_std_data,3)
        HKL indices of low var trajectories
    """

    def __init__(self, mask, bin_size=None, threshold_type="KL", max_iter=100,set_cutoff=-1e6):
        """Initialize the threshold settings.

        Parameters
        ----------
        mask : Mask_Zeros
             Non-zero data
        bin_size : float, optional
            Bin size for the intensity distribution, by default None.
            If None, the optimal bin size is estimated from Freedman Diaconis
            rule.
        threshold_type : 'KL'|'simple'|'no thresh', optional
            by default 'KL'.
            'KL' uses the Kullback-Leibler method.
            'simple' uses a `mean+2sigma` cutoff,
             'no thresh' sets the cutoff to 0.
        max_iter : int, optional
            Maximum number of iterations, by default 100
        set_cutoff: manually set cutoff for Threshold type "no thresh"
        """
        #  shape=(num_T, num_data), the  input data to be thresholded.
        data = mask.data_nonzero
        self.bin_size = bin_size  # bin size for histogram
        self.data_shape = data.shape  # data_shape = (num_T, num_data)
        self.threshold_type = threshold_type  #
        self.data_shape_orig = (
            mask.data_shape
        )  # original data shape from the mask
        data = data.reshape(
            (data.shape[0], np.prod(data.shape[1:]))
        )  # data.shape= (num_T, num_data)
        logData_mean = np.log(np.mean(data, axis=0))  # = log[avg_T[I(q)]]
        logData_mean_sorted = np.sort(logData_mean)
        # Calculate optimal bin_size from Freedman-Diaconis rule
        if self.bin_size is None:
            self.bin_size = self.Freedman_Diaconis_for_bin_width(
                logData_mean_sorted
            )
            
        # becomes true only when thresholding is successful and complete
        self.success = False
        if self.threshold_type == "KL":
            self.y_bins, self.x_bins = self.hist(logData_mean, self.bin_size)
            try:
                self.optimal_x_ind_cut = self.Truncate(
                    self.x_bins, self.y_bins, logData_mean_sorted, max_iter
                )
                # optimal intensity cutoff
                self.LogI_cutoff = self.x_bins[self.optimal_x_ind_cut]

                # Sanity Check:
                naive_mean = np.mean(logData_mean_sorted)
                naive_std = np.std(logData_mean_sorted)
                if (
                    self.LogI_cutoff > naive_mean + 2 * naive_std
                    or self.LogI_cutoff < naive_mean - naive_std
                ):
                    self.success = False  # over/under threshold
                else:
                    # cut off lies in (mean-sigma,mean+2*sigma)
                    # of the full data log[avg_T[I(q)]]
                    self.success = True

            except Exception:
                self.success = False
            self.LogI_cutoff = self.x_bins[self.optimal_x_ind_cut]
            self.LTS_ind_cut = np.searchsorted(
                logData_mean_sorted, self.LogI_cutoff
            )
            self.mean_opt = np.mean(
                logData_mean_sorted[: self.LTS_ind_cut]
            )  # avg_q of {log[avg_T[I(q)]]} of cutoff data
            self.std_dev_opt = (
                np.var(logData_mean_sorted[: self.LTS_ind_cut]) ** 0.5
            )  # std_q of {log[avg_T[I(q)]]} of cutoff data
        elif self.threshold_type == "simple":
            self.naive_mean = np.mean(logData_mean_sorted)
            self.naive_std = np.std(logData_mean_sorted)
            self.LogI_cutoff = self.naive_mean + 2 * self.naive_std
        else:
            #print("No thresholding performed")
            #self.LogI_cutoff = -1e6
            self.LogI_cutoff = set_cutoff

        self.mask_threshold = (
            logData_mean > self.LogI_cutoff
        )  # True/False whether data above I cutoff
        self.data_thresholded = mask.data_nonzero[
            :, self.mask_threshold
        ]  # True/False whether non zero masked data above I cutoff
        self.ind_thresholded = mask.ind_cols_nonzeros[self.mask_threshold]
        self.thresholded = np.zeros(self.data_shape_orig[1:])
        if len(self.data_shape_orig[1:]) == 2:
            self.thresholded[
                self.ind_thresholded[:, 0], self.ind_thresholded[:, 1]
            ] = True
        elif len(self.data_shape_orig[1:]) == 3:
            self.thresholded[
                self.ind_thresholded[:, 0],
                self.ind_thresholded[:, 1],
                self.ind_thresholded[:, 2],
            ] = True
        self.thresholded = self.thresholded.astype(int)

    # functions for getting high variance data
    def Rescale_mean(self, data):
        self.traj_means = np.mean(data, axis=0)
        rescale_data = data / self.traj_means - 1
        return rescale_data

    def Rescale_zscore(self, data):
        self.traj_means = np.mean(data, axis=0)
        self.traj_std = np.std(data, axis=0)
        rescale_data = (data - self.traj_means) / self.traj_std
        return rescale_data

    def Get_High_Variance(self, data, std_dev_cutoff=0.5):
        """Gets data whose std dev in temperature > std_dev_cutoff

        Parameters
        ----------
        data : array-like, shape=(num_T, num_data)
            Thresholded and rescaled data
        std_dev_cutoff : float, optional
            Standard deviation cut-off, by default 0.5
        """
        std_dev = np.std(data, axis=0)
        mask_std_dev = std_dev >= std_dev_cutoff
        self.data_high_std_dev = data[:, mask_std_dev]
        self.ind_high_std_dev = self.ind_thresholded[mask_std_dev]
        self.data_low_std_dev = data[:, ~mask_std_dev]
        self.ind_low_std_dev = self.ind_thresholded[~mask_std_dev]
        self.mask_std_dev= mask_std_dev

    def Gaussian(self, x, mean, std_dev):
        """Evaluates Gaussian(x)= N(x|mean, std_dev)

        Parameters
        ----------
        x : array-like
            Independent variable
        mean : float
            Gaussian mean
        std_dev : float
            Gaussian standard deviation
        """

        return (
            1.0
            / (2 * np.pi * std_dev ** 2) ** 0.5
            * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))
        )

    def hist(self, x, bin_size):
        """Returns probability density as a function of the x-bins.

        Parameters
        ----------
        x : array-like
            Independent variables
        bin_size : int
            Bin size
        """

        max_val = np.max(x)
        min_val = np.min(x)
        self.bin_size = bin_size
        bin_num = int(np.ceil((max_val - min_val) / bin_size))
        x_hist = np.histogram(x, bin_num)
        return (
            x_hist[0] / (x.shape[0] * bin_size),
            x_hist[1][:-1],
        )  # returns prob density y= freq/[(total #x)dx] , and x-bins

    def KL(self, x_bins, y_bins, ind, mean, std_dev):
        """Returns KL divergence between p(x): y_bins and q(x): a Gaussian."""

        # normalized Gaussian, 1e-12 to avoid zeros in q(x)
        Model = self.Gaussian(x_bins[:ind], mean, std_dev) + 1e-12
        # sum p(x)dx
        norm_factor = np.sum(y_bins[:ind]) * (x_bins[1] - x_bins[0])
        # Renorm data after truncation
        y_bins_normed = y_bins[:ind] / norm_factor
        # p(x)log[p(x)/q(x)]
        KL_vec = -xlogy(y_bins_normed, Model) + xlogy(
            y_bins_normed, y_bins_normed
        )
        KL = np.sum(KL_vec)
        return KL

    def Truncate(self, x_bins, y_bins, logData_mean_sorted, max_iter):
        def Calc_KL(x_index, x_bins, y_bins, logData_mean_sorted):
            """x_index of x_bins to truncate calculating mean and std"""

            LTS_ind = np.searchsorted(
                logData_mean_sorted, x_bins[x_index]
            )  # location of corresponding intensity in sorted list
            mean_guess = np.mean(logData_mean_sorted[:LTS_ind])
            std_dev_guess = np.var(logData_mean_sorted[:LTS_ind]) ** 0.5
            KL_div = self.KL(
                x_bins, y_bins, x_index, mean_guess, std_dev_guess
            )
            return KL_div

        init_mean = np.mean(logData_mean_sorted)  # avg_q{log[avg_T[I(q)]]}
        init_std = np.std(logData_mean_sorted)  # std_q{log[avg_T[I(q)]]}
        init_ind = np.searchsorted(
            x_bins, init_mean + init_std
        )  # initial index=location in x_bins for (mean+std)
        x_ind_guess = int(0.5 * np.argmax(y_bins) + 0.5 * init_ind)
        if x_ind_guess >= y_bins.shape[0]:
            x_ind_guess = int(np.argmax(y_bins)) - 2
        x_ind_guess_plus_dx = x_ind_guess + 1  # For right derivative

        # Do gradient descent to find optimal x_index cutoff, max is 100
        delta_x_guess = 1e6
        counter = 0
        while delta_x_guess > self.bin_size and counter < max_iter:
            KL_div = Calc_KL(x_ind_guess, x_bins, y_bins, logData_mean_sorted)
            KL_div_plus_dx = Calc_KL(
                x_ind_guess_plus_dx, x_bins, y_bins, logData_mean_sorted
            )
            DLogKL = (np.log(KL_div_plus_dx) - np.log(KL_div)) / self.bin_size
            delta_x_guess = np.abs(self.bin_size * DLogKL)
            x_guess = x_bins[x_ind_guess] - self.bin_size * DLogKL
            x_ind_guess = np.searchsorted(
                x_bins, x_guess
            )  # can make this const time
            if x_ind_guess >= x_bins.shape[0]:
                x_ind_guess = x_bins.shape[0] - 2

            x_ind_guess_plus_dx = x_ind_guess + 1
            counter = counter + 1
        return x_ind_guess

    def Freedman_Diaconis_for_bin_width(self, sorted_data_):
        """Returns optimal bin size using the Freedman-Diaconis method."""

        upper_ind = int(np.round(sorted_data_.shape[0] * 0.75))
        lower_ind = int(np.round(sorted_data_.shape[0] * 0.25))
        IQR = sorted_data_[upper_ind] - sorted_data_[lower_ind]
        bin_size = 0.5 * (
            2 * IQR / (sorted_data_.shape[0] ** (1 / 3))
        )  # Uses half the FD bin width
        return bin_size

    def plot_cutoff(self, figsize_=(15, 15)):
        """Plots the Intensity distribution and marks the cutoff point

        The distribution of intensity (averaged over T), shows the threshold
        cutoff and the fitting function

        Parameters
        ----------
        figsize_ : tuple, optional
            Size of the figure, by default (15,15)
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize_)
        plt.plot(self.x_bins, self.y_bins, label="Raw Data")
        plt.ylabel(r"Density at $\log(\overline{I_q(T)})$")
        plt.xlabel(r"$\log(\overline{I_q(T)})$")
        if self.threshold_type == "KL":
            plt.plot(
                self.x_bins,
                self.Gaussian(self.x_bins, self.mean_opt, self.std_dev_opt),
                label="Background Fit",
            )
            plt.scatter(
                self.x_bins[self.optimal_x_ind_cut],
                0,
                color="red",
                zorder=10,
                label="Background Truncation",
            )
        elif self.threshold_type == "simple":
            plt.plot(
                self.x_bins,
                self.Gaussian(self.x_bins, self.naive_mean, self.naive_std),
                label="Naive Background Fit",
            )
            plt.scatter(
                self.naive_mean + 2 * self.naive_std,
                0,
                color="green",
                zorder=10,
                label="Background Truncation",
            )
        else:
            print("No thresholding performed")
        plt.rcParams.update({"font.size": 14})
        plt.legend()

    def plot_thresholding_2D_slice(
        self, figsize_=(10, 10), slice_ind=None, axis_=None
    ):
        """Plots the 2D figure showing points that pass thresholding.
        These are displayed with lightgrey pixels.

        Parameters
        ----------
        slice_ind : int, optional
            If 3D, the index along which to take the data slice,
            by default None
        axis_ : int, optional
            Axis_ of slice_ind, by default None. Set 0
            for L plane, 1 for K plane and 2 for H plane,
        """
        import matplotlib.pyplot as plt
        from matplotlib import colors

        # make a color map of fixed colors
        color_list = ["white", "lightgrey"]
        cluster_cmap = colors.ListedColormap(color_list)
        bounds = []
        for i in range(len(color_list) + 1):
            bounds.append(i - 0.5)
        norm = colors.BoundaryNorm(bounds, cluster_cmap.N)

        # plot 2D thresholded data, or 2D slice of 3D thresholded data
        data_shape = self.data_shape_orig[1:]  # (num_l, num_k, num_h)

        if len(data_shape) == 2:
            plotting_matrix = self.thresholded
        elif len(data_shape) == 3:
            plotting_matrix = self.thresholded.take(slice_ind, axis=axis_)

        plt.figure(figsize=figsize_)
        plt.imshow(
            plotting_matrix, origin="lower", cmap=cluster_cmap, norm=norm
        )

        return plt

    def plot_thresholding_3D(self):
        """Plots the 3D figure showing points that pass thresholding.

        These are shown with lightgrey pixels.
        """
        import matplotlib.pyplot as plt

        Ql_cell = np.arange(self.data_shape_orig[1]) / (
            self.data_shape_orig[1] - 1
        )
        Qk_cell = np.arange(self.data_shape_orig[2]) / (
            self.data_shape_orig[2] - 1
        )
        Qh_cell = np.arange(self.data_shape_orig[3]) / (
            self.data_shape_orig[3] - 1
        )

        X, Y, Z = np.meshgrid(Ql_cell, Qk_cell, Qh_cell, indexing="ij")

        blue = np.array([66, 134, 244]) / 255.0
        blue = blue[np.newaxis, np.newaxis, np.newaxis, :] + np.zeros(
            self.data_shape_orig[1:] + (3,)
        )
        rgba_mat = self.thresholded
        rgba_mat = rgba_mat[:, :, :, np.newaxis]
        rgba_mat = np.append(blue, rgba_mat, axis=3)

        X = np.reshape(X, np.prod(self.data_shape_orig[1:]))
        Y = np.reshape(Y, np.prod(self.data_shape_orig[1:]))
        Z = np.reshape(Z, np.prod(self.data_shape_orig[1:]))
        rgba_mat = np.reshape(rgba_mat, (np.prod(self.data_shape_orig[1:]), 4))

        mask = self.thresholded.astype(bool).reshape(
            np.prod(self.data_shape_orig[1:])
        )
        rgba_mat_subset = rgba_mat[mask, :]
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]

        fig = plt.figure(dpi=70)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X, Y, Z, c=rgba_mat_subset, s=30, edgecolors="none")
        ax.set_xlabel("$Q_l$", fontsize=15)
        ax.set_ylabel("$Q_k$", fontsize=15)
        ax.set_zlabel("$Q_h$", fontsize=15)
        ax.view_init(azim=20)
        plt.tight_layout()



class Peak_averaging(object):
    """Averages intensities within each peak.

    Attributes
    ----------
    peak_avg_data : array-like, shape=(num_T, num_peaks)
        Peak averaged data for each peak
        (cluster of conneced pixels)
    peak_avg_ind_list : list of arrays
        i^th list entry gives  HKL indices of the
        pixels inside the i^th peak
    """

    def __init__(self, intensity, threshold):
        """Initialize the peak-averaged data.

        Parameters
        ----------
        intensity : array-like, shape=(num_T, num_L, num_K, num_H)
            Intensity to be averaged.
        threshold : class Threshold_Background
            Threshold data 
        """

        # identify connected features
        data_shape = threshold.data_shape_orig[1:]  # (num_l, num_k, num_h)
        if len(data_shape) == 2:
            # to identify features if they are connected, even diagonally
            structure_element = np.ones((3, 3))
        elif len(data_shape) == 3:
            # to identify features if they are connected, even diagonally
            structure_element = np.ones((3, 3, 3))

        labeled_array, num_features = ndimage.label(
            threshold.thresholded, structure=structure_element
        )
        # labelled_array(h,l) has connected features (#num_features)
        # marked by different integers

        P_avg_data = []
        P_avg_ind_list = []
        
        P_max_data = []
        P_max_ind_list = []
        
        for i in range(1, num_features + 1):
            label_i = np.isin(labeled_array, i)
            data_i = intensity[:, label_i]
            ind_i = np.array(np.where(label_i)).transpose()

            peak_avg_i = np.mean(data_i, axis=1)
            P_avg_data.append(peak_avg_i)
            P_avg_ind_list.append(ind_i)
            
            peak_max_i = np.amax(data_i, axis=1)
            P_max_data.append(peak_max_i)
            P_max_ind_list.append(ind_i)
            

        # shape=(num_T, num_peak avg data)
        self.peak_avg_data = np.vstack(P_avg_data).transpose()
        self.peak_avg_ind_list = P_avg_ind_list
        
        self.peak_max_data = np.vstack(P_max_data).transpose()
        self.peak_max_ind_list = P_max_ind_list
        
        
