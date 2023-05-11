import numpy as np
from nexpy.gui.datadialogs import GridParameters, NXDialog
from nexpy.gui.utils import display_message, report_error
from nexpy.gui.pyqt import QtWidgets
from nexpy.gui.widgets import NXComboBox, NXLabel, NXSpinBox
from nexusformat.nexus import NeXusError, NXdata, NXfield
from sklearn.mixture import GaussianMixture
from xtec.Preprocessing import Mask_Zeros, Peak_averaging, Threshold_Background
from xtec.GMM import GMM, GMM_kernels


def show_dialog():
    try:
        dialog = XTECDialog()
        dialog.show()
    except NeXusError as error:
        report_error("XTEC analysis", error)


class XTECDialog(NXDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.select_data(self.choose_data)
        self.set_layout(self.entry_layout, self.close_buttons(close=True))

        self.defaults = {
            "n_cluster": 4,
            "min_nc": 2,
            "max_nc": 14,
            "threshold": "Yes",
            "rescale": "mean",
            "L_scale": 0.05,
            "smooth_type": "local",
        }
        self.parameters = {}

        self.setWindowTitle("XTEC fit")

    def choose_data(self):
        if self.layout.count() == 2:
            axis_grid = QtWidgets.QGridLayout()
            axis_grid.setSpacing(10)
            headers = ["Axis", "Minimum", "Maximum"]
            width = [50, 100, 100]
            column = 0
            for header in headers:
                label = NXLabel(header, bold=True, align="center")
                axis_grid.addWidget(label, 0, column)
                axis_grid.setColumnMinimumWidth(column, width[column])
                column += 1

            row = 0
            self.minbox = {}
            self.maxbox = {}
            for axis in range(self.ndim):
                row += 1
                self.minbox[axis] = NXSpinBox()
                self.maxbox[axis] = NXSpinBox()
                axis_grid.addWidget(
                    self.label(self.axes[axis].nxname, align="center"), row, 0
                )
                axis_grid.addWidget(self.minbox[axis], row, 1)
                axis_grid.addWidget(self.maxbox[axis], row, 2)

            self.XTEC_method = NXComboBox(
                slot=self.change_method,
                items=[
                    "XTEC-d",
                    "XTEC-s (peak avg)",
                    "XTEC label smooth",
                    "BIC XTEC-d",
                    "BIC XTEC-s (peak avg)",
                ],
            )

            self.insert_layout(1, self.make_layout(axis_grid, align="center"))
            self.insert_layout(
                2, self.make_layout(self.XTEC_method, align="center")
            )
            self.parameters = self.define_parameters()
            self.insert_layout(
                3, self.make_layout(self.parameters.grid(), align="center")
            )
            self.parameters.grid_layout.setHorizontalSpacing(20)
            self.insert_layout(
                4,
                self.action_buttons(("Cluster Data", self.select_method)),
            )
            self.insert_layout(
                5,
                self.action_buttons(
                    ("Plot Q Map", self.plot_qmap),
                    ("Plot Trajectories", self.plot_trajectories),
                    ("Plot Average Intensity", self.plot_average_intensities),
                ),
            )
        else:
            display_message(
                "Cannot change data after initialization",
                "Please relaunch dialog",
            )
            return

        for axis in range(self.ndim):
            self.minbox[axis].data = self.maxbox[axis].data = self.axes[
                axis
            ].nxvalue
            self.minbox[axis].setMaximum(self.minbox[axis].data.size - 1)
            self.maxbox[axis].setMaximum(self.maxbox[axis].data.size - 1)
            self.minbox[axis].diff = self.maxbox[axis].diff = None
            self.minbox[axis].setValue(self.minbox[axis].data.min())
            self.maxbox[axis].setValue(self.maxbox[axis].data.max())

        self.enable_plots(enable=False)

    def define_parameters(self):
        for p in self.parameters:
            self.defaults[p] = self.parameters[p].value
        parameters = GridParameters()
        if self.XTEC_method.selected.startswith("BIC"):
            parameters.add(
                "min_nc", self.defaults["min_nc"], "Min cluster num"
            )
            parameters.add(
                "max_nc", self.defaults["max_nc"], "Max cluster num"
            )
        else:
            parameters.add(
                "n_cluster", self.defaults["n_cluster"], "Number of Clusters"
            )
        parameters.add("threshold", ["Yes", "No"], "Remove Background")
        parameters["threshold"].value = self.defaults["threshold"]
        parameters.add(
            "rescale", ["mean", "z-score", "log-mean", "None"], "Rescaling"
        )
        parameters["rescale"].value = self.defaults["rescale"]
        if self.XTEC_method.selected == "XTEC label smooth":
            parameters.add(
                "L_scale",
                self.defaults["L_scale"],
                "Smoothen length (pixel units)",
            )
            parameters.add(
                "smooth_type", ["local", "periodic"], "Smoothen type"
            )
            parameters["smooth_type"].value = self.defaults["smooth_type"]
        return parameters

    def change_method(self):
        self.parameters.delete_grid()
        self.parameters = self.define_parameters()
        self.insert_layout(
            3, self.make_layout(self.parameters.grid(), align="center")
        )
        self.parameters.grid_layout.setHorizontalSpacing(20)

    def select_method(self):
        if self.XTEC_method.selected == "XTEC-d":
            self.XTEC_d()
        elif self.XTEC_method.selected == "XTEC-s (peak avg)":
            self.XTEC_s_PA()
        elif self.XTEC_method.selected == "XTEC label smooth":
            self.XTEC_label_smooth()
        elif self.XTEC_method.selected == "BIC XTEC-d":
            self.BIC_XTEC_d()
        elif self.XTEC_method.selected == "BIC XTEC-s (peak avg)":
            self.BIC_XTEC_s()

    @property
    def data(self):
        limits = []
        for axis in range(self.selected_data.ndim):
            limits.append(
                slice(self.minbox[axis].value(), self.maxbox[axis].value())
            )
        return self.selected_data[limits]

    @property
    def ndim(self):
        return self.selected_data.ndim

    @property
    def axes(self):
        return self.selected_data.nxaxes

    def enable_plots(self, enable=True):
        self.pushbutton["Plot Q Map"].setEnabled(enable)
        self.pushbutton["Plot Trajectories"].setEnabled(enable)
        self.pushbutton["Plot Average Intensity"].setEnabled(enable)

    def XTEC_d(self):
        self.enable_plots(enable=False)
        self.nc = int(self.parameters["n_cluster"].value)

        if self.parameters["threshold"].value == "Yes":
            thresh_type = "KL"
        else:
            thresh_type = "No threshold"

        masked = Mask_Zeros(self.data.nxsignal.nxvalue)
        threshold = Threshold_Background(masked, threshold_type=thresh_type)
        Data_thresh = threshold.data_thresholded
        Data_ind = threshold.ind_thresholded
        self.Data_thresh = Data_thresh
        self.Data_ind = Data_ind

        rescale_text = self.parameters["rescale"].value
        if rescale_text == "mean":
            Rescaled_data = threshold.Rescale_mean(Data_thresh)
        if rescale_text == "z-score":
            Rescaled_data = threshold.Rescale_zscore(Data_thresh)
        if rescale_text == "log-mean":
            Rescaled_data = np.log(1 + Data_thresh)
            Rescaled_data = Rescaled_data - np.mean(Rescaled_data, axis=0)
        if rescale_text == "None":
            Rescaled_data = Data_thresh

        Data_for_GMM = Rescaled_data.transpose()
        gm = GaussianMixture(
            n_components=self.nc, covariance_type="diag", random_state=0
        ).fit(Data_for_GMM)
        cluster_assigns = gm.predict(Data_for_GMM)

        self.cluster_means = gm.means_
        self.cluster_covs = gm.covariances_
        self.cluster_assigns = cluster_assigns
        self.pixel_assigns = cluster_assigns
        self.image_data = None
        self.enable_plots(enable=True)

    def XTEC_s_PA(self):
        self.enable_plots(enable=False)
        self.nc = int(self.parameters["n_cluster"].value)
        if self.parameters["threshold"].value == "Yes":
            thresh_type = "KL"
        else:
            thresh_type = "No threshold"

        masked = Mask_Zeros(self.data.nxsignal.nxvalue)
        threshold = Threshold_Background(masked, threshold_type=thresh_type)

        Peak_avg = Peak_averaging(self.data.nxsignal.nxvalue, threshold)
        Peak_avg_data = Peak_avg.peak_avg_data
        Data_thresh = Peak_avg_data

        rescale_text = self.parameters["rescale"].value
        if rescale_text == "mean":
            Rescaled_data = threshold.Rescale_mean(Data_thresh)
        if rescale_text == "z-score":
            Rescaled_data = threshold.Rescale_zscore(Data_thresh)
        if rescale_text == "log-mean":
            Rescaled_data = np.log(1 + Data_thresh)
            Rescaled_data = Rescaled_data - np.mean(Rescaled_data, axis=0)
        if rescale_text == "None":
            Rescaled_data = Data_thresh

        Data_for_GMM = Rescaled_data.transpose()
        clusterGMM = GMM(Data_for_GMM, self.nc)
        clusterGMM.RunEM()
        clusterGMM.Get_pixel_labels(Peak_avg)
        Data_ind = clusterGMM.Data_ind
        cluster_assigns = clusterGMM.cluster_assignments
        pixel_assigns = clusterGMM.Pixel_assignments

        self.Data_thresh = Data_thresh
        self.Data_ind = Data_ind
        self.cluster_assigns = cluster_assigns
        self.pixel_assigns = pixel_assigns
        self.cluster_means = clusterGMM.means
        self.cluster_covs = [clusterGMM.cluster[i].cov for i in range(self.nc)]

        self.image_data = None

        self.enable_plots(enable=True)

    def XTEC_label_smooth(self):
        self.enable_plots(enable=False)
        self.nc = int(self.parameters["n_cluster"].value)

        if self.parameters["threshold"].value == "Yes":
            thresh_type = "KL"
        else:
            thresh_type = "No threshold"

        masked = Mask_Zeros(self.data.nxsignal.nxvalue)
        threshold = Threshold_Background(masked, threshold_type=thresh_type)
        Data_thresh = threshold.data_thresholded
        Data_ind = threshold.ind_thresholded
        self.Data_thresh = Data_thresh
        self.Data_ind = Data_ind

        rescale_text = self.parameters["rescale"].value
        if rescale_text == "mean":
            Rescaled_data = threshold.Rescale_mean(Data_thresh)
        if rescale_text == "z-score":
            Rescaled_data = threshold.Rescale_zscore(Data_thresh)
        if rescale_text == "log-mean":
            Rescaled_data = np.log(1 + Data_thresh)
            Rescaled_data = Rescaled_data - np.mean(Rescaled_data, axis=0)
        if rescale_text == "None":
            Rescaled_data = Data_thresh

        smooth_type = self.parameters["smooth_type"].value
        if smooth_type == "local":
            kernel_type = "local"
            unit_cell_shape = None
        else:
            kernel_type = "periodic"

            unit_cell_shape = []
            for nxq in self.data.nxaxes[1:]:
                q = nxq.nxvalue
                x = np.min(q[q % 1 == 0])
                l = len(nxq[x : x + 1]) - 1
                unit_cell_shape.append(l)
            unit_cell_shape = np.array(unit_cell_shape)
        L_scale = self.parameters["L_scale"].value

        Data_for_GMM = Rescaled_data.transpose()
        Markov_matrix = GMM_kernels.Build_Markov_Matrix(
            self.Data_ind, L_scale, kernel_type, unit_cell_shape
        )

        label_smoothing_flag = True
        clusterGMM = GMM(
            Data_for_GMM, self.nc
        )  # set the data and number of clusters for GMM
        clusterGMM.RunEM(
            label_smoothing_flag, Markov_matrix
        )  # RunEM with Markov matrix

        cluster_assigns = clusterGMM.cluster_assignments

        cluster_assigns = clusterGMM.cluster_assignments
        pixel_assigns = cluster_assigns

        self.Data_thresh = Data_thresh
        self.Data_ind = Data_ind
        self.cluster_assigns = cluster_assigns
        self.pixel_assigns = pixel_assigns
        self.cluster_means = clusterGMM.means
        self.cluster_covs = [clusterGMM.cluster[i].cov for i in range(self.nc)]
        self.image_data = None
        self.enable_plots(enable=True)

    def BIC_XTEC_d(self):
        nc_min = int(self.parameters["min_nc"].value)
        nc_max = int(self.parameters["max_nc"].value)

        thresh_type = self.parameters["threshold"].value
        masked = Mask_Zeros(self.data.nxsignal.nxvalue)
        threshold = Threshold_Background(masked, threshold_type=thresh_type)

        Data_thresh = threshold.data_thresholded

        rescale_text = self.parameters["rescale"].value
        if rescale_text == "mean":
            Rescaled_data = threshold.Rescale_mean(Data_thresh)
        if rescale_text == "z-score":
            Rescaled_data = threshold.Rescale_zscore(Data_thresh)
        if rescale_text == "log-mean":
            Rescaled_data = np.log(1 + Data_thresh)
            Rescaled_data = Rescaled_data - np.mean(Rescaled_data, axis=0)

        if rescale_text == "None":
            Rescaled_data = Data_thresh

        Data_for_GMM = Rescaled_data.transpose()

        ks = np.arange(nc_min, nc_max)
        bics = []
        for k in ks:
            gm = GaussianMixture(n_components=k, covariance_type="diag")
            gm.fit(Data_for_GMM)
            bics.append(gm.bic(Data_for_GMM))
        y_data = NXdata(
            NXfield(bics, name=f"BIC"),
            NXfield(ks, name=f"N_cluster"),
            title="XTEC-D BIC score",
        )
        y_data.plot(fmt="-", lw=2, marker="o", markersize=4)

    def BIC_XTEC_s(self):
        nc_min = int(self.parameters["min_nc"].value)
        nc_max = int(self.parameters["max_nc"].value)

        thresh_type = self.parameters["threshold"].value
        masked = Mask_Zeros(self.data.nxsignal.nxvalue)
        threshold = Threshold_Background(masked, threshold_type=thresh_type)
        Peak_avg = Peak_averaging(self.data.nxsignal.nxvalue, threshold)

        Data_thresh = Peak_avg.peak_avg_data

        rescale_text = self.parameters["rescale"].value
        if rescale_text == "mean":
            Rescaled_data = threshold.Rescale_mean(Data_thresh)
        if rescale_text == "z-score":
            Rescaled_data = threshold.Rescale_zscore(Data_thresh)
        if rescale_text == "log-mean":
            Rescaled_data = np.log(1 + Data_thresh)
            Rescaled_data = Rescaled_data - np.mean(Rescaled_data, axis=0)

        if rescale_text == "None":
            Rescaled_data = Data_thresh

        Data_for_GMM = Rescaled_data.transpose()

        ks = np.arange(nc_min, nc_max)
        bics = []
        for k in ks:
            gm = GaussianMixture(n_components=k, covariance_type="diag")
            gm.fit(Data_for_GMM)
            bics.append(gm.bic(Data_for_GMM))
        y_data = NXdata(
            NXfield(bics, name=f"BIC"),
            NXfield(ks, name=f"N_cluster"),
            title="XTEC-S (peak avg) BIC score",
        )
        y_data.plot(fmt="-", lw=2, marker="o", markersize=4)

    def plot_qmap(self):
        if not self.image_data:
            cluster_image = np.zeros(self.data[0].shape)

            for i in range(self.nc):
                cluster_mask = self.pixel_assigns == i
                c_ind = self.Data_ind[cluster_mask]
                c_ind = tuple(np.array(c_ind).T)
                cluster_image[c_ind] = i + 1

            image_data = NXdata(
                NXfield(cluster_image, name="cluster_map"),
                self.data.nxaxes[1:],
                title="Cluster Q Map",
            )
            self.image_data = image_data
        self.image_data.plot(cmap="xtec")

    def plot_trajectories(self):
        cluster_means = self.cluster_means
        cluster_covs = self.cluster_covs
        rescale = self.parameters["rescale"].value
        xc = self.data.nxaxes[0]
        for i in range(self.nc):
            yc = cluster_means[i]
            yc_std = (cluster_covs[i]) ** 0.5
            yc_data = NXdata(
                NXfield(yc, name=f"cluster{i+1}"),
                xc,
                long_name=f"Intensity: rescaled = {rescale}",
                title="Cluster Means and Variances",
            )
            if i == 0:
                yc_data.plot(
                    fmt="-", color=f"C{i}", lw=3, marker="o", markersize=3
                )
            else:
                yc_data.oplot(
                    fmt="-", color=f"C{i}", lw=3, marker="o", markersize=3
                )
            self.plotview.ax.fill_between(
                xc.nxvalue, yc - yc_std, yc + yc_std, color=f"C{i}", alpha=0.4
            )

    def plot_average_intensities(self):
        xc = self.data.nxaxes[0]
        for i in range(self.nc):
            cluster_mask = self.cluster_assigns == i
            data_c = self.Data_thresh[:, cluster_mask]
            yc = NXfield(
                np.mean(data_c, axis=1),
                name=f"cluster{i+1}",
                long_name="Cluster Intensity",
            )
            yc_data = NXdata(yc, xc, title="Average Intensity in Cluster")
            if i == 0:
                yc_data.plot(fmt="-", lw=1, marker="o", color=f"C{i}")
            else:
                yc_data.oplot(fmt="-", lw=1, marker="o", color=f"C{i}")
