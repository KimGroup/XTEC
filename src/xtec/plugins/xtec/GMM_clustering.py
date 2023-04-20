import numpy as np
from nexpy.gui.datadialogs import GridParameters, NXDialog
from nexpy.gui.utils import display_message, report_error
from nexpy.gui.pyqt import QtWidgets
from nexpy.gui.widgets import NXLabel, NXSpinBox
from nexusformat.nexus import NeXusError, NXdata, NXfield
from sklearn.mixture import GaussianMixture
from xtec.Preprocessing import Mask_Zeros, Peak_averaging, Threshold_Background
from xtec.GMM import GMM


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

        self.setWindowTitle('XTEC fit')

    def choose_data(self):
        if self.layout.count() == 2:
            self.parameters = GridParameters()
            self.parameters.add('n_cluster', 4, 'Number of Clusters')
            self.parameters.add('threshold', ['KL', 'No threshold'],
                                'Background Threshold')
            self.parameters.add('rescale',
                                ['mean', 'z-score', 'log-mean', 'None'],
                                'Rescaling')
            axis_grid = QtWidgets.QGridLayout()
            axis_grid.setSpacing(10)
            headers = ['Axis', 'Minimum', 'Maximum']
            width = [50, 100, 100]
            column = 0
            for header in headers:
                label = NXLabel(header, bold=True, align='center')
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
                axis_grid.addWidget(self.label(self.axes[axis].nxname),
                                    row, 0)
                axis_grid.addWidget(self.minbox[axis], row, 1)
                axis_grid.addWidget(self.maxbox[axis], row, 2)

            self.insert_layout(1, self.parameters.grid())
            self.insert_layout(2, axis_grid)
            self.insert_layout(3, self.action_buttons(('XTEC-d', self.XTEC_d),
                                                      ('XTEC-s', self.XTEC_s)))
            self.insert_layout(4, self.action_buttons(
                ('Plot Q Map', self.plot_qmap),
                ('Plot Trajectories', self.plot_trajectories),
                ('Plot Average Intensity', self.plot_average_intensities)))
        else:
            display_message('Cannot change data after initialization',
                            'Please relaunch dialog')
            return

        for axis in range(self.ndim):
            self.minbox[axis].data = self.maxbox[axis].data = \
                self.axes[axis].nxvalue
            self.minbox[axis].setMaximum(self.minbox[axis].data.size-1)
            self.maxbox[axis].setMaximum(self.maxbox[axis].data.size-1)
            self.minbox[axis].diff = self.maxbox[axis].diff = None
            self.minbox[axis].setValue(self.minbox[axis].data.min())
            self.maxbox[axis].setValue(self.maxbox[axis].data.max())

        self.enable_plots(enable=False)

    @property
    def data(self):
        limits = []
        for axis in range(self.selected_data.ndim):
            limits.append(slice(self.minbox[axis].value(),
                                self.maxbox[axis].value()))
        return self.selected_data[limits]

    @property
    def ndim(self):
        return self.selected_data.ndim

    @property
    def axes(self):
        return self.selected_data.nxaxes

    def enable_plots(self, enable=True):
        self.pushbutton['Plot Q Map'].setEnabled(enable)
        self.pushbutton['Plot Trajectories'].setEnabled(enable)
        self.pushbutton['Plot Average Intensity'].setEnabled(enable)

    def XTEC_d(self):
        self.enable_plots(enable=False)
        self.nc = int(self.parameters['n_cluster'].value)
        thresh_type = self.parameters['threshold'].value
        masked = Mask_Zeros(self.data.nxsignal.nxvalue)
        threshold = Threshold_Background(masked, threshold_type=thresh_type)
        Data_thresh = threshold.data_thresholded
        Data_ind = threshold.ind_thresholded
        self.Data_thresh = Data_thresh
        self.Data_ind = Data_ind

        rescale_text = self.parameters['rescale'].value
        if (rescale_text == 'mean'):
            Rescaled_data = threshold.Rescale_mean(Data_thresh)
        if (rescale_text == 'z-score'):
            Rescaled_data = threshold.Rescale_zscore(Data_thresh)
        if (rescale_text == 'log-mean'):
            Rescaled_data = np.log(1+Data_thresh)
            Rescaled_data = Rescaled_data-np.mean(Rescaled_data, axis=0)
        if (rescale_text == 'None'):
            Rescaled_data = Data_thresh

        Data_for_GMM = Rescaled_data.transpose()
        gm = GaussianMixture(n_components=self.nc, covariance_type='diag',
                             random_state=0).fit(Data_for_GMM)
        cluster_assigns = gm.predict(Data_for_GMM)

        self.cluster_means = gm.means_
        self.cluster_covs = gm.covariances_
        self.cluster_assigns = cluster_assigns
        self.pixel_assigns = cluster_assigns
        self.image_data = None
        self.enable_plots(enable=True)

    def XTEC_s(self):
        self.enable_plots(enable=False)
        self.nc = int(self.parameters['n_cluster'].value)
        thresh_type = self.parameters["threshold"].value
        masked = Mask_Zeros(self.data.nxsignal.nxvalue)
        threshold = Threshold_Background(masked, threshold_type=thresh_type)

        Peak_avg = Peak_averaging(self.data.nxsignal.nxvalue, threshold)
        Peak_avg_data = (Peak_avg.peak_avg_data)
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

        Data_for_GMM = (Rescaled_data.transpose())
        clusterGMM = GMM(Data_for_GMM, self.nc)
        clusterGMM.RunEM()
        clusterGMM.Get_pixel_labels(Peak_avg)
        Data_ind = (clusterGMM.Data_ind)
        cluster_assigns = (clusterGMM.cluster_assignments)
        pixel_assigns = (clusterGMM.Pixel_assignments)

        self.Data_thresh = Data_thresh
        self.Data_ind = Data_ind
        self.cluster_assigns = cluster_assigns
        self.pixel_assigns = pixel_assigns
        self.cluster_means = clusterGMM.means
        self.cluster_covs = [clusterGMM.cluster[i].cov for i in range(self.nc)]

        self.image_data = None

        self.enable_plots(enable=True)

    def plot_qmap(self):
        if (not self.image_data):
            cluster_image = np.zeros(self.data[0].shape)

            for i in range(self.nc):
                cluster_mask = (self.pixel_assigns == i)
                c_ind = self.Data_ind[cluster_mask]
                c_ind = tuple(np.array(c_ind).T)
                cluster_image[c_ind] = i + 1

            image_data = NXdata(NXfield(cluster_image, name='cluster_map'),
                                self.data.nxaxes[1:], title='Cluster Q Map')
            self.image_data = image_data
        self.image_data.plot(cmap='xtec')

    def plot_trajectories(self):
        cluster_means = self.cluster_means
        cluster_covs = self.cluster_covs
        rescale = self.parameters['rescale'].value
        xc = self.data.nxaxes[0]
        for i in range(self.nc):
            yc = NXfield(cluster_means[i], name=f'cluster{i+1}',
                         long_name=f'Intensity: rescaled = {rescale}')
            yc_std = (cluster_covs[i])**0.5
            yc_data = NXdata(NXfield(yc, name=f'cluster{i+1}'), xc,
                             title='Cluster Means and Variances')
            if i == 0:
                yc_data.plot(fmt='-', color=f'C{i}',
                             lw=3, marker='o', markersize=3)
            else:
                yc_data.oplot(fmt='-', color=f'C{i}',
                              lw=3, marker='o', markersize=3)
            self.plotview.ax.fill_between(xc.nxvalue, yc-yc_std, yc+yc_std,
                                          color=f'C{i}', alpha=0.4)

    def plot_average_intensities(self):
        xc = self.data.nxaxes[0]
        for i in range(self.nc):
            cluster_mask = (self.cluster_assigns == i)
            data_c = self.Data_thresh[:, cluster_mask]
            yc = NXfield(np.mean(data_c, axis=1), name=f'cluster{i+1}',
                         long_name='Cluster Intensity')
            yc_data = NXdata(yc, xc, title='Average Intensity in Cluster')
            if i == 0:
                yc_data.plot(fmt='-', lw=1, marker='o', color=f'C{i}')
            else:
                yc_data.oplot(fmt='-', lw=1, marker='o', color=f'C{i}')
