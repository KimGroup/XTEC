import numpy as np
from nexpy.gui.datadialogs import GridParameters, NXDialog
from nexpy.gui.utils import display_message, report_error
from nexpy.gui.pyqt import QtWidgets
from nexpy.gui.widgets import NXLabel, NXSpinBox
from nexusformat.nexus import NeXusError, NXdata, NXfield
from sklearn.mixture import GaussianMixture
from xtec.Preprocessing import Mask_Zeros, Peak_averaging, Threshold_Background


def show_dialog():
    try:
        dialog = BICDialog()
        dialog.show()
    except NeXusError as error:
        report_error("XTEC analysis", error)


class BICDialog(NXDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.select_data(self.choose_data)
        self.set_layout(self.entry_layout, self.close_buttons(close=True))

        self.setWindowTitle("Estimate cluster number")

    def choose_data(self):
        if self.layout.count() == 2:
            self.parameters = GridParameters()
            self.parameters.add("min_nc", int(2), "Min cluster num")
            self.parameters.add("max_nc", int(14), "Max cluster num")
            self.parameters.add(
                "threshold", ["KL", "No threshold"], "Background Threshold"
            )
            self.parameters.add(
                "rescale", ["mean", "z-score", "log-mean", "None"], "Rescaling"
            )
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
                axis_grid.addWidget(self.label(self.axes[axis].nxname), row, 0)
                axis_grid.addWidget(self.minbox[axis], row, 1)
                axis_grid.addWidget(self.maxbox[axis], row, 2)

            self.insert_layout(1, self.parameters.grid())
            self.insert_layout(2, axis_grid)
            self.insert_layout(
                3,
                self.action_buttons(
                    ("BIC XTEC-d", self.BIC_XTEC_d),
                    ("BIC XTEC-s", self.BIC_XTEC_s),
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

        self.title = "XTEC-D BIC score"

        ks = np.arange(nc_min, nc_max)
        bics = []
        for k in ks:
            gm = GaussianMixture(n_components=k, covariance_type="diag")
            gm.fit(Data_for_GMM)
            bics.append(gm.bic(Data_for_GMM))

        y_data = NXdata(
            NXfield(bics, name=f"BIC"),
            NXfield(ks, name=f"N_cluster"),
            title=self.title,
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

        self.title = "XTEC-S BIC score"

        ks = np.arange(nc_min, nc_max)
        bics = []
        for k in ks:
            gm = GaussianMixture(n_components=k, covariance_type="diag")
            gm.fit(Data_for_GMM)
            bics.append(gm.bic(Data_for_GMM))
        y_data = NXdata(
            NXfield(bics, name=f"BIC"),
            NXfield(ks, name=f"N_cluster"),
            title=self.title,
        )
        y_data.plot(fmt="-", lw=2, marker="o", markersize=4)
