"""Function with useful service widgets, like progressbars etc...

"""

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


class ProgressBar(QtWidgets.QWidget):
    """Simple progressbar widget."""
    def __init__(self, parent=None, total1=20):
        super().__init__(parent)
        self.name_line1  =  QtWidgets.QLineEdit()

        self.progressbar1  =  QtWidgets.QProgressBar()
        self.progressbar1.setMinimum(1)
        self.progressbar1.setMaximum(total1)

        main_layout  =  QtWidgets.QGridLayout()
        main_layout.addWidget(self.progressbar1, 0, 0)

        self.setLayout(main_layout)
        self.setWindowTitle("Progress")
        self.setGeometry(500, 300, 300, 50)

    def update_progressbar1(self, val1):
        """Update progressbar."""
        self.progressbar1.setValue(val1)
        QtWidgets.qApp.processEvents()


class ProgressBarDouble(QtWidgets.QWidget):
    """Double Progressbar widget."""
    def __init__(self, parent=None, total1=20, total2=20):
        super().__init__(parent)
        self.name_line1  =  QtWidgets.QLineEdit()

        self.progressbar1  =  QtWidgets.QProgressBar()
        self.progressbar1.setMinimum(1)
        self.progressbar1.setMaximum(total1)

        self.progressbar2  =  QtWidgets.QProgressBar()
        self.progressbar2.setMinimum(1)
        self.progressbar2.setMaximum(total2)

        main_layout  =  QtWidgets.QGridLayout()
        main_layout.addWidget(self.progressbar1, 0, 0)
        main_layout.addWidget(self.progressbar2, 1, 0)

        self.setLayout(main_layout)
        self.setWindowTitle("Progress")
        self.setGeometry(500, 300, 300, 50)

    def update_progressbar1(self, val1):
        """Update progressbar 1."""
        self.progressbar1.setValue(val1)
        QtWidgets.qApp.processEvents()

    def update_progressbar2(self, val2):
        """Update progressbar 2."""
        self.progressbar2.setValue(val2)
        QtWidgets.qApp.processEvents()


class ChannelNumber(QtWidgets.QDialog):
    """Popup tool to input the channel number to work on"""
    def __init__(self, qsd, parent=None):
        super().__init__(parent)

        ksf_h  =  np.load('keys_size_factor.npy')[0]
        ksf_w  =  np.load('keys_size_factor.npy')[1]

        ch_numb_lbl  =  QtWidgets.QLabel("Spots channel", self)
        ch_numb_lbl.setFixedSize(int(ksf_h * 140), int(ksf_w * 25))

        choose_channel_combo  =  QtWidgets.QComboBox(self)
        # for k in range(int(qsd)):
        #     choose_channel_combo.addItem(str(k + 1))
        for k in qsd:
            choose_channel_combo.addItem(k)
        choose_channel_combo.setCurrentIndex(0)
        choose_channel_combo.setFixedSize(int(ksf_h * 200), int(ksf_w * 25))

        input_close_btn  =  QtWidgets.QPushButton("Ok", self)
        input_close_btn.clicked.connect(self.input_close)
        input_close_btn.setToolTip('Input values')
        input_close_btn.setFixedSize(int(ksf_h * 50), int(ksf_w * 25))

        ch_numb_lbl_edit_box  =  QtWidgets.QHBoxLayout()
        ch_numb_lbl_edit_box.addWidget(ch_numb_lbl)
        ch_numb_lbl_edit_box.addWidget(choose_channel_combo)

        input_close_box  =  QtWidgets.QHBoxLayout()
        input_close_box.addStretch()
        input_close_box.addWidget(input_close_btn)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addLayout(ch_numb_lbl_edit_box)
        layout.addLayout(input_close_box)

        self.choose_channel_combo  =  choose_channel_combo

        self.setWindowModality(Qt.ApplicationModal)
        self.setLayout(layout)
        self.setGeometry(300, 300, 350, 50)
        self.setWindowTitle("Spot Nuc Max Distance")

    def input_close(self):
        """Close"""
        self.close()

    def ch_numb(self):
        """Return the channel number."""
        # return int(self.choose_channel_combo.currentText())
        return int(self.choose_channel_combo.currentIndex())

    @staticmethod
    def getNumb(parent=None):
        """For signal sending."""
        dialog   =  ChannelNumber(parent)
        result   =  dialog.exec_()
        ch_numb  =  dialog.ch_numb()
        return ch_numb


class InputPixSize(QtWidgets.QDialog):
    """Popup tool to input pixel sizes in xy and z."""
    def __init__(self, parent=None):
        super().__init__(parent)

        ksf_h  =  np.load('keys_size_factor.npy')[0]
        ksf_w  =  np.load('keys_size_factor.npy')[1]

        pix_sizexy_lbl  =  QtWidgets.QLabel("Pix size x-y", self)
        pix_sizexy_lbl.setFixedSize(int(ksf_h * 140), int(ksf_w * 25))

        pix_sizez_lbl  =  QtWidgets.QLabel("Pix size z", self)
        pix_sizez_lbl.setFixedSize(int(ksf_h * 140), int(ksf_w * 25))

        pix_sizexy_edt  =  QtWidgets.QLineEdit(self)
        pix_sizexy_edt.setToolTip("Set pixels size in x-y")
        pix_sizexy_edt.setFixedSize(int(ksf_h * 30), int(ksf_w * 22))
        pix_sizexy_edt.textChanged[str].connect(self.pix_sizexy_var)

        pix_sizez_edt  =  QtWidgets.QLineEdit(self)
        pix_sizez_edt.setToolTip("Set pixels size in z")
        pix_sizez_edt.setFixedSize(int(ksf_h * 30), int(ksf_w * 22))
        pix_sizez_edt.textChanged[str].connect(self.pix_sizez_var)

        input_close_btn  =  QtWidgets.QPushButton("Ok", self)
        input_close_btn.clicked.connect(self.input_close)
        input_close_btn.setToolTip('Input values')
        input_close_btn.setFixedSize(int(ksf_h * 50), int(ksf_w * 25))

        pix_sizexy_lbl_edit_box  =  QtWidgets.QHBoxLayout()
        pix_sizexy_lbl_edit_box.addWidget(pix_sizexy_lbl)
        pix_sizexy_lbl_edit_box.addWidget(pix_sizexy_edt)

        pix_sizez_lbl_edit_box  =  QtWidgets.QHBoxLayout()
        pix_sizez_lbl_edit_box.addWidget(pix_sizez_lbl)
        pix_sizez_lbl_edit_box.addWidget(pix_sizez_edt)

        input_close_box  =  QtWidgets.QHBoxLayout()
        input_close_box.addStretch()
        input_close_box.addWidget(input_close_btn)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addLayout(pix_sizexy_lbl_edit_box)
        layout.addLayout(pix_sizez_lbl_edit_box)
        layout.addLayout(input_close_box)

        self.setWindowModality(Qt.ApplicationModal)
        self.setLayout(layout)
        self.setGeometry(300, 300, 70, 50)
        self.setWindowTitle("Spot Nuc Max Distance")

    def pix_sizexy_var(self, text):
        """Input the pixel size in xy."""
        self.pix_sizexy_value  =  float(text)

    def pix_sizez_var(self, text):
        """Input the pixel size in z."""
        self.pix_sizez_value  =  float(text)

    def input_close(self):
        """Input both pixel size values."""
        self.close()

    def pix_sizes(self):
        """Output the pixel size values."""
        return [self.pix_sizexy_value, self.pix_sizez_value]

    @staticmethod
    def get_vals(parent=None):
        """Send the output."""
        dialog     =  InputPixSize(parent)
        result     =  dialog.exec_()
        pix_sizes  =  dialog.pix_sizes()
        return pix_sizes


class InputScalar(QtWidgets.QDialog):
    """Dialog to input a scalar number."""
    def __init__(self, text_list, parent=None):
        super().__init__(parent)

        text_label  =  text_list[0]
        text_btn1   =  text_list[1]
        text_btn2   =  text_list[2]
        text_tip1   =  text_list[3]
        text_tip2   =  text_list[4]
        text_title  =  text_list[5]

        ksf_h  =  np.load('keys_size_factor.npy')[0]
        ksf_w  =  np.load('keys_size_factor.npy')[1]

        input_val_lbl  =  QtWidgets.QLabel(text_label, self)
        input_val_lbl.setFixedSize(int(ksf_h * 10 * len(text_label)), int(ksf_w * 22))

        input_val_edt  =  QtWidgets.QLineEdit(text_btn1, self)
        input_val_edt.setToolTip(text_tip1)
        input_val_edt.setFixedSize(int(ksf_h * 35), int(ksf_w * 25))
        input_val_edt.textChanged[str].connect(self.input_var)

        input_val_box  =  QtWidgets.QHBoxLayout()
        input_val_box.addWidget(input_val_lbl)
        input_val_box.addStretch()
        input_val_box.addWidget(input_val_edt)

        insert_btn  =  QtWidgets.QPushButton(text_btn2, self)
        insert_btn.setToolTip(text_tip2)
        insert_btn.setFixedSize(int(ksf_h * 50), int(ksf_w * 25))
        insert_btn.clicked.connect(self.insert)

        insert_box  =  QtWidgets.QHBoxLayout()
        insert_box.addStretch()
        insert_box.addWidget(insert_btn)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addLayout(input_val_box)
        layout.addLayout(insert_box)

        self.setWindowModality(Qt.ApplicationModal)
        self.setLayout(layout)
        self.setGeometry(300, 300, int(ksf_h * 200), int(ksf_w * 25))
        self.setWindowTitle(text_title)

    def input_var(self, text):
        """Choose flag first."""
        self.input_value  =  float(text)

    def insert(self):
        """Choose flag second."""
        self.close()

    def params(self):
        """Function to send choice."""
        return self.input_value

    @staticmethod
    def getFlag(parent=None):
        """Send choice."""
        dialog  =  InputScalar(parent)
        result  =  dialog.exec_()
        flag    =  dialog.params()
        return flag


class PresmoothingFlag(QtWidgets.QDialog):
    """Popup tool to input the channel number to work on"""
    def __init__(self, parent=None):
        super().__init__(parent)

        ksf_h  =  np.load('keys_size_factor.npy')[0]
        ksf_w  =  np.load('keys_size_factor.npy')[1]

        presmoothing_btn  =  QtWidgets.QPushButton("Presmoothing", self)
        presmoothing_btn.clicked.connect(self.presmoothing)
        presmoothing_btn.setToolTip('Presmoothing')
        presmoothing_btn.setFixedSize(int(ksf_h * 130), int(ksf_w * 30))

        nopresmoothing_btn  =  QtWidgets.QPushButton("No-Presmoothing", self)
        nopresmoothing_btn.clicked.connect(self.nopresmoothing)
        nopresmoothing_btn.setToolTip('No-Presmoothing')
        nopresmoothing_btn.setFixedSize(int(ksf_h * 130), int(ksf_w * 30))

        layout  =  QtWidgets.QHBoxLayout()
        layout.addWidget(presmoothing_btn)
        layout.addWidget(nopresmoothing_btn)

        self.smooth_presmooth_flag  =  None

        self.setWindowModality(Qt.ApplicationModal)
        self.setLayout(layout)
        self.setGeometry(300, 300, 70, 50)
        self.setWindowTitle("Spot Nuc Max Distance")

    def input_close(self):
        """Close"""
        self.close()

    def pre_nopre_flag(self):
        """Return smoothing flag."""
        return self.smooth_presmooth_flag

    def presmoothing(self):
        """Run presmoothing signal."""
        self.smooth_presmooth_flag  =  True
        self.input_close()

    def nopresmoothing(self):
        """Don't run presmoothing signal."""
        self.smooth_presmooth_flag  =  False
        self.input_close()


    @staticmethod
    def getFlag(parent=None):
        """For signal sending."""
        dialog   =  PresmoothingFlag(parent)
        result   =  dialog.exec_()
        flag     =  dialog.pre_nopre_flag()
        return flag
