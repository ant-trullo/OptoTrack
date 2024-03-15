"""This is the main window of the software to detect and follow TS without nuclei in drosophila embryo.
This is version 1.0, since July 2022.

developer: antonio.trullo@igmm.cnrs.fr

"""

import sys
import shutil
import os.path
import time
from importlib import reload
import traceback
from natsort import natsorted
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5 import QtGui, QtWidgets, QtCore

import LoadRawData
import AnalysisSaver
import SpotsDetectionChopper
import SpotsTracker
import SpotsFeatureExtractor
import SpotsDetection3D
import AnalysisLoader
import GalleryDividedByBckg
import SpatiallySelectedSaver
import PhotoBleachingEstimate
import GenerateFigureCircledSpots
import SaveReadMatrix
import ServiceWidgets
# import SegmentNucsClstr
import GenerateHullImage


class MainWindow(QtWidgets.QMainWindow):
    """Main windows: coordinates all the actions, algorithms, visualization tools and analysis tools."""
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        ksf_h  =  np.load('keys_size_factor.npy')[0]
        ksf_w  =  np.load('keys_size_factor.npy')[1]

        widget = QtWidgets.QWidget(self)
        self.setCentralWidget(widget)

        load_data_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/load-hi.png'), "&Load Data", self)
        load_data_action.setShortcut("Ctrl+L")
        load_data_action.setStatusTip("Load raw data files with a single channel")
        load_data_action.triggered.connect(self.load_raw_data)

        load_analysis_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/load-hi.png'), "&Load Analysis", self)
        load_analysis_action.setShortcut("Ctrl+A")
        load_analysis_action.setStatusTip("Load Analysis")
        load_analysis_action.triggered.connect(self.load_analysis)

        save_spts_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/save-md.png'), "&Save Analysis", self)
        save_spts_action.setShortcut("Ctrl+S")
        save_spts_action.setStatusTip("Save the analysis in a folder")
        save_spts_action.triggered.connect(self.save_analysis)

        settings_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/settings.png'), "&Settings", self)
        settings_action.setShortcut("Ctrl+T")
        settings_action.setStatusTip("Changes default settings values")
        settings_action.triggered.connect(self.settings_changes)

        test_spots_detection_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/sccpre.png'), "&Test Spots Detection", self)
        test_spots_detection_action.setStatusTip("Test the spots detection on a sample of your data")
        test_spots_detection_action.triggered.connect(self.test_spots_detection)
        test_spots_detection_action.setShortcut("Ctrl+M")

        chop_stack_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/scissors.png'), "&Chop Stack", self)
        chop_stack_action.setShortcut("Ctrl+k")
        chop_stack_action.setStatusTip("Chop Stack")
        chop_stack_action.triggered.connect(self.chop_stack)

        crop_stack_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/crop.png'), "&Crop Stack", self)
        crop_stack_action.setShortcut("Ctrl+C")
        crop_stack_action.setStatusTip("Crop the stack before the analysis")
        crop_stack_action.triggered.connect(self.crop_stack)

        show_gallery_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/gallery.png'), "&Show Gallery", self)
        show_gallery_action.setStatusTip("Show gallery of spots traces intensity divided by the background")
        show_gallery_action.triggered.connect(self.show_gallery)
        show_gallery_action.setShortcut("Ctrl+O")

        spatial_analysis_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/spatial_spots.jpg'), "&Spatial Analysis", self)
        spatial_analysis_action.setStatusTip("Organize data in a manually selected spatial region")
        spatial_analysis_action.triggered.connect(self.spatial_analysis)
        spatial_analysis_action.setShortcut("Ctrl+H")

        generate_figure_circledspots_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/circled_spots.jpg'), "&Generate Figure", self)
        generate_figure_circledspots_action.setStatusTip("Generate figures with raw data on the background and a circle around all the spots")
        generate_figure_circledspots_action.triggered.connect(self.generate_figure_circledspots)
        generate_figure_circledspots_action.setShortcut("Ctrl+D")

        check_photobleaching_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/bleaching.jpeg'), "&Check Photobleaching", self)
        check_photobleaching_action.setStatusTip("Check Photobleaching on the analysis you run")
        check_photobleaching_action.triggered.connect(self.check_photobleaching)
        check_photobleaching_action.setShortcut("Ctrl+w")

        hull_image_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/convex_hull.png'), "&Convex Hull Image", self)
        hull_image_action.setStatusTip("Generate convex hull image")
        hull_image_action.triggered.connect(self.hull_image)

        exit_action  =  QtWidgets.QAction(QtGui.QIcon('Icons/exit.png'), "&Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)

        menubar   =  self.menuBar()

        file_menu  =  menubar.addMenu("&File")
        file_menu.addAction(load_data_action)
        file_menu.addAction(save_spts_action)
        file_menu.addAction(load_analysis_action)
        file_menu.addAction(settings_action)
        file_menu.addAction(exit_action)

        modify_menu  =  menubar.addMenu('&Modify')
        modify_menu.addAction(chop_stack_action)
        modify_menu.addAction(crop_stack_action)

        test_menu  =  menubar.addMenu('&Test Analysis')
        test_menu.addAction(test_spots_detection_action)

        postprocessing_menu  =  menubar.addMenu('&Post-Processing')
        postprocessing_menu.addAction(show_gallery_action)
        postprocessing_menu.addAction(spatial_analysis_action)
        postprocessing_menu.addAction(check_photobleaching_action)
        postprocessing_menu.addAction(generate_figure_circledspots_action)
        postprocessing_menu.addAction(hull_image_action)

        fname_raw_lbl  =  QtWidgets.QLineEdit("File: ", self)
        fname_raw_lbl.setToolTip("Names of the files you are working on")

        busy_lbl  =  QtWidgets.QLabel("Ready")
        busy_lbl.setStyleSheet("color: green")

        pixsize_x_lbl  =  QtWidgets.QLabel("pix size XY =;")
        pixsize_z_lbl  =  QtWidgets.QLabel("Z step =;")
        time_step_lbl  =  QtWidgets.QLabel("Time step =")

        bottom_labels_box  =  QtWidgets.QHBoxLayout()
        bottom_labels_box.addWidget(busy_lbl)
        bottom_labels_box.addStretch()
        bottom_labels_box.addWidget(pixsize_x_lbl)
        bottom_labels_box.addWidget(pixsize_z_lbl)
        bottom_labels_box.addWidget(time_step_lbl)

        frame_raw_mip  =  pg.ImageView(self, name="RawMip")
        frame_raw_mip.ui.roiBtn.hide()
        frame_raw_mip.ui.menuBtn.hide()
        frame_raw_mip.view.setXLink("SegmMip")
        frame_raw_mip.view.setYLink("SegmMip")
        frame_raw_mip.timeLine.sigPositionChanged.connect(self.update_frame_from_raw_mip)

        frame_segm_mip  =  pg.ImageView(self, name="SegmMip")
        frame_segm_mip.ui.roiBtn.hide()
        frame_segm_mip.ui.menuBtn.hide()
        frame_segm_mip.view.setYLink("RawMip")
        frame_segm_mip.view.setXLink("RawMip")
        frame_segm_mip.timeLine.sigPositionChanged.connect(self.update_frame_from_segm_mip)

        tabs_mip      =  QtWidgets.QTabWidget()
        tab_mip_raw   =  QtWidgets.QWidget()
        tab_mip_segm  =  QtWidgets.QWidget()

        tabs_mip.addTab(tab_mip_raw, "raw")
        tabs_mip.addTab(tab_mip_segm, "Segmented")

        frame_raw_mip_box  =  QtWidgets.QHBoxLayout()
        frame_raw_mip_box.addWidget(frame_raw_mip)

        frame_segm_mip_box  =  QtWidgets.QHBoxLayout()
        frame_segm_mip_box.addWidget(frame_segm_mip)

        tab_mip_raw.setLayout(frame_raw_mip_box)
        tab_mip_segm.setLayout(frame_segm_mip_box)

        tabs_mip_sld_box  =  QtWidgets.QVBoxLayout()
        tabs_mip_sld_box.addWidget(tabs_mip)

        tabs_box  =  QtWidgets.QHBoxLayout()
        tabs_box.addLayout(tabs_mip_sld_box)

        detect_spts_btn  =  QtWidgets.QPushButton("Detect Spots", self)
        detect_spts_btn.clicked.connect(self.detect_spts)
        detect_spts_btn.setToolTip("Detect Spots")
        detect_spts_btn.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))
        detect_spts_btn.setEnabled(True)

        merge_radius_lbl  =  QtWidgets.QLabel("Merge Rad", self)
        merge_radius_lbl.setFixedSize(int(ksf_h * 95), int(ksf_w * 25))

        merge_radius_edt  =  QtWidgets.QLineEdit(self)
        merge_radius_edt.textChanged[str].connect(self.merge_radius_var)
        merge_radius_edt.setToolTip("Set the maximmum distance between 2 spots to merge them; suggested value 5")
        merge_radius_edt.setFixedSize(int(ksf_h * 35), int(ksf_w * 25))

        merge_radius_edt_lbl_box  =  QtWidgets.QHBoxLayout()
        merge_radius_edt_lbl_box.addWidget(merge_radius_lbl)
        merge_radius_edt_lbl_box.addWidget(merge_radius_edt)

        spts_thr_lbl  =  QtWidgets.QLabel("Spts Thr", self)
        spts_thr_lbl.setFixedSize(int(ksf_h * 95), int(ksf_w * 25))

        spts_thr_edt  =  QtWidgets.QLineEdit(self)
        spts_thr_edt.textChanged[str].connect(self.spts_thr_var)
        spts_thr_edt.setToolTip("Sets the detection threshold for spots; suggested value 7")
        spts_thr_edt.setFixedSize(int(ksf_h * 35), int(ksf_w * 25))

        spts_thr_edt_lbl_box  =  QtWidgets.QHBoxLayout()
        spts_thr_edt_lbl_box.addWidget(spts_thr_lbl)
        spts_thr_edt_lbl_box.addWidget(spts_thr_edt)

        min_volthr_lbl  =  QtWidgets.QLabel("Vol Thr", self)
        min_volthr_lbl.setFixedSize(int(ksf_h * 95), int(ksf_w * 25))

        min_volthr_edt  =  QtWidgets.QLineEdit(self)
        min_volthr_edt.textChanged[str].connect(self.min_volthr_var)
        min_volthr_edt.setToolTip("Sets the minimum volume allowded for a spots; suggested value 5")
        min_volthr_edt.setFixedSize(int(ksf_h * 35), int(ksf_w * 25))

        min_volthr_edt_lbl_box  =  QtWidgets.QHBoxLayout()
        min_volthr_edt_lbl_box.addWidget(min_volthr_lbl)
        min_volthr_edt_lbl_box.addWidget(min_volthr_edt)

        dist_thr_lbl  =  QtWidgets.QLabel("Dist Thr", self)
        dist_thr_lbl.setFixedSize(int(ksf_h * 95), int(ksf_w * 25))

        dist_thr_edt  =  QtWidgets.QLineEdit(self)
        dist_thr_edt.textChanged[str].connect(self.dist_thr_var)
        dist_thr_edt.setToolTip("Sets the distance threshold for tracking; suggested value 10")
        dist_thr_edt.setFixedSize(int(ksf_h * 35), int(ksf_w * 25))

        dist_thr_edt_lbl_box  =  QtWidgets.QHBoxLayout()
        dist_thr_edt_lbl_box.addWidget(dist_thr_lbl)
        dist_thr_edt_lbl_box.addWidget(dist_thr_edt)

        track_spts_btn  =  QtWidgets.QPushButton("Track Spots", self)
        track_spts_btn.clicked.connect(self.track_spts)
        track_spts_btn.setToolTip("Track Spots")
        track_spts_btn.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))
        track_spts_btn.setEnabled(True)

        tstep_lbl  =  QtWidgets.QLabel("T step 0", self)
        tstep_lbl.setFixedSize(int(ksf_h * 95), int(ksf_w * 25))

        time_lbl  =  QtWidgets.QLabel("time " + '0', self)
        time_lbl.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        shuffle_colors_btn  =  QtWidgets.QPushButton("Shuffle Clrs", self)
        shuffle_colors_btn.clicked.connect(self.shuffle_colors)
        shuffle_colors_btn.setToolTip("Shuffle Colors")
        shuffle_colors_btn.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))
        shuffle_colors_btn.setEnabled(True)

        commands  =  QtWidgets.QVBoxLayout()
        commands.addLayout(merge_radius_edt_lbl_box)
        commands.addLayout(spts_thr_edt_lbl_box)
        commands.addLayout(min_volthr_edt_lbl_box)
        commands.addWidget(detect_spts_btn)
        commands.addLayout(dist_thr_edt_lbl_box)
        commands.addWidget(track_spts_btn)
        commands.addStretch()
        commands.addWidget(shuffle_colors_btn)
        commands.addWidget(tstep_lbl)
        commands.addWidget(time_lbl)

        tabs_commands_box  =  QtWidgets.QHBoxLayout()
        tabs_commands_box.addLayout(tabs_box)
        tabs_commands_box.addLayout(commands)

        layout  =  QtWidgets.QVBoxLayout(widget)
        layout.addWidget(fname_raw_lbl)
        layout.addLayout(tabs_commands_box)
        layout.addLayout(bottom_labels_box)

        mycmap      =  np.fromfile("mycmap.bin", "uint16").reshape((10000, 3))      # / 255.0
        colors4map  =  []
        for k in range(mycmap.shape[0]):
            colors4map.append(mycmap[k, :])
        colors4map[0]  =  np.array([0, 0, 0])

        self.frame_segm_mip    =  frame_segm_mip
        self.frame_raw_mip     =  frame_raw_mip
        self.fname_raw_lbl     =  fname_raw_lbl
        self.busy_lbl          =  busy_lbl
        self.tabs_mip          =  tabs_mip
        self.pixsize_x_lbl     =  pixsize_x_lbl
        self.pixsize_z_lbl     =  pixsize_z_lbl
        self.time_step_lbl     =  time_step_lbl
        self.tstep_lbl         =  tstep_lbl
        self.time_lbl          =  time_lbl
        self.fname_raw_lbl     =  fname_raw_lbl
        self.spts_thr_edt      =  spts_thr_edt
        self.min_volthr_edt    =  min_volthr_edt
        self.merge_radius_edt  =  merge_radius_edt
        self.dist_thr_edt      =  dist_thr_edt
        self.soft_version      =  "OptoTrack_v1.0"
        self.colors4map        =  colors4map

        self.setGeometry(800, 100, 700, 500)
        self.setWindowTitle(self.soft_version)
        self.setWindowIcon(QtGui.QIcon('Icons/MLL_Logo2.png'))
        self.show()

    def closeEvent(self, event):
        """Close the GUI, asking confirmation."""
        quit_msg  =  "Are you sure you want to exit the program?"
        reply     =  QtWidgets.QMessageBox.question(self, 'Message', quit_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def busy_indicator(self):
        """Write a red text (BUSY) as a label on the GUI (bottom left)."""
        self.busy_lbl.setText("Busy")
        self.busy_lbl.setStyleSheet('color: red')

    def ready_indicator(self):
        """Write a green text (READY) as a label on the GUI (bottom left)."""
        self.busy_lbl.setText("Ready")
        self.busy_lbl.setStyleSheet('color: green')

    def settings_changes(self):
        """Change settings."""
        self.mpp2  =  SettingsChanges()
        self.mpp2.show()
        self.mpp2.procStart.connect(self.settings_update)

    def settings_update(self):
        """Restart the GUI to make changes in button size effective."""
        self.mpp2.close()
        os.execl(sys.executable, sys.executable, *sys.argv)

    def merge_radius_var(self, text):
        """Set the spots merge radius."""
        self.merge_radius_value  =  int(text)

    def min_volthr_var(self, text):
        """Set the minimum volume threshold."""
        self.min_volthr_value  =  int(text)

    def dist_thr_var(self, text):
        """Set the distance threshold for tracking."""
        self.dist_thr_value  =  int(text)

    def spts_thr_var(self, text):
        """Set the minimum volume threshold."""
        self.spts_thr_value  =  float(text)

    def update_frame_from_raw_mip(self):
        """Update the frame of segmented mip from raw mip."""
        self.tstep_lbl.setText("T step " + str(self.frame_raw_mip.currentIndex))
        self.time_lbl.setText("time " + time.strftime("%M:%S", time.gmtime(self.frame_raw_mip.currentIndex * self.raw_data.time_step_value)))
        try:
            self.frame_segm_mip.setCurrentIndex(self.frame_raw_mip.currentIndex)
        except AttributeError:
            pass

    def update_frame_from_segm_mip(self):
        """Update the frame of raw mip from segmented mip."""
        try:
            self.frame_raw_mip.setCurrentIndex(self.frame_segm_mip.currentIndex)
        except AttributeError:
            pass

    def shuffle_colors(self):
        """Shuffle spots colors."""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            colors_bff  =  self.colors4map[1:]
            np.random.shuffle(colors_bff)
            self.colors4map[1:]  =  colors_bff
            if self.tabs_mip.tabText(1) == "Segmented":
                mycmap  =  pg.ColorMap(np.linspace(0, 1, self.spots_3d_det.spots_2d_lbls.max()), color=self.colors4map)
                self.frame_segm_mip.setColorMap(mycmap)
            elif self.tabs_mip.tabText(1) == "Tracked":
                mycmap  =  pg.ColorMap(np.linspace(0, 1, self.spots_trckd.spts_trck_fin.max()), color=self.colors4map)
                self.frame_segm_mip.setColorMap(mycmap)
        except Exception:
            traceback.print_exc()

        self.ready_indicator()

    def chop_stack(self):
        """Call the popup tool to chop the stack."""
        self.mpp3  =  ChopStack(self.raw_data.raw_data_mip, self.raw_data.time_step_value)
        self.mpp3.show()
        self.mpp3.procStart.connect(self.chop_stack_sgnl)

    def chop_stack_sgnl(self):
        """Input results of chop tool."""
        self.raw_data.raw_data_mip  =  self.raw_data.raw_data_mip[self.mpp3.first_last_frame[0]:self.mpp3.first_last_frame[1]]
        self.raw_data.raw_data      =  self.raw_data.raw_data[self.mpp3.first_last_frame[0]:self.mpp3.first_last_frame[1]]
        # self.first_last_frame       =  self.mpp3.first_last_frame
        self.mpp3.close()
        self.frame_raw_mip.setImage(self.raw_data.raw_data_mip)

    def crop_stack(self):
        """Call the popup tool to crop the stack."""
        self.mpp4  =  CropStack(self.raw_data.raw_data_mip, self.raw_data.time_step_value)
        self.mpp4.show()
        self.mpp4.procStart.connect(self.crop_stack_sgnl)

    def crop_stack_sgnl(self):
        """Input results of crop tool."""
        pts  =  self.mpp4.crop_roi_raw.parentBounds()
        x0   =  np.round(np.max([0, pts.x()])).astype(np.int64)
        y0   =  np.round(np.max([0, pts.y()])).astype(np.int64)
        x1   =  np.round(np.min([pts.x() + pts.width(), self.raw_data.raw_data_mip.shape[1]])).astype(np.int64)
        y1   =  np.round(np.min([pts.y() + pts.height(), self.raw_data.raw_data_mip.shape[2]])).astype(np.int64)
        self.raw_data.raw_data      =  self.raw_data.raw_data[:, :, x0:x1, y0:y1]
        self.raw_data.raw_data_mip  =  self.raw_data.raw_data_mip[:, x0:x1, y0:y1]
        self.frame_raw_mip.setImage(self.raw_data.raw_data_mip, autoRange=False)
        self.mpp4.close()
        self.crop_roi_raw  =  np.array([x0, y0, x1, y1])

    def load_raw_data(self):
        """Load and visualize raw_data."""
        self.fnames    =  natsorted(QtWidgets.QFileDialog.getOpenFileNames(None, "Select czi (or lsm) data files to concatenate...", filter="*.lsm *.czi *.tif *.lif")[0])
        # self.fnames    =  ['/home/atrullo/Dropbox/NucleiLessBurstingSoftware/RawData/03012022_M120x317_1_561_E2a_Out.czi']
        joined_fnames  =  ' '
        for fname in self.fnames:
            joined_fnames  +=  str(fname[fname.rfind('/') + 1:]) +  ' ~~~ '

        self.fname_raw_lbl.setText(joined_fnames)

        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            self.frame_raw_mip.clear()
            self.frame_segm_mip.clear()
            msgBox  =  QtWidgets.QMessageBox()
            msgBox.setText("You did not select any file.")
            if len(self.fnames) == 0:
                msgBox.exec()
            if self.fnames[0][-4:] == ".czi":
                self.pre_nopre_flag  =  ServiceWidgets.PresmoothingFlag.getFlag()
                if self.pre_nopre_flag:
                    # self.raw_data  =  LoadRawData.LoadRawDataCziPresmooth(self.fnames[::-1])    # THIS COMES FOR A PARTICULAR ISSUE VIRGINIA HAD
                    self.raw_data  =  LoadRawData.LoadRawDataCziPresmooth(self.fnames)
                elif not self.pre_nopre_flag:
                    # self.raw_data  =  LoadRawData.LoadRawDataCzi(self.fnames[::-1])
                    self.raw_data  =  LoadRawData.LoadRawDataCzi(self.fnames)
            elif self.fnames[0][-4:] == ".tif":
                self.raw_data  =  LoadRawData.LoadRawDataTiff(self.fnames)

            self.frame_raw_mip.setImage(self.raw_data.raw_data_mip)
            self.pixsize_x_lbl.setText("pix size XY = " + str(np.round(self.raw_data.pix_size_xy, decimals=4)) + "µm;")
            self.pixsize_z_lbl.setText("Z step = " + str(np.round(self.raw_data.pix_size_z, decimals=4)) + "µm;")
            # self.pixsize_x_lbl.setText("pix size XY = " + str(np.round(self.raw_data.pix_size_xy * 1000000, decimals=4)) + "µm;")
            # self.pixsize_z_lbl.setText("Z step = " + str(np.round(self.raw_data.pix_size_z * 1000000, decimals=4)) + "µm;")
            self.time_step_lbl.setText("Time Step = " + str(np.round(self.raw_data.time_step_value, decimals=4)) + "s")
            self.crop_roi_raw  =  [0, 0, self.raw_data.raw_data_mip.shape[1], self.raw_data.raw_data_mip.shape[2]]

        except Exception:
            traceback.print_exc()
        self.ready_indicator()

    def detect_spts(self):
        """Detect spots."""
        reload(SpotsDetectionChopper)
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            self.spots_3d_det  =  SpotsDetectionChopper.SpotsDetectionChopper(self.raw_data.raw_data, self.spts_thr_value, self.min_volthr_value, self.merge_radius_value)
            self.frame_segm_mip.setImage(self.spots_3d_det.spots_2d_lbls, autoRange=False)
            self.rnd_cmap      =  pg.ColorMap(np.linspace(0, 1, self.spots_3d_det.spots_2d_lbls.max()), color=self.colors4map)
            self.frame_segm_mip.setColorMap(self.rnd_cmap)
            self.tabs_mip.setTabText(1, "Segmented")
        except Exception:
            traceback.print_exc()
        self.ready_indicator()

    def track_spts(self):
        """Track spots in time."""
        reload(SpotsTracker)
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            self.spots_trckd  =  SpotsTracker.SpotsTracker(self.spots_3d_det.spots_2d_lbls, self.dist_thr_value)
            self.frame_segm_mip.setImage(self.spots_trckd.spts_trck_fin, autoRange=False)
            self.frame_segm_mip.setCurrentIndex(self.frame_raw_mip.currentIndex)
            self.rnd_cmap     =  pg.ColorMap(np.linspace(0, 1, self.spots_trckd.spts_trck_fin.max()), color=self.colors4map)
            self.frame_segm_mip.setColorMap(self.rnd_cmap)
            self.tabs_mip.setTabText(1, "Tracked")

        except Exception:
            traceback.print_exc()
        self.ready_indicator()

    def save_analysis(self):
        """Save the analysis."""
        reload(AnalysisSaver)
        reload(SpotsFeatureExtractor)
        # reload(GalleryDividedByBckg)
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            analysis_folder  =  str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select the Directory to save the Analysis"))
            spots_features   =  SpotsFeatureExtractor.SpotsFeatureExtractor(self.spots_trckd.spts_trck_fin, self.spots_3d_det.spots_coords, self.raw_data.raw_data)
            AnalysisSaver.AnalysisSaver(analysis_folder, self.fnames, self.raw_data, self.soft_version, self.spots_3d_det, self.spots_trckd, self.merge_radius_value, self.spts_thr_value,
                                        self.min_volthr_value, self.dist_thr_value, spots_features, self.crop_roi_raw, self.raw_data.ch_numb, self.pre_nopre_flag)
            # shutil.copyfile('gauss_kern_size.npy', '/home/atrullo/Dropbox/JamesData/NC_13th_2023_03_11_e01/gauss_kern_size.npy')
            shutil.copyfile('gauss_kern_size.npy', analysis_folder + '/gauss_kern_size.npy')
            # GalleryDividedByBckg.GalleryDividedByBckg(analysis_folder + '/SpotsAnalysis.xlsx')
        except Exception:
            traceback.print_exc()
        self.ready_indicator()

    def test_spots_detection(self):
        """Launch test spots detection tool."""
        self.mpp1  =  TestSpotsDetectionSetting(self.raw_data.raw_data_mip, self.raw_data.raw_data, self.raw_data.time_step_value)
        self.mpp1.show()
        self.mpp1.procStart.connect(self.insert_close_spots_test)

    def insert_close_spots_test(self):
        """Insert analysis parameters from the test spots analysis tool."""
        self.spts_thr_edt.setText(str(self.mpp1.spots_thr_value))
        self.min_volthr_edt.setText(str(self.mpp1.volume_thr_value))
        self.merge_radius_edt.setText(str(self.mpp1.merge_radius_value))
        self.mpp1.close()

    def load_analysis(self):
        """Load previously done analysis."""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            analysis_folder  =  str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select the Directory with Analyzed data."))
            self.fnames      =  natsorted(QtWidgets.QFileDialog.getOpenFileNames(None, "Select czi (or lsm) data files to concatenate...", filter="*.lsm *.czi *.tif *.lif")[0])
            # self.fnames      =  ['/home/atrullo/Dropbox/Virginia_Anto/spt5/16112022_spt5DARK_TEST_e2f_light_Airyscan Processing.czi']
            joined_fnames  =  ' '
            for fname in self.fnames:
                joined_fnames  +=  str(fname[fname.rfind('/') + 1:]) +  ' ~~~ '
            self.fnames  =  self.fnames
            self.fname_raw_lbl.setText(joined_fnames)
            self.frame_raw_mip.clear()
            self.frame_segm_mip.clear()

            self.raw_data            =  AnalysisLoader.RawData(analysis_folder, self.fnames)
            params                   =  AnalysisLoader.AnalysisParameters(analysis_folder)
            self.merge_radius_value  =  params.merge_radius_value
            self.dist_thr_value      =  params.dist_thr_value
            self.min_volthr_value    =  params.min_volthr_value
            self.spts_thr_value      =  params.spts_thr_value
            self.spts_thr_edt.setText(str(self.spts_thr_value))
            self.min_volthr_edt.setText(str(self.min_volthr_value))
            self.merge_radius_edt.setText(str(self.merge_radius_value))
            self.dist_thr_edt.setText(str(self.dist_thr_value))
            self.frame_raw_mip.setImage(self.raw_data.raw_data_mip)
            self.pixsize_x_lbl.setText("pix size XY = " + str(np.round(self.raw_data.pix_size_xy * 1000000, decimals=4)) + "µm;")
            self.pixsize_z_lbl.setText("Z step = " + str(np.round(self.raw_data.pix_size_z * 1000000, decimals=4)) + "µm;")
            self.time_step_lbl.setText("Time Step = " + str(np.round(self.raw_data.time_step_value, decimals=4)) + "s")
            self.crop_roi_raw  =  np.load(analysis_folder + '/crop_roi.npy')

            self.spots_3d_det  =  AnalysisLoader.SpotsDetected(analysis_folder)
            self.spots_trckd   =  AnalysisLoader.SpotsTracked(analysis_folder)
            self.frame_segm_mip.setImage(self.spots_trckd.spts_trck_fin, autoRange=False)
            self.frame_segm_mip.setCurrentIndex(self.frame_raw_mip.currentIndex)
            self.rnd_cmap      =  pg.ColorMap(np.linspace(0, 1, self.spots_trckd.spts_trck_fin.max()), color=self.colors4map)
            self.frame_segm_mip.setColorMap(self.rnd_cmap)
            self.tabs_mip.setTabText(1, "Tracked")

        except Exception:
            traceback.print_exc()
        self.ready_indicator()

    def show_gallery(self):
        """Show the gallery of spots intensity divided by background."""
        xlsx_filename  =  str(QtWidgets.QFileDialog.getOpenFileNames(None, "Select the excell file SpotsAnalysis to read traces", filter="*.xlsx")[0][0])
        GalleryDividedByBckg.GalleryDividedByBckg(xlsx_filename)

    def spatial_analysis(self):
        """Activate popup tool for sptial analysis."""
        raw_data         =  None
        analysis_folder  =  str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select the analysis folder"))
        fnames           =  natsorted(QtWidgets.QFileDialog.getOpenFileNames(None, "Select czi (or lsm) data files to concatenate...", filter="*.lsm *.czi *.tif *.lif")[0])
        raw_data          =  LoadRawData.LoadRawDataCzi(fnames)
        self.mpp_spatial  =  SpatialAnalisys(raw_data, analysis_folder)
        self.mpp_spatial.show()

    def generate_figure_circledspots(self):
        """Generate figures with mip and filtered raw data with circles around spots."""
        self.busy_indicator()
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            analysis_folder  =  str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select the Directory with Analyzed data."))
            fnames           =  natsorted(QtWidgets.QFileDialog.getOpenFileNames(None, "Select czi (or lsm) data files to concatenate...", filter="*.lsm *.czi *.tif *.lif")[0])
            raw_data_mip     =  AnalysisLoader.RawData(analysis_folder, fnames).raw_data_mip
            self.mpp5        =  GenerateImage(raw_data_mip, analysis_folder)
            self.mpp5.show()

        except Exception:
            traceback.print_exc()
        self.ready_indicator()

    def check_photobleaching(self):
        """Check photobleaching."""
        analysis_folder  =  str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select the analysis folder"))
        fnames           =  natsorted(QtWidgets.QFileDialog.getOpenFileNames(None, "Select czi (or lsm) data files to concatenate...", filter="*.lsm *.czi *.tif *.lif")[0])
        raw_data         =  AnalysisLoader.RawData(analysis_folder, fnames)
        phtblc           =  PhotoBleachingEstimate.PhotoBleachingEstimate(analysis_folder, raw_data)
        self.mpp6        =  PhotobleachingTool(phtblc, fnames, analysis_folder, self.soft_version)
        self.mpp6.show()

    def check_photobleaching2(self):
        """Check photobleaching."""
        goon_flag         =  True
        analysis_folders  =  []
        fnamess           =  []
        while goon_flag is True:
            analysis_folder  =  str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select the analysis folder"))
            fnames           =  natsorted(QtWidgets.QFileDialog.getOpenFileNames(None, "Select czi (or lsm) data files to concatenate...", filter="*.lsm *.czi *.tif *.lif")[0])
            if analysis_folder == "":
                goon_flag  =  False
            else:
                analysis_folders.append(analysis_folder)
                fnamess.append(fnames)

        phtblc           =  PhotoBleachingEstimate.PhotoBleachingEstimateSeveralAnalysis(analysis_folders, fnamess)
        self.mpp6        =  PhotobleachingTool(phtblc, fnames, analysis_folder, self.soft_version)
        self.mpp6.show()

    def hull_image(self):
        """Generate the convex hull image."""
        analysis_folder  =  str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select the Directory with Analyzed data."))
        fnames           =  natsorted(QtWidgets.QFileDialog.getOpenFileNames(None, "Select czi (or lsm) data files to concatenate...", filter="*.lsm *.czi *.tif *.lif")[0])
        GenerateHullImage.GenerateHullImage(analysis_folder, fnames)


class SettingsChanges(QtWidgets.QWidget):
    """Tool to change visualization and analysis parameters."""
    procStart  =  QtCore.pyqtSignal()

    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        self.ksf_h   =  np.load('keys_size_factor.npy')[0]
        self.ksf_w   =  np.load('keys_size_factor.npy')[1]
        self.g_kern  =  np.load('gauss_kern_size.npy').astype(np.float64)

        ksf_h_lbl  =  QtWidgets.QLabel("Keys Scale Factor W")

        ksf_h_edt  =  QtWidgets.QLineEdit(self)
        ksf_h_edt.textChanged[str].connect(self.ksf_h_var)
        ksf_h_edt.setToolTip("Sets keys scale size (width)")
        ksf_h_edt.setFixedSize(int(self.ksf_h * 50), int(self.ksf_w * 25))
        ksf_h_edt.setText(str(self.ksf_h))

        ksf_w_lbl  =  QtWidgets.QLabel("Keys Scale Factor H")

        ksf_w_edt  =  QtWidgets.QLineEdit(self)
        ksf_w_edt.textChanged[str].connect(self.ksf_w_var)
        ksf_w_edt.setToolTip("Sets keys scale size (heigth)")
        ksf_w_edt.setFixedSize(int(self.ksf_h * 50), int(self.ksf_w * 25))
        ksf_w_edt.setText(str(self.ksf_w))

        gaus_kern_lbl  =  QtWidgets.QLabel("Gauss Kernel Size")

        gaus_kern_edt  =  QtWidgets.QLineEdit(self)
        gaus_kern_edt.textChanged[str].connect(self.g_kern_var)
        gaus_kern_edt.setToolTip("Sets Gaussian filter kernel size")
        gaus_kern_edt.setFixedSize(int(self.ksf_h * 50), int(self.ksf_w * 25))
        gaus_kern_edt.setText(str(self.g_kern))

        save_btn  =  QtWidgets.QPushButton("Save", self)
        save_btn.clicked.connect(self.save_vars)
        save_btn.setToolTip('Make default the choseen parameters')
        save_btn.setFixedSize(int(self.ksf_h * 55), int(self.ksf_w * 25))

        close_btn  =  QtWidgets.QPushButton("Close", self)
        close_btn.clicked.connect(self.close_)
        close_btn.setToolTip('Close Widget')
        close_btn.setFixedSize(int(self.ksf_h * 55), int(self.ksf_w * 25))

        restart_btn  =  QtWidgets.QPushButton("Refresh", self)
        restart_btn.clicked.connect(self.restart)
        restart_btn.setToolTip('Refresh GUI')
        restart_btn.setFixedSize(int(self.ksf_h * 75), int(self.ksf_w * 25))

        btns_box  =  QtWidgets.QHBoxLayout()
        btns_box.addWidget(save_btn)
        btns_box.addWidget(restart_btn)
        btns_box.addWidget(close_btn)

        layout_grid  =  QtWidgets.QGridLayout()
        layout_grid.addWidget(ksf_h_lbl, 0, 0)
        layout_grid.addWidget(ksf_h_edt, 0, 1)
        layout_grid.addWidget(ksf_w_lbl, 1, 0)
        layout_grid.addWidget(ksf_w_edt, 1, 1)
        layout_grid.addWidget(gaus_kern_lbl, 2, 0)
        layout_grid.addWidget(gaus_kern_edt, 2, 1)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addLayout(layout_grid)
        layout.addStretch()
        layout.addLayout(btns_box)

        self.setLayout(layout)
        self.setGeometry(300, 300, 60, 60)
        self.setWindowTitle("Settings Tool")

    def ksf_h_var(self, text):
        """Set keys size factor value (hight)."""
        self.ksf_h  =  np.float64(text)

    def ksf_w_var(self, text):
        """Set keys size factor value (width)."""
        self.ksf_w  =  np.float64(text)

    def g_kern_var(self, text):
        """Set the kernel size of gaussian filter."""
        self.g_kern  =  np.float64(text)

    def save_vars(self):
        """Save new settings."""
        np.save('keys_size_factor.npy', [self.ksf_h, self.ksf_w])
        np.save('gauss_kern_size.npy', self.g_kern)

    def close_(self):
        """Close the widget."""
        self.close()

    @QtCore.pyqtSlot()
    def restart(self):
        """Send message to main GUI."""
        self.procStart.emit()


class TestSpotsDetectionSetting(QtWidgets.QWidget):
    """Popup tool to study spots detection."""
    procStart  =  QtCore.pyqtSignal()

    def __init__(self, spts_mip, spts_4d, time_step_value):
        QtWidgets.QWidget.__init__(self)

        ksf_h  =  np.load('keys_size_factor.npy')[0]
        ksf_w  =  np.load('keys_size_factor.npy')[1]

        frame1  =  pg.ImageView(self, name='Frame1')
        # frame1.setImage(raw_mip_data)
        frame1.setImage(spts_mip)
        frame1.ui.roiBtn.hide()
        frame1.ui.menuBtn.hide()
        frame1.timeLine.sigPositionChanged.connect(self.update_time_and_frame)

        frame2  =  pg.ImageView(self, name='Frame2')
        frame2.ui.roiBtn.hide()
        frame2.ui.menuBtn.hide()
        frame2.view.setXLink('Frame1')
        frame2.view.setYLink('Frame1')

        frame3  =  pg.ImageView(self, name='Frame3')
        frame3.ui.roiBtn.hide()
        frame3.ui.menuBtn.hide()
        frame3.timeLine.sigPositionChanged.connect(self.update_frame4)
        frame3.view.setXLink('Frame1')
        frame3.view.setYLink('Frame1')

        frame4  =  pg.ImageView(self, name='Frame4')
        frame4.ui.roiBtn.hide()
        frame4.ui.menuBtn.hide()
        frame4.timeLine.sigPositionChanged.connect(self.update_frame3)
        frame4.view.setXLink('Frame1')
        frame4.view.setYLink('Frame1')

        choose_frame_tggl  =  QtWidgets.QCheckBox("Set Frame", self)
        choose_frame_tggl.stateChanged.connect(self.choose_frame)
        choose_frame_tggl.setToolTip('Choose this frame to work on')
        choose_frame_tggl.setFixedSize(int(ksf_h * 120), int(ksf_w * 25))

        merge_radius_lbl  =  QtWidgets.QLabel("Mer Rad", self)
        merge_radius_lbl.setFixedSize(int(ksf_h * 70), int(ksf_w * 25))

        merge_radius_edt  =  QtWidgets.QLineEdit(self)
        merge_radius_edt.textChanged[str].connect(self.merge_radius_var)
        merge_radius_edt.setToolTip("Set the maximmum distance between 2 spots to merge them; suggested value 5")
        merge_radius_edt.setFixedSize(int(ksf_h * 35), int(ksf_w * 25))

        merge_radius_edt_lbl_box  =  QtWidgets.QHBoxLayout()
        merge_radius_edt_lbl_box.addWidget(merge_radius_lbl)
        merge_radius_edt_lbl_box.addWidget(merge_radius_edt)

        spots_thr_lbl  =  QtWidgets.QLabel('Spots Thr', self)
        spots_thr_lbl.setFixedSize(int(ksf_h * 70), int(ksf_w * 25))

        spots_thr_edt  =  QtWidgets.QLineEdit(self)
        spots_thr_edt.textChanged[str].connect(self.spots_thr_var)
        spots_thr_edt.setToolTip('Intensity threshold to segment spots: it is expressed in terms of standard deviation, suggested value 7')
        spots_thr_edt.setFixedSize(int(ksf_h * 35), int(ksf_w * 25))

        volume_thr_lbl  =  QtWidgets.QLabel('Vol Thr', self)
        volume_thr_lbl.setFixedSize(int(ksf_h * 70), int(ksf_w * 25))

        volume_thr_edt  =  QtWidgets.QLineEdit(self)
        volume_thr_edt.textChanged[str].connect(self.volume_thr_var)
        volume_thr_edt.setToolTip('Threshold volume on spot detection: suggested value 5')
        volume_thr_edt.setFixedSize(int(ksf_h * 35), int(ksf_w * 25))

        spots_det_btn  =  QtWidgets.QPushButton("Detect Spots", self)
        spots_det_btn.clicked.connect(self.spots_det)
        spots_det_btn.setToolTip('Detect spots in the choosen frame')
        spots_det_btn.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        close_insert_btn  =  QtWidgets.QPushButton("Close && Insert", self)
        close_insert_btn.clicked.connect(self.close_insert)
        close_insert_btn.setToolTip('Close test tool and insert the values')
        close_insert_btn.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        frame1_box  =  QtWidgets.QHBoxLayout()
        frame1_box.addWidget(frame1)

        frame2_box  =  QtWidgets.QHBoxLayout()
        frame2_box.addWidget(frame2)

        frame3_box  =  QtWidgets.QHBoxLayout()
        frame3_box.addWidget(frame3)

        frame4_box  =  QtWidgets.QHBoxLayout()
        frame4_box.addWidget(frame4)

        tabs_a  =  QtWidgets.QTabWidget()
        tab1_a  =  QtWidgets.QWidget()
        tab2_a  =  QtWidgets.QWidget()

        tab1_a.setLayout(frame1_box)
        tab2_a.setLayout(frame2_box)

        tabs_a.addTab(tab1_a, "Raw Spots")
        tabs_a.addTab(tab2_a, "Detected")

        tabs_b  =  QtWidgets.QTabWidget()
        tab1_b  =  QtWidgets.QWidget()
        tab2_b  =  QtWidgets.QWidget()

        tab1_b.setLayout(frame3_box)
        tab2_b.setLayout(frame4_box)

        tabs_b.addTab(tab1_b, "Raw Spots")
        tabs_b.addTab(tab2_b, "Detected")

        spots_thr_hor  =  QtWidgets.QHBoxLayout()
        spots_thr_hor.addWidget(spots_thr_lbl)
        spots_thr_hor.addWidget(spots_thr_edt)

        volume_thr_hor  =  QtWidgets.QHBoxLayout()
        volume_thr_hor.addWidget(volume_thr_lbl)
        volume_thr_hor.addWidget(volume_thr_edt)

        tstep_lbl  =  QtWidgets.QLabel("time  0", self)
        tstep_lbl.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        t_frame_lbl  =  QtWidgets.QLabel("t frame " + '0', self)
        t_frame_lbl.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        z_frame_lbl  =  QtWidgets.QLabel("z frame " + '0', self)
        z_frame_lbl.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        ready_busy_lbl  =  QtWidgets.QLabel(self)
        ready_busy_lbl.setFixedSize(int(ksf_h * 45), int(ksf_h * 25))
        ready_busy_lbl.setToolTip("When red the pc is running, when green it is ready")
        ready_busy_lbl.setStyleSheet("background-color: green")

        commands  =  QtWidgets.QVBoxLayout()
        commands.addWidget(choose_frame_tggl)
        commands.addLayout(merge_radius_edt_lbl_box)
        commands.addLayout(spots_thr_hor)
        commands.addLayout(volume_thr_hor)
        commands.addWidget(spots_det_btn)
        commands.addStretch()
        commands.addWidget(close_insert_btn)
        commands.addStretch()
        commands.addWidget(tstep_lbl)
        commands.addWidget(t_frame_lbl)
        commands.addWidget(z_frame_lbl)
        commands.addWidget(ready_busy_lbl)

        layout  =  QtWidgets.QHBoxLayout()
        layout.addWidget(tabs_a)
        layout.addWidget(tabs_b)
        layout.addLayout(commands)

        self.frame1           =  frame1
        self.frame2           =  frame2
        self.frame3           =  frame3
        self.frame4           =  frame4
        self.z_frame_lbl      =  z_frame_lbl
        self.t_frame_lbl      =  t_frame_lbl
        self.tstep_lbl        =  tstep_lbl
        self.time_step_value  =  time_step_value
        self.spts_4d          =  spts_4d
        self.spts_mip         =  spts_mip
        self.ready_busy_lbl   =  ready_busy_lbl

        self.setLayout(layout)
        self.setGeometry(300, 300, 1000, 500)
        self.setWindowTitle("Test Spot Detection")

    def merge_radius_var(self, text):
        """Set the spots merge radius."""
        self.merge_radius_value  =  int(text)

    def spots_thr_var(self, text):
        """Set the spots threshold value."""
        self.spots_thr_value  =  float(text)

    def volume_thr_var(self, text):
        """Set spots volume threshold value."""
        self.volume_thr_value  =  int(text)

    def update_frame4(self):
        """Update frame 3 from changes in frame 4."""
        self.frame4.setCurrentIndex(self.frame3.currentIndex)

    def update_time_and_frame(self):
        """Update time and frame labels from frame1."""
        self.tstep_lbl.setText("time  " + time.strftime("%M:%S", time.gmtime(self.frame1.currentIndex * self.time_step_value)))
        self.t_frame_lbl.setText("t frame  "  +  str(self.frame1.currentIndex))

    def update_frame3(self):
        """Update frame 3 from changes in frame 4."""
        self.frame3.setCurrentIndex(self.frame4.currentIndex)
        self.z_frame_lbl.setText("z frame  " + str(self.frame4.currentIndex))

    def choose_frame(self, state):
        """Choose the frame to work on."""
        if state == QtCore.Qt.Checked:
            self.cif  =  self.frame1.currentIndex
            self.frame1.setImage(self.spts_mip[self.frame1.currentIndex])

        else:
            self.frame1.setImage(self.spts_mip)
            self.frame1.setCurrentIndex(self.cif)

    def spots_det(self):
        """Detect spots."""
        reload(SpotsDetection3D)
        self.ready_busy_lbl.setStyleSheet("background-color: red")
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

        try:
            spots  =  SpotsDetection3D.SpotsDetection3D_Single4Test([self.spts_4d[self.cif, :, :, :], self.spots_thr_value, self.volume_thr_value, self.merge_radius_value])
            # self.frame2.setImage(np.sign(spots.spots_ints))
            self.frame2.setImage(spots.spots_clean)
            pg.image(spots.spots_clean, title='clean')
            self.frame3.setImage(self.spts_4d[self.cif])
            self.frame4.setImage(spots.spots_lbls)
        except Exception:
            traceback.print_exc()

        self.ready_busy_lbl.setStyleSheet("background-color: green")

    @QtCore.pyqtSlot()
    def close_insert(self):
        """Send results."""
        self.procStart.emit()


class ChopStack(QtWidgets.QWidget):
    """Popup tool to remove frames from the raw data stack."""
    procStart  =  QtCore.pyqtSignal()

    def __init__(self, raw_data, time_step_value):
        QtWidgets.QWidget.__init__(self)

        ksf_h  =  np.load('keys_size_factor.npy')[0]
        ksf_w  =  np.load('keys_size_factor.npy')[1]

        frame_raw  =  pg.ImageView(self, name='FrameSpts1')
        frame_raw.ui.roiBtn.hide()
        frame_raw.ui.menuBtn.hide()
        frame_raw.setImage(raw_data)
        frame_raw.timeLine.sigPositionChanged.connect(self.update_frame_raw_data)

        current_frame_lbl  =  QtWidgets.QLabel("Frame: 0", self)
        current_frame_lbl.setFixedSize(int(ksf_h * 100), int(ksf_w * 25))

        time_lbl  =  QtWidgets.QLabel("T: 0", self)
        time_lbl.setFixedSize(int(ksf_h * 70), int(ksf_w * 25))

        set_first_frame_btn  =  QtWidgets.QPushButton("Set First 0", self)
        set_first_frame_btn.clicked.connect(self.set_first_frame)
        set_first_frame_btn.setToolTip('Set current frame as first analysis frame')
        set_first_frame_btn.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        set_last_frame_btn  =  QtWidgets.QPushButton("Set Last " + str(raw_data.shape[0]), self)
        set_last_frame_btn.clicked.connect(self.set_last_frame)
        set_last_frame_btn.setToolTip('Set current frame as last analysis frame')
        set_last_frame_btn.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        insert_close_btn  =  QtWidgets.QPushButton("Insert", self)
        insert_close_btn.clicked.connect(self.insert_close)
        insert_close_btn.setToolTip("Insert changes and close")
        insert_close_btn.setFixedSize(int(ksf_h * 100), int(ksf_w * 25))

        bottom_box  =  QtWidgets.QHBoxLayout()
        bottom_box.addWidget(set_first_frame_btn)
        bottom_box.addWidget(set_last_frame_btn)
        bottom_box.addWidget(insert_close_btn)
        bottom_box.addStretch()
        bottom_box.addWidget(current_frame_lbl)
        bottom_box.addWidget(time_lbl)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addWidget(frame_raw)
        layout.addLayout(bottom_box)

        self.frame_raw            =  frame_raw
        self.current_frame_lbl    =  current_frame_lbl
        self.time_lbl             =  time_lbl
        self.time_step_value      =  time_step_value
        self.set_first_frame_btn  =  set_first_frame_btn
        self.set_last_frame_btn   =  set_last_frame_btn
        self.first_last_frame     =  [0, raw_data.shape[0]]

        self.setLayout(layout)
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle("ChopStack Tool")

    def update_frame_raw_data(self):
        """Update frame number and time of the widget."""
        self.current_frame_lbl.setText("Frame: " + str(self.frame_raw.currentIndex))
        self.time_lbl.setText("T: " + time.strftime("%M:%S", time.gmtime(self.frame_raw.currentIndex * self.time_step_value)))

    def set_first_frame(self):
        """Set first frame choise."""
        self.first_last_frame[0]  =  self.frame_raw.currentIndex
        self.set_first_frame_btn.setText("First Frame " + str(self.first_last_frame[0]))

    def set_last_frame(self):
        """Set last frame choise."""
        self.first_last_frame[1]  =  self.frame_raw.currentIndex
        self.set_last_frame_btn.setText("Last Frame " + str(self.first_last_frame[1]))

    @QtCore.pyqtSlot()
    def insert_close(self):
        """Send message to main GUI."""
        self.procStart.emit()


class CropStack(QtWidgets.QWidget):
    """Popup tool to remove frames from the raw data stack."""
    procStart  =  QtCore.pyqtSignal()

    def __init__(self, raw_data, time_step_value):
        QtWidgets.QWidget.__init__(self)

        ksf_h  =  np.load('keys_size_factor.npy')[0]
        ksf_w  =  np.load('keys_size_factor.npy')[1]

        crop_roi_raw  =  pg.RectROI([80, 80], [80, 80], pen='r')

        frame_raw  =  pg.ImageView(self)
        frame_raw.ui.roiBtn.hide()
        frame_raw.ui.menuBtn.hide()
        frame_raw.setImage(raw_data)
        frame_raw.addItem(crop_roi_raw)
        frame_raw.timeLine.sigPositionChanged.connect(self.update_frame_raw_data)

        current_frame_lbl  =  QtWidgets.QLabel("Frame: 0", self)
        current_frame_lbl.setFixedSize(int(ksf_h * 100), int(ksf_w * 25))

        time_lbl  =  QtWidgets.QLabel("T: 0", self)
        time_lbl.setFixedSize(int(ksf_h * 70), int(ksf_w * 25))

        insert_close_btn  =  QtWidgets.QPushButton("Insert", self)
        insert_close_btn.clicked.connect(self.insert_close)
        insert_close_btn.setToolTip("Insert changes and close")
        insert_close_btn.setFixedSize(int(ksf_h * 100), int(ksf_w * 25))

        bottom_box  =  QtWidgets.QHBoxLayout()
        bottom_box.addWidget(insert_close_btn)
        bottom_box.addStretch()
        bottom_box.addWidget(current_frame_lbl)
        bottom_box.addWidget(time_lbl)

        layout  =  QtWidgets.QVBoxLayout()
        layout.addWidget(frame_raw)
        layout.addLayout(bottom_box)

        self.frame_raw            =  frame_raw
        self.current_frame_lbl    =  current_frame_lbl
        self.time_lbl             =  time_lbl
        self.time_step_value      =  time_step_value
        self.crop_roi_raw         =  crop_roi_raw

        self.setLayout(layout)
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle("CropStack Tool")

    def update_frame_raw_data(self):
        """Update frame number and time of the widget."""
        self.current_frame_lbl.setText("Frame: " + str(self.frame_raw.currentIndex))
        self.time_lbl.setText("T: " + time.strftime("%M:%S", time.gmtime(self.frame_raw.currentIndex * self.time_step_value)))

    @QtCore.pyqtSlot()
    def insert_close(self):
        """Send message to main GUI."""
        self.procStart.emit()


class SpatialAnalisys(QtWidgets.QWidget):
    """Popup tool to study spots detection."""
    def __init__(self, raw_data, analysis_folder):
        QtWidgets.QWidget.__init__(self)

        ksf_h  =  np.load('keys_size_factor.npy')[0]
        ksf_w  =  np.load('keys_size_factor.npy')[1]

        frame1  =  pg.ImageView(self, name='Frame1')
        frame1.setImage(raw_data.raw_data_mip)
        frame1.ui.roiBtn.hide()
        frame1.ui.menuBtn.hide()
        frame1.timeLine.sigPositionChanged.connect(self.update_from_frame1)

        foldername_lbl  =  QtWidgets.QLabel(analysis_folder, self)

        first_analyzed_lbl  =  QtWidgets.QLabel(self)
        last_analyzed_lbl   =  QtWidgets.QLabel(self)
        first_frame         =  np.load(analysis_folder + '/first_mip_frame.npy')
        last_frame          =  np.load(analysis_folder + '/last_mip_frame.npy')

        uu_first  =  np.sum(raw_data.raw_data_mip - first_frame, axis=(1, 2))
        try:
            first_fr  =  np.where(uu_first == 0)[0][0]
            first_analyzed_lbl.setText("First Analyzed Frm: " + str(first_fr))
        except IndexError:
            first_analyzed_lbl.setText("First Analyzed Frm out")

        uu_last  =  np.sum(raw_data.raw_data_mip - last_frame, axis=(1, 2))
        try:
            last_fr  =  np.where(uu_last == 0)[0][0]
            last_analyzed_lbl.setText("Last Analyzed Frm: " + str(last_fr))
        except IndexError:
            last_analyzed_lbl.setText("Last Analyzed Frm out")

        foldname_frame_box  =  QtWidgets.QVBoxLayout()
        foldname_frame_box.addWidget(foldername_lbl)
        foldname_frame_box.addWidget(frame1)

        time_lbl  =  QtWidgets.QLabel("time 00:00", self)
        time_lbl.setFixedSize(int(ksf_h * 100), int(ksf_w * 25))

        frame_lbl  =  QtWidgets.QLabel("frame 0", self)
        frame_lbl.setFixedSize(int(ksf_h * 100), int(ksf_w * 25))

        title_lbl  =  QtWidgets.QLabel("Roi Coords (µm)", self)
        title_lbl.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        roi_left_edt  =  QtWidgets.QLineEdit(self)
        roi_left_edt.textChanged[str].connect(self.roi_left_var)
        roi_left_edt.setToolTip("Sets/Read left corners x-coordinate")
        roi_left_edt.setFixedSize(int(ksf_h * 70), int(ksf_w * 25))
        roi_left_edt.returnPressed.connect(self.roi_left_update)

        left_lbl  =  QtWidgets.QLabel("left ", self)
        left_lbl.setFixedSize(int(ksf_h * 100), int(ksf_w * 25))

        left_box  =  QtWidgets.QVBoxLayout()
        left_box.addWidget(left_lbl)
        left_box.addWidget(roi_left_edt)

        roi_right_edt  =  QtWidgets.QLineEdit(self)
        roi_right_edt.textChanged[str].connect(self.roi_right_var)
        roi_right_edt.setToolTip("Sets/Read right corners x-coordinate")
        roi_right_edt.setFixedSize(int(ksf_h * 70), int(ksf_w * 25))
        roi_right_edt.returnPressed.connect(self.roi_right_update)

        right_lbl  =  QtWidgets.QLabel("right ", self)
        right_lbl.setFixedSize(int(ksf_h * 100), int(ksf_w * 25))

        right_box  =  QtWidgets.QVBoxLayout()
        right_box.addWidget(right_lbl)
        right_box.addWidget(roi_right_edt)

        roi_bottom_edt  =  QtWidgets.QLineEdit(self)
        roi_bottom_edt.textChanged[str].connect(self.roi_bottom_var)
        roi_bottom_edt.setToolTip("Sets/Read bottom corner y-coordinate")
        roi_bottom_edt.setFixedSize(int(ksf_h * 70), int(ksf_w * 25))
        roi_bottom_edt.returnPressed.connect(self.roi_bottom_update)

        bottom_lbl  =  QtWidgets.QLabel("bottom ", self)
        bottom_lbl.setFixedSize(int(ksf_h * 100), int(ksf_w * 25))

        bottom_box  =  QtWidgets.QVBoxLayout()
        bottom_box.addWidget(bottom_lbl)
        bottom_box.addWidget(roi_bottom_edt)

        roi_top_edt  =  QtWidgets.QLineEdit(self)
        roi_top_edt.textChanged[str].connect(self.roi_top_var)
        roi_top_edt.setToolTip("Sets/Read top corners y-coordinate")
        roi_top_edt.setFixedSize(int(ksf_h * 70), int(ksf_w * 25))
        roi_top_edt.returnPressed.connect(self.roi_top_update)

        top_lbl  =  QtWidgets.QLabel("top ", self)
        top_lbl.setFixedSize(int(ksf_h * 100), int(ksf_w * 25))

        top_box  =  QtWidgets.QVBoxLayout()
        top_box.addWidget(top_lbl)
        top_box.addWidget(roi_top_edt)

        roi_coords_box  =  QtWidgets.QGridLayout()
        roi_coords_box.addLayout(left_box, 0, 0)
        roi_coords_box.addLayout(right_box, 0, 1)
        roi_coords_box.addLayout(bottom_box, 1, 0)
        roi_coords_box.addLayout(top_box, 1, 1)

        save_region_btn  =  QtWidgets.QPushButton("Save", self)
        save_region_btn.clicked.connect(self.save_region)
        save_region_btn.setToolTip("Save Excel file with only the spots present in the region of interest")
        save_region_btn.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        commands  =  QtWidgets.QVBoxLayout()
        commands.addWidget(title_lbl)
        commands.addLayout(roi_coords_box)
        commands.addStretch()
        commands.addWidget(first_analyzed_lbl)
        commands.addWidget(last_analyzed_lbl)
        commands.addStretch()
        commands.addWidget(save_region_btn)
        commands.addWidget(time_lbl)
        commands.addWidget(frame_lbl)

        layout  =  QtWidgets.QHBoxLayout()
        layout.addLayout(foldname_frame_box)
        layout.addLayout(commands)

        crop_roi  =  pg.RectROI([80, 80], [80, 80], pen='r')
        crop_roi.sigRegionChanged.connect(self.roi_update_coords)

        frame1.addItem(crop_roi)

        self.frame1     =  frame1
        self.frame_lbl  =  frame_lbl
        self.time_lbl   =  time_lbl
        self.raw_data   =  raw_data
        self.crop_roi   =  crop_roi

        self.roi_left_edt     =  roi_left_edt
        self.roi_right_edt    =  roi_right_edt
        self.roi_bottom_edt   =  roi_bottom_edt
        self.roi_top_edt      =  roi_top_edt
        self.analysis_folder  =  analysis_folder

        self.setLayout(layout)
        self.setGeometry(300, 300, 1000, 500)
        self.setWindowTitle("Test Spot Detection")

    def update_from_frame1(self):
        """Update labels from frame current index."""
        self.frame_lbl.setText("frame " + str(self.frame1.currentIndex))
        self.time_lbl.setText("time " + time.strftime("%M:%S", time.gmtime(self.frame1.currentIndex * self.raw_data.time_step_value)))

    def roi_update_coords(self):
        """Update coordinate value of the roi corners coordinate."""
        self.roi_left_edt.setText(str(np.round(self.crop_roi.getState()["pos"][0] * self.raw_data.pix_size_xy, 2)))
        self.roi_right_edt.setText(str(np.round((self.crop_roi.getState()["pos"][0] + self.crop_roi.getState()["size"][0]) * self.raw_data.pix_size_xy, 2)))
        self.roi_top_edt.setText(str(np.round(self.crop_roi.getState()["pos"][1] * self.raw_data.pix_size_xy, 2)))
        self.roi_bottom_edt.setText(str(np.round((self.crop_roi.getState()["pos"][1] + self.crop_roi.getState()["size"][1]) * self.raw_data.pix_size_xy, 2)))

    def roi_top_var(self, text):
        """Set roi top y-coordinates value."""
        self.roi_top_value  =  float(text)

    def roi_bottom_var(self, text):
        """Set roi bottom y-coordinates value."""
        self.roi_bottom_value  =  float(text)

    def roi_right_var(self, text):
        """Set roi right x-coordinates value."""
        self.roi_right_value  =  float(text)

    def roi_left_var(self, text):
        """Set roi left x-coordinates value."""
        self.roi_left_value  =  float(text)

    def roi_right_update(self):
        """Set the right corners x-coordinate."""
        self.crop_roi.sigRegionChanged.disconnect(self.roi_update_coords)
        self.crop_roi.setSize([(self.roi_right_value / self.raw_data.pix_size_xy - self.crop_roi.getState()["pos"][0]), self.crop_roi.getState()["size"][1]])
        self.crop_roi.sigRegionChanged.connect(self.roi_update_coords)

    def roi_left_update(self):
        """Set the right corners x-coordinate."""
        self.crop_roi.sigRegionChanged.disconnect(self.roi_update_coords)
        self.crop_roi.setPos([self.roi_left_value / self.raw_data.pix_size_xy, self.crop_roi.getState()["pos"][1]])
        self.crop_roi.sigRegionChanged.connect(self.roi_update_coords)
        self.roi_right_update()

    def roi_top_update(self):
        """Set the top corners y-coordinate."""
        self.crop_roi.sigRegionChanged.disconnect(self.roi_update_coords)
        self.crop_roi.setPos([self.crop_roi.getState()["pos"][0], self.roi_top_value / self.raw_data.pix_size_xy])
        self.crop_roi.sigRegionChanged.connect(self.roi_update_coords)
        self.roi_bottom_update()

    def roi_bottom_update(self):
        """Set the bottom corners y-coordinate."""
        self.crop_roi.sigRegionChanged.disconnect(self.roi_update_coords)
        self.crop_roi.setSize([self.crop_roi.getState()["size"][0], self.roi_bottom_value / self.raw_data.pix_size_xy - self.crop_roi.getState()["pos"][1]])
        self.crop_roi.sigRegionChanged.connect(self.roi_update_coords)

    def save_region(self):
        """Save the spatially organized Excell file."""
        reload(SpatiallySelectedSaver)
        xlsx_filename  =  str(QtWidgets.QFileDialog.getSaveFileName(None, "Define the excell file to write spatially selected traces", filter="*.xlsx")[0])
        left           =  min(self.roi_left_value / self.raw_data.pix_size_xy, self.roi_right_value / self.raw_data.pix_size_xy)
        right          =  max(self.roi_left_value / self.raw_data.pix_size_xy, self.roi_right_value / self.raw_data.pix_size_xy)
        top            =  min(self.roi_top_value / self.raw_data.pix_size_xy, self.roi_bottom_value / self.raw_data.pix_size_xy)
        bottom         =  max(self.roi_top_value / self.raw_data.pix_size_xy, self.roi_bottom_value / self.raw_data.pix_size_xy)
        SpatiallySelectedSaver.SpatiallySelectedSaver(xlsx_filename, self.analysis_folder, left, right, top, bottom, self.raw_data.pix_size_xy)


class GenerateImage(QtWidgets.QWidget):
    """Popup tool to choose the fame to popup with squares around the spots."""

    def __init__(self, raw_data_mip, analysis_folder):
        QtWidgets.QWidget.__init__(self)

        mycmap      =  np.fromfile("mycmap.bin", "uint16").reshape((10000, 3))      # / 255.0
        colors4map  =  []
        for k in range(mycmap.shape[0]):
            colors4map.append(mycmap[k, :])
        colors4map     =  colors4map + colors4map + colors4map + colors4map + colors4map + colors4map
        colors4map[0]  =  np.array([0, 0, 0])

        spots_tr  =  SaveReadMatrix.SpotsMatrixReader(analysis_folder, '/spots_trck.npy').spts_lbls

        ksf_h  =  np.load('keys_size_factor.npy')[0]
        ksf_w  =  np.load('keys_size_factor.npy')[1]

        frame_raw  =  pg.ImageView(self, name='FrameRaw')
        frame_raw.ui.roiBtn.hide()
        frame_raw.ui.menuBtn.hide()
        frame_raw.setImage(raw_data_mip)
        frame_raw.timeLine.sigPositionChanged.connect(self.update_from_raw_data)

        frame_segmspots  =  pg.ImageView(self, name='FrameSegm')
        frame_segmspots.ui.roiBtn.hide()
        frame_segmspots.ui.menuBtn.hide()
        frame_segmspots.setImage(spots_tr)
        frame_segmspots.view.setXLink("FrameRaw")
        frame_segmspots.view.setYLink("FrameRaw")
        frame_segmspots.timeLine.sigPositionChanged.connect(self.update_from_segmspots)
        rnd_cmap       =  pg.ColorMap(np.linspace(0, 1, spots_tr.max()), color=colors4map)
        frame_segmspots.setColorMap(rnd_cmap)

        tabs_tot  =  QtWidgets.QTabWidget()
        tab_raw   =  QtWidgets.QWidget()
        tab_segm  =  QtWidgets.QWidget()

        frame_raw_box  =  QtWidgets.QHBoxLayout()
        frame_raw_box.addWidget(frame_raw)

        frame_segm_box  =  QtWidgets.QHBoxLayout()
        frame_segm_box.addWidget(frame_segmspots)

        tab_raw.setLayout(frame_raw_box)
        tab_segm.setLayout(frame_segm_box)

        tabs_tot.addTab(tab_raw, "Raw")
        tabs_tot.addTab(tab_segm, "Segmented")

        current_frame_lbl  =  QtWidgets.QLabel("Frame: 0", self)
        current_frame_lbl.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        select_frame_btn  =  QtWidgets.QPushButton("Select", self)
        select_frame_btn.clicked.connect(self.select_frame)
        select_frame_btn.setToolTip("Generate the image on this frame")
        select_frame_btn.setFixedSize(int(ksf_h * 130), int(ksf_w * 25))

        keys_box  =  QtWidgets.QVBoxLayout()
        keys_box.addWidget(select_frame_btn)
        keys_box.addStretch()
        keys_box.addWidget(current_frame_lbl)

        layout  =  QtWidgets.QHBoxLayout()
        layout.addWidget(tabs_tot)
        layout.addLayout(keys_box)

        self.frame_raw          =  frame_raw
        self.frame_segmspots    =  frame_segmspots
        self.current_frame_lbl  =  current_frame_lbl
        self.analysis_folder    =  analysis_folder
        self.raw_data_mip       =  raw_data_mip

        self.setLayout(layout)
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle("Generate Image")

    def update_from_segmspots(self):
        """Update raw data frame  and frame number from segmspots."""
        self.frame_raw.setCurrentIndex(self.frame_segmspots.currentIndex)
        self.current_frame_lbl.setText("Frame: " + str(self.frame_segmspots.currentIndex))

    def update_from_raw_data(self):
        """Update segmented frame and frame number from raw data frame."""
        self.frame_segmspots.setCurrentIndex(self.frame_raw .currentIndex)
        self.current_frame_lbl.setText("Frame:" + str(self.frame_raw.currentIndex))

    def select_frame(self):
        """Pop up this frame with circled spots."""
        GenerateFigureCircledSpots.GenerateFigureCircledSpots(self.raw_data_mip, self.analysis_folder, self.frame_raw.currentIndex)


class PhotobleachingTool(QtWidgets.QWidget):
    """Popup tool to check photobleaching."""
    def __init__(self, phtblc, fnames, analysis_folder, soft_version):
        QtWidgets.QWidget.__init__(self)

        ksf_h  =  np.load('keys_size_factor.npy')[0]
        ksf_w  =  np.load('keys_size_factor.npy')[1]

        frame_signal_prof  =  pg.PlotWidget(title="Signal")
        frame_signal_prof.plot(phtblc.spts_ints_prof, pen="b", symbol="x")
        frame_signal_prof.plot(phtblc.yy_spts_prof, pen='r')
        frame_signal_prof.setBackground("w")
        frame_signal_prof.setLabel("bottom", "Time", "frames")
        frame_signal_prof.setLabel("left", "Intensity", "A.U.")

        frame_bckg_prof  =  pg.PlotWidget(title="Background")
        frame_bckg_prof.plot(phtblc.bckg_ints_prof, pen="k", symbol="s")
        frame_bckg_prof.plot(phtblc.yy_bckg_prof, pen='r')
        frame_bckg_prof.setBackground("w")
        frame_bckg_prof.setLabel("bottom", "Time", "frames")
        frame_bckg_prof.setLabel("left", "Intensity", 'A.U')

        frames_box  =  QtWidgets.QHBoxLayout()
        frames_box.addWidget(frame_signal_prof)
        frames_box.addWidget(frame_bckg_prof)

        alpha_spts_lbl  =  QtWidgets.QLabel('<font>&alpha;' + "_signal = " + str(np.round(phtblc.popt_spts[1], 5)))
        alpha_bckg_lbl  =  QtWidgets.QLabel('<font>&alpha;' + "_bckg  = " + str(np.round(phtblc.popt_bckg[1], 5)))

        save_results_btn  =  QtWidgets.QPushButton("Write", self)
        save_results_btn.clicked.connect(self.save_results)
        save_results_btn.setToolTip("Write analysis results")
        save_results_btn.setFixedSize(int(ksf_h * 110), int(ksf_w * 25))

        keys_box  =  QtWidgets.QVBoxLayout()
        keys_box.addWidget(save_results_btn)
        keys_box.addWidget(alpha_spts_lbl)
        keys_box.addWidget(alpha_bckg_lbl)
        keys_box.addStretch()

        layout  =  QtWidgets.QHBoxLayout()
        layout.addLayout(frames_box)
        layout.addLayout(keys_box)

        self.phtblc           =  phtblc
        self.fnames           =  fnames
        self.analysis_folder  =  analysis_folder
        self.soft_version     =  soft_version

        self.setLayout(layout)
        self.setGeometry(300, 300, 1000, 500)
        self.setWindowTitle("Photobleaching")

    def save_results(self):
        """Save the analysis results."""
        reload(PhotoBleachingEstimate)
        PhotoBleachingEstimate.SavePhotobleachingResults(self.phtblc, self.fnames, self.analysis_folder, self.soft_version)


def main():
    """Main function."""
    app         =  QtWidgets.QApplication(sys.argv)
    splash_pix  =  QtGui.QPixmap('Icons/MLL_Logo2.png')
    splash      =  QtWidgets.QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()
    app.processEvents()
    ex  =  MainWindow()
    splash.finish(ex)
    sys.exit(app.exec_())


def except_hook(cls, exception, traceback):
    """Function to track the errors."""
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':

    main()
    sys.excepthook = except_hook
