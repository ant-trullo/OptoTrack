"""This function loads an already done analysis.

Input is the folder path.
"""

import numpy as np
from openpyxl import load_workbook
from PyQt5 import QtWidgets

import LoadRawData
import SaveReadMatrix


class RawData:
    """Load raw data."""
    def __init__(self, analysis_folder, fnames, no_chop=False):

        first_fr  =  None
        last_fr   =  None
        raw_data  =  None

        err_msg         =  QtWidgets.QMessageBox()
        first_frame     =  np.load(analysis_folder + '/first_mip_frame.npy')                                               # load the first analyzed raw frame
        last_frame      =  np.load(analysis_folder + '/last_mip_frame.npy')                                                # load the last analyzed raw frame
        ch_numb         =  np.load(analysis_folder + '/ch_numb.npy')                                                       # load channels number
        crop_roi        =  np.load(analysis_folder + '/crop_roi.npy')                                                      # load crop corners coordinates
        pre_nopre_flag  =  np.load(analysis_folder + '/pre_nopre_flag.npy')                                             # load presmoothing / no-presmoothing flag
        if fnames[0][-4:] == ".czi":                                                                                    # load data
            if pre_nopre_flag:
                raw_data  =  LoadRawData.LoadRawDataCziPresmooth(fnames, ch_numb)
            elif not pre_nopre_flag:
                raw_data  =  LoadRawData.LoadRawDataCzi(fnames, ch_numb)
        elif fnames[0][-4:] == ".tif":
            raw_data  =  LoadRawData.LoadRawDataTiff(fnames, ch_numb)

        raw_data.raw_data_mip  =  raw_data.raw_data_mip[:, crop_roi[0]:crop_roi[2], crop_roi[1]:crop_roi[3]]            # crop mip matrix accodringly with crop info
        raw_data.raw_data      =  raw_data.raw_data[:, :, crop_roi[0]:crop_roi[2], crop_roi[1]:crop_roi[3]]             # crop raw matrix accodringly with crop info

        if not no_chop:                                                                                                 # choping in reload must be optional because raw data are reloaded for the spatial analysis in which I need to see all the evolution captured to set my spatial referencies
            uu_first  =  np.sum(raw_data.raw_data_mip - first_frame, axis=(1, 2))                                       # subtract 'first_frame' from all the images of the mip stack and then sum in x-y to find the first analyzed frame

            try:
                first_fr  =  np.where(uu_first == 0)[0][0]                                                              # if there is no zero in the series it means that probably the raw data does not match the analysis folder
            except IndexError:
                QtWidgets.QMessageBox.about(err_msg, "Error", "Raw data selected are not the same as the analysis.")

            uu_last  =  np.sum(raw_data.raw_data_mip - last_frame, axis=(1, 2))                                             # same as for the first
            try:
                last_fr  =  np.where(uu_last == 0)[0][0]
            except IndexError:
                QtWidgets.QMessageBox.about(err_msg, "Error", "Raw data selected are not the same as the analysis.")

            raw_data.raw_data_mip  =  raw_data.raw_data_mip[first_fr:last_fr + 1]
            raw_data.raw_data      =  raw_data.raw_data[first_fr:last_fr + 1]

        self.raw_data_mip     =  raw_data.raw_data_mip
        self.raw_data         =  raw_data.raw_data
        self.pix_size_z       =  raw_data.pix_size_z
        self.pix_size_xy      =  raw_data.pix_size_xy
        self.ch_numb          =  ch_numb
        self.time_step_value  =  raw_data.time_step_value


class AnalysisParameters:
    """Load analysis parameters."""
    def __init__(self, analysis_folder):

        wb                       =  load_workbook(analysis_folder + '/SpotsAnalysis.xlsx')
        sm_sheet                 =  wb[wb.sheetnames[0]]
        self.merge_radius_value  =  int(sm_sheet.cell(column=2, row=1).value)
        self.spts_thr_value      =  sm_sheet.cell(column=2, row=2).value
        self.min_volthr_value    =  int(sm_sheet.cell(column=2, row=3).value)
        self.dist_thr_value      =  int(sm_sheet.cell(column=2, row=4).value)


class SpotsTracked:
    """Load tracked spots."""
    def __init__(self, analysis_folder):

        self.spts_trck_fin  =  SaveReadMatrix.SpotsMatrixReader(analysis_folder, '/spots_trck.npy').spts_lbls
        self.orig_tags      =  np.load(analysis_folder + '/spots_orig_tags.npy')


class SpotsDetected:
    """Load detected spots."""
    def __init__(self, analysis_folder):

        self.spots_ints     =  SaveReadMatrix.SpotsMatrixReader(analysis_folder, '/spots_ints.npy').spts_lbls
        self.spots_vol      =  SaveReadMatrix.SpotsMatrixReader(analysis_folder, '/spots_vol.npy').spts_lbls
        self.spots_2d_lbls  =  SaveReadMatrix.SpotsMatrixReader(analysis_folder, '/spots_2d_lbls.npy').spts_lbls
        self.spots_coords   =  np.load(analysis_folder + '/spots_coords.npy')
