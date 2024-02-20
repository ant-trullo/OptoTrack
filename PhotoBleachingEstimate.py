"""This function extimate photobleaching.

Input is the fnames and the analysis folder.
"""

import numpy as np
from skimage.segmentation import expand_labels
from scipy.optimize import curve_fit
import xlsxwriter
import datetime

import AnalysisLoader
import ServiceWidgets


def exp_func(x, a, c, d):
    """exponential function for fitting."""
    return a * np.exp(-c * x) + d


class PhotoBleachingEstimate:
    """Only class, does all the job."""
    def __init__(self, analysis_folder, raw_data):

        spts_det  =  AnalysisLoader.SpotsDetected(analysis_folder)                                                      # load detected spots
        tlen      =  spts_det.spots_2d_lbls.shape[0]                                                                    # number of time steps

        spts_ints_prof  =  np.zeros(tlen)                                                                               # initialize the vector with the average signal intensity
        bckg_ints_prof  =  np.zeros(tlen)                                                                               # initialize the vector with the average background intensity

        pbar  =  ServiceWidgets.ProgressBar(total1=tlen)
        pbar.show()
        pbar.update_progressbar1(0)

        for tt in range(tlen):                                                                                          # for each time frame
            pbar.update_progressbar1(tt)
            sub_coords                                                       =  spts_det.spots_coords[spts_det.spots_coords[:, 0] == tt]     # extract the coordinate of the current time frame
            sing_fr3d                                                        =  np.zeros((spts_det.spots_coords[-1, 1], spts_det.spots_coords[-1, 2], spts_det.spots_coords[-1, 3]), dtype=np.uint16)   # initialize the 3D single time frame matrix
            sing_fr3d[sub_coords[:, 1], sub_coords[:, 2], sub_coords[:, 3]]  =  1                                       # build the 3D single frame
            sing_fr3d_exp                                                    =  expand_labels(sing_fr3d, distance=3)    # expand the spots to avoid to include diffraction patterns in the background estimation
            spts_ints_prof[tt]                                               =  np.sum(raw_data.raw_data[tt] * sing_fr3d) / sing_fr3d.sum()     # mask 3D single frame raw data with spots mask and divide by the total volume of the detected spots
            bckg_ints_prof[tt]                                               =  np.sum(raw_data.raw_data[tt] * (1 - sing_fr3d_exp)) / (1 - sing_fr3d_exp).sum()     # mask raw data with the 'negative' of the expanded spots and divided by the total volume of the 'negative'

        pbar.close()

        popt_spts, pcov_spts  =  curve_fit(exp_func, np.arange(tlen), spts_ints_prof, p0=(spts_ints_prof[0], 0.5, spts_ints_prof[-1]), maxfev=3000)     # exponential fitting of the signal average intensity trace
        popt_bckg, pcov_bckg  =  curve_fit(exp_func, np.arange(tlen), bckg_ints_prof, p0=(bckg_ints_prof[0], 0.5, bckg_ints_prof[-1]), maxfev=3000)     # exponential fitting of the background averabe intensity trace

        yy_spts_prof  =  exp_func(np.arange(tlen), * popt_spts)                                                         # generate the fitting curve for the signal
        yy_bckg_prof  =  exp_func(np.arange(tlen), * popt_bckg)                                                         # generate the fitting curve for the background

        self.spts_ints_prof  =  spts_ints_prof
        self.bckg_ints_prof  =  bckg_ints_prof
        self.popt_spts       =  popt_spts
        self.popt_bckg       =  popt_bckg
        self.yy_spts_prof    =  yy_spts_prof
        self.yy_bckg_prof    =  yy_bckg_prof


class SavePhotobleachingResults:
    """Write an Excel file with the results."""
    def __init__(self, phtblc, fnames, analysis_folder, soft_version):

        tlen    =  phtblc.spts_ints_prof.size
        book    =  xlsxwriter.Workbook(analysis_folder + '/PhotoBleachingStudy.xlsx')
        sheet2  =  book.add_worksheet("PhotoBleaching")
        sheet1  =  book.add_worksheet("Info")

        sheet1.write(0, 0, "Date")
        sheet1.write(0, 1, datetime.datetime.now().strftime("%d-%b-%Y"))

        sheet1.write(1, 0, "Software Version")
        sheet1.write(1, 1, soft_version)

        sheet1.write(3, 0, "fnames")
        for cnt, fname in enumerate(fnames):
            sheet1.write(3 + cnt, 0, fname)

        sheet2.write(0, 0, "Frame")
        sheet2.write(0, 1, "Signal")
        sheet2.write(0, 2, "SignalFitting")
        sheet2.write(0, 3, "Background")
        sheet2.write(0, 4, "BackgroundFitting")

        for tt in range(tlen):
            sheet2.write(tt + 1, 0, tt + 1)
            sheet2.write(tt + 1, 1, phtblc.spts_ints_prof[tt])
            sheet2.write(tt + 1, 2, phtblc.yy_spts_prof[tt])
            sheet2.write(tt + 1, 3, phtblc.bckg_ints_prof[tt])
            sheet2.write(tt + 1, 4, phtblc.yy_bckg_prof[tt])

        sheet2.write(0, 7, "alpha signal")
        sheet2.write(1, 7, phtblc.popt_spts[1])
        sheet2.write(3, 7, "alpha bckg")
        sheet2.write(4, 7, phtblc.popt_bckg[1])

        book.close()











