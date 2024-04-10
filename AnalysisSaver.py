"""This function save the analysis results.

Input are segmented nuclei and parameters.
"""

from importlib import reload
import datetime
import xlsxwriter
import numpy as np

import SaveReadMatrix


class AnalysisSaver:
    """Only class, does all the job."""
    def __init__(self, analysis_folder, fnames, raw_data, software_version, spots_3d_det, spots_trckd, merge_radius,
                 detect_thr, min_volthr_value, dist_thr, spots_features, crop_roi, ch_numb, pre_nopre_flag):

        reload(SaveReadMatrix)

        book    =  xlsxwriter.Workbook(analysis_folder + '/SpotsAnalysis.xlsx')
        sheet1  =  book.add_worksheet("Info")
        sheet2  =  book.add_worksheet("Volume")
        sheet3  =  book.add_worksheet("Intensity")
        sheet4  =  book.add_worksheet("Background")
        sheet5  =  book.add_worksheet("Ints by Bckg")
        sheet7  =  book.add_worksheet("Coords X-Y")
        sheet6  =  book.add_worksheet("Coords Z-X-Y")
        sheet8  =  book.add_worksheet("Activations")

        sheet1.write(0, 0, "Gauss Kernel")
        sheet1.write(0, 1, np.load('gauss_kern_size.npy'))
        sheet1.write(1, 0, "merge radius")
        sheet1.write(1, 1, merge_radius)
        sheet1.write(2, 0, "detect threshold")
        sheet1.write(2, 1, detect_thr)
        sheet1.write(3, 0, "min volume thr")
        sheet1.write(3, 1, min_volthr_value)
        sheet1.write(4, 0, "Spots dist Thr")
        sheet1.write(4, 1, dist_thr)
        sheet1.write(5, 0, "Min number of active frames")
        sheet1.write(5, 1, spots_features.min_act_frames)
        sheet1.write(6, 0, "pixels size x-y (µm)")
        sheet1.write(6, 1, raw_data.pix_size_xy)
        sheet1.write(7, 0, "pixels size z (µm)")
        sheet1.write(7, 1, raw_data.pix_size_z)
        sheet1.write(8, 0, "time step (s)")
        sheet1.write(8, 1, raw_data.time_step_value)
        sheet1.write(10, 0, "software version")
        sheet1.write(10, 1, software_version)
        sheet1.write(11, 0, "date")
        sheet1.write(11, 1, datetime.datetime.now().strftime("%d-%b-%Y"))
        sheet1.write(13, 0, "File names")
        for cnt, fname in enumerate(fnames):
            sheet1.write(14 + cnt, 0, fname[fname.rfind('/') + 1:])

        sheet2.write(0, 0, "Time")
        sheet3.write(0, 0, "Time")
        sheet4.write(0, 0, "Time")
        sheet5.write(0, 0, "Time")
        sheet6.write(0, 0, "Time")
        sheet7.write(0, 0, "Time")
        sheet8.write(0, 0, "Time")
        sheet8.write(0, 1, "# of Spots")
        for k in range(spots_features.spots_features.shape[0]):
            sheet2.write(k + 1, 0, k)
            sheet3.write(k + 1, 0, k)
            sheet4.write(k + 1, 0, k)
            sheet5.write(k + 1, 0, k)
            sheet6.write(k + 2, 0, k)
            sheet7.write(k + 2, 0, k)
            sheet8.write(k + 1, 0, k)

        if len(spots_trckd.orig_tags) == 0:
            for ccnt, spot_idx in enumerate(spots_features.spots_idxs):
                sheet2.write(0, 1 + ccnt, "Spts_" + str(spot_idx))
                sheet3.write(0, 1 + ccnt, "Spts_" + str(spot_idx))
                sheet4.write(0, 1 + ccnt, "Spts_" + str(spot_idx))
                sheet5.write(0, 1 + ccnt, "Spts_" + str(spot_idx))
                sheet6.write(0, 1 + ccnt * 3, "Spts_" + str(spot_idx))
                sheet6.write(1, 1 + ccnt * 3, "z-coord")
                sheet6.write(1, 2 + ccnt * 3, "x-coord")
                sheet6.write(1, 3 + ccnt * 3, "y-coord")
                sheet7.write(0, 1 + ccnt * 2, "Spts_" + str(spot_idx))
                sheet7.write(1, 1 + ccnt * 2, "x-coord")
                sheet7.write(1, 2 + ccnt * 2, "y-coord")

        elif len(spots_trckd.orig_tags) != 0:
            for ccnt, spot_idx in enumerate(spots_trckd.orig_tags):
                sheet2.write(0, 1 + ccnt, "Spts_" + str(spot_idx))
                sheet3.write(0, 1 + ccnt, "Spts_" + str(spot_idx))
                sheet4.write(0, 1 + ccnt, "Spts_" + str(spot_idx))
                sheet5.write(0, 1 + ccnt, "Spts_" + str(spot_idx))
                sheet6.write(0, 1 + ccnt * 3, "Spts_" + str(spot_idx))
                sheet6.write(1, 1 + ccnt * 3, "z-coord")
                sheet6.write(1, 2 + ccnt * 3, "x-coord")
                sheet6.write(1, 3 + ccnt * 3, "y-coord")
                sheet7.write(0, 1 + ccnt * 2, "Spts_" + str(spot_idx))
                sheet7.write(1, 1 + ccnt * 2, "x-coord")
                sheet7.write(1, 2 + ccnt * 2, "y-coord")

        for ii1 in range(spots_features.spots_features.shape[0]):
            for ii2 in range(spots_features.spots_features.shape[2]):
                sheet2.write(1 + ii1, 1 + ii2, spots_features.spots_features[ii1, 0, ii2])
                sheet3.write(1 + ii1, 1 + ii2, spots_features.spots_features[ii1, 1, ii2])
                sheet4.write(1 + ii1, 1 + ii2, spots_features.spots_features[ii1, 2, ii2])
                sheet5.write(1 + ii1, 1 + ii2, spots_features.spots_features[ii1, 3, ii2])
                sheet6.write(2 + ii1, 1 + ii2 * 3, spots_features.spots_features[ii1, 4, ii2])
                sheet6.write(2 + ii1, 2 + ii2 * 3, spots_features.spots_features[ii1, 5, ii2])
                sheet6.write(2 + ii1, 3 + ii2 * 3, spots_features.spots_features[ii1, 6, ii2])
                sheet7.write(2 + ii1, 1 + ii2 * 2, spots_features.spots_features[ii1, 5, ii2])
                sheet7.write(2 + ii1, 2 + ii2 * 2, spots_features.spots_features[ii1, 6, ii2])

        precense  =  np.sign(spots_features.spots_features[:, 0, :]).sum(1)
        for jj in range(spots_features.spots_features.shape[0]):
            sheet8.write(jj + 1, 1, precense[jj])

        book.close()
        np.save(analysis_folder + '/spots_features.npy', spots_features.spots_features)
        np.save(analysis_folder + '/spots_idxs.npy', spots_features.spots_idxs)
        np.save(analysis_folder + '/crop_roi.npy', crop_roi)
        np.save(analysis_folder + '/ch_numb.npy', ch_numb)
        np.save(analysis_folder + '/first_mip_frame.npy', raw_data.raw_data_mip[0])
        np.save(analysis_folder + '/last_mip_frame.npy', raw_data.raw_data_mip[-1])
        SaveReadMatrix.SpotsMatrixSaver(spots_3d_det.spots_ints, analysis_folder, '/spots_ints.npy')
        SaveReadMatrix.SpotsMatrixSaver(spots_3d_det.spots_vol, analysis_folder, '/spots_vol.npy')
        SaveReadMatrix.SpotsMatrixSaver(spots_3d_det.spots_2d_lbls, analysis_folder, '/spots_2d_lbls.npy')
        np.save(analysis_folder + '/spots_coords.npy', spots_3d_det.spots_coords)
        np.save(analysis_folder + '/spots_orig_tags.npy', spots_trckd.orig_tags)
        np.save(analysis_folder + '/pre_nopre_flag.npy', pre_nopre_flag)
        SaveReadMatrix.SpotsMatrixSaver(spots_trckd.spts_trck_fin, analysis_folder, '/spots_trck.npy')
