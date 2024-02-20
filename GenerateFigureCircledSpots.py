"""This function generates images with raw data (mip and filtered) and red circles around spots.

Input is raw data and analysis folder.
"""

import numpy as np
from skimage.measure import regionprops_table
import pyqtgraph as pg

import SaveReadMatrix


class GenerateFigureCircledSpots:
    """Only class, does all the job."""
    def __init__(self, raw_data_mip, analysis_folder, t_frame):

        xlen, ylen        =  raw_data_mip[0].shape                                                                      # x and y size of the matrix
        img2show          =  np.zeros((xlen, ylen, 3), dtype=raw_data_mip.dtype)                                # initialize the final 3 channel and single frame mip matrix to show
        img2show[..., 0]  =  raw_data_mip[t_frame]                                                                      # add raw data mip in every channells to have raw data in grey
        img2show[..., 1]  =  raw_data_mip[t_frame]
        img2show[..., 2]  =  raw_data_mip[t_frame]

        spts_trck  =  SaveReadMatrix.SpotsMatrixReader(analysis_folder, '/spots_trck.npy').spts_lbls[t_frame]   # load segmented spots (2D plus time) and select the time frame in question
        rgp        =  regionprops_table(spts_trck, properties=["label", "centroid"])                                    # regionprops of the 2D spots in the frame
        xlen      -=  1
        ylen      -=  1
        for cnt, kk in enumerate(rgp["centroid-0"]):
            msk   =  np.zeros_like(img2show[:, :, 0])                                                                   # initialize mask to define the boxes
            ctrs  =  [int(kk), int(rgp["centroid-1"][cnt])]                                                             # record the centroid coordinates

            msk[max(ctrs[0] - 5, 0), max(ctrs[1] - 5, 0):min(ctrs[1] + 5, ylen)]     =  1                               # define the mask with the borders of the boxex for the spots
            msk[min(ctrs[0] + 5, xlen), max(ctrs[1] - 5, 0):min(ctrs[1] + 5, ylen)]  =  1
            msk[max(ctrs[0] - 5, 0):min(ctrs[0] + 5, xlen), max(ctrs[1] - 5, 0)]     =  1
            msk[max(ctrs[0] - 5, 0):min(ctrs[0] + 5, xlen), min(ctrs[1] + 5, ylen)]  =  1

            img2show[..., 0]  *=  (1 - msk)                                                                             # add the boxes on the image to show
            img2show[..., 0]  +=  255 * msk
            img2show[..., 1]  *=  (1 - msk)
            img2show[..., 2]  *=  (1 - msk)

        pg.image(img2show)                                                                                              # pop it up