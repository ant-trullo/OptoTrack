"""This function generates convex hull images starting from the positions spots take during all the evolution.

Input argument is the analysis folder, output the image with nuclei, positions spots take during the evolution
and the contour of the convex hull image.
"""

import numpy as np
from skimage.measure import regionprops_table
from skimage.filters import difference_of_gaussians, threshold_otsu
from skimage.morphology import remove_small_objects, convex_hull_image, binary_erosion
import pyqtgraph as pg

import AnalysisLoader
import SaveReadMatrix


class GenerateHullImage:
    """Only class, does all the job."""
    def __init__(self, analysis_folder, fnames):

        raw_data   =  AnalysisLoader.RawData(analysis_folder, fnames)                                                   # load raw data
        spts_lbls  =  SaveReadMatrix.SpotsMatrixReader(analysis_folder, '/spots_2d_lbls.npy').spts_lbls         # load detetcted spots
        tlen       =  spts_lbls.shape[0]                                                                                # number of time frames
        spts_ctrs  =  np.zeros_like(spts_lbls[0])                                                                       # initialize spots-centroids matrix
        for tt in range(tlen):                                                                                          # for each time frame
            rgp_spts   =  regionprops_table(spts_lbls[tt], properties=["centroid", "label"])                            # regionprops of all the spots
            for cnt, kk in enumerate(rgp_spts["centroid-0"]):                                                           # for each spot in the time frame
                spts_ctrs[int(kk), int(rgp_spts["centroid-1"][cnt])]  =  rgp_spts["label"][cnt]                         # add a point in the centroid position

        spts_ctrs  =  remove_small_objects(spts_ctrs, 8)                                                     # remove small objects, which are fake detections

        chull        =  convex_hull_image(spts_ctrs) * 1                                                                # hull image
        brdr         =  np.zeros_like(chull)                                                                            # image borders can give probelms
        brdr[:, 0]   =  1                                                                                               # border of the image
        brdr[:, -1]  =  1
        brdr[0, :]   =  1
        brdr[-1, :]  =  1
        chull_brdr   =  chull + brdr                                                                                    # sum of the border plus the convex hull umage
        chull_brdr[chull_brdr == 2]  =  0                                                                               # remove the overlap
        chull_brdr  *=  chull                                                                                           # remove the image border not included in the hull image
        cntr         =  chull_brdr ^ binary_erosion(binary_erosion(binary_erosion(chull_brdr)))                         # erosion in order to keep only the border
        z_mean       =  int(raw_data.raw_data.shape[1] / 2)                                                             # intermediate z value
        raw2show     =  raw_data.raw_data[:, z_mean].sum(0)                                                             # sum all the intermediate z
        raw2show     =  difference_of_gaussians(raw2show, 1.0, 10.)                             # different of gaussian to clean the image for visualization purpouse

        # nucs_fin            =  (raw2show > threshold_otsu(raw2show)) * raw_data.raw_data[:, 15].sum(0)                  # threshold to get the mask and multiply times the raw data
        # nucs2show           =  np.zeros(nucs_fin.shape + (3,), dtype=nucs_fin.dtype)                                    # create the 3 channels image
        # nucs2show[:, :, 0]  =  raw_data.raw_data_mip.sum(0) * (1 - np.sign(spts_ctrs + cntr)) + np.sign(spts_ctrs) * raw_data.raw_data_mip.sum(0).max() # add raw data masked and the spots centroid traces
        # nucs2show[:, :, 1]  =  raw_data.raw_data_mip.sum(0) * (1 - np.sign(spts_ctrs + cntr))                           # add masked raw data
        # nucs2show[:, :, 2]  =  raw_data.raw_data_mip.sum(0) * (1 - np.sign(spts_ctrs + cntr)) + np.sign(cntr) * raw_data.raw_data_mip.sum(0).max()   # add masked raw data with the hull image border
        #
        # imv  =  pg.ImageView()
        # imv.setImage(nucs2show)
        # imv.setWindowTitle("Surface: " + str(np.sum(chull)) + "  erosion 3 blu")
        # imv.show()
        # print(StrangePatch)                                                                                              # strange patch, otherwise the plot closes

        nucs_fin            =  (raw2show > threshold_otsu(raw2show)) * raw_data.raw_data[:, 15].sum(0)                  # threshold to get the mask and multiply times the raw data
        nucs2show           =  np.zeros(nucs_fin.shape + (3,), dtype=nucs_fin.dtype)                                    # create the 3 channels image
        nucs2show[:, :, 0]  =  raw_data.raw_data_mip.sum(0) * (1 - np.sign(spts_ctrs + cntr)) + np.sign(spts_ctrs) * raw_data.raw_data_mip.sum(0).max() # add raw data masked and the spots centroid traces
        nucs2show[:, :, 1]  =  raw_data.raw_data_mip.sum(0) * (1 - np.sign(spts_ctrs + cntr)) + np.sign(cntr) * raw_data.raw_data_mip.sum(0).max()                           # add masked raw data
        nucs2show[:, :, 2]  =  raw_data.raw_data_mip.sum(0) * (1 - np.sign(spts_ctrs + cntr)) + np.sign(cntr) * raw_data.raw_data_mip.sum(0).max()   # add masked raw data with the hull image border

        imv  =  pg.ImageView()
        imv.setImage(nucs2show)
        imv.setWindowTitle("Surface: " + str(np.sum(chull)) + " erosion 3 cyan")
        imv.show()
        print(StrangePatch)                                                                                              # strange patch, otherwise the plot closes
