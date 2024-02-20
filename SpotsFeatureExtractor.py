"""This function estimates the background of detected spots.

Input are spots_3D_coords and 2D tracked spots.
"""

import multiprocessing
import numpy as np
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels

import ServiceWidgets


def reconstruc3d(spots_3d_coords_def, frame):
    """Reconstruct spots in 3D in a given frame."""
    zlen, xlen, ylen     =  spots_3d_coords_def[-1, 1:]                                 # last row of spots_3d_coords gives info on the raw data size
    spots_fr             =  np.zeros((zlen, xlen, ylen), dtype=np.int64)                # initialize output matrix
    spots_3d_coords_def  =  spots_3d_coords_def[:-1, :]                                 # remove the row with the size info (not a fundamental operation)
    spots_coord_fr       =  spots_3d_coords_def[spots_3d_coords_def[:, 0] == frame]     # select all the coordinate of the selected time frame
    for mm in spots_coord_fr:                                                           # build the 3d matrix
        spots_fr[mm[1], mm[2], mm[3]]  =  1
    return spots_fr


class SpotsFeatureExtractor:
    """Define spots cages and measure the background for each of them."""
    def __init__(self, spots_trckd, spots_3d_coords, raw_data):

        min_act_frames  =  ServiceWidgets.InputScalar.getFlag(["Act frames thr:", " ", "Enter", "Minimum number of non-zeros frames to keep the nuclear time trace", "Enter the value", "Active Frames"])
        # cpu_owe         =  multiprocessing.cpu_count()

        rgp_spts_trckd  =  regionprops_table(spots_trckd, properties=["label", "coords"])
        for kk in rgp_spts_trckd["coords"]:
            if np.unique(kk[:, 0]).size < min_act_frames:
                spots_trckd[kk[:, 0], kk[:, 1], kk[:, 2]]  =  0

        tlen            =  spots_trckd.shape[0]                         # number of time steps
        spots_idxs      =  np.unique(spots_trckd[spots_trckd != 0])     # collect tags in increasing order
        spots_features  =  np.zeros((tlen, 7, spots_idxs.size))         # initialize output matrix (for each spot at each time frame we have volume, intensity, background value, intensity divided by background and the 3D coordinate of the centroid)

        pbar  =  ServiceWidgets.ProgressBar(total1=tlen)
        pbar.update_progressbar1(0)
        pbar.show()

        for tt in range(tlen):
            pbar.update_progressbar1(tt)
            spots_fr   =  reconstruc3d(spots_3d_coords, tt)                                                                        # reconstruct spot in 3d (it is b&w)
            spots_fr  *=  spots_trckd[tt]                                                                                          # give proper label to the 3d spots
            cage_fr    =  expand_labels(spots_fr, 5) - expand_labels(spots_fr, 3)                                                  # expand in 3d each label 5 times and subtract the same label expoanded 3 times to have cages around
            rgp_cage   =  regionprops_table(cage_fr, raw_data[tt], properties=["label", "area", "intensity_image"])                # regionprops of the cages
            rgp_spots  =  regionprops_table(spots_fr, raw_data[tt], properties=["label", "area", "intensity_image", "centroid"])   # regionprops of the spots (of course cage and spots label are the same)
            for cnt, lb in enumerate(rgp_cage["label"]):                                                                           # store info: volume, intensity, av_bckgd, ints by bckgd, centroid
                lb_idx                         =  np.where(lb == spots_idxs)[0]
                spots_features[tt, :, lb_idx]  =  rgp_spots["area"][cnt], np.sum(rgp_spots["intensity_image"][cnt]), np.sum(rgp_cage["intensity_image"][cnt]) / rgp_cage["area"][cnt], np.sum(rgp_spots["intensity_image"][cnt]) / (np.sum(rgp_cage["intensity_image"][cnt]) / rgp_cage["area"][cnt]), rgp_spots["centroid-0"][cnt], rgp_spots["centroid-1"][cnt], rgp_spots["centroid-2"][cnt]

        pbar.close()

        self.spots_features  =  spots_features
        self.spots_idxs      =  spots_idxs
        self.min_act_frames  =  min_act_frames


class SpotsFeatureExtractor2:
    """Coordinate the multiprocessed action of the utility function tio calculate the background of each spot."""
    def __init__(self, spots_trckd, spots_3d_coords, raw_data):

        min_act_frames  =  ServiceWidgets.InputScalar.getFlag(["Act frames thr:", " ", "Enter", "Minimum number of non-zeros frames to keep the nuclear time trace", "Enter the value", "Active Frames"])

        rgp_spts_trckd  =  regionprops_table(spots_trckd, properties=["label", "coords"])
        for kk in rgp_spts_trckd["coords"]:
            if np.unique(kk[:, 0]).size < min_act_frames:
                spots_trckd[kk[:, 0], kk[:, 1], kk[:, 2]]  =  0

        tlen            =  spots_trckd.shape[0]                         # number of time steps
        spots_idxs      =  np.unique(spots_trckd[spots_trckd != 0])     # collect tags in increasing order

        cpu_own  =  multiprocessing.cpu_count()
        if tlen > 5 * cpu_own:
            t_chops     =  np.array_split(np.arange(tlen), cpu_own)
            args_input  =  []
            for mm in range(cpu_own):
                args_input.append([spots_trckd[t_chops[mm]], spots_idxs, spots_3d_coords, raw_data[t_chops[mm]], t_chops[mm]])

            pool     =  multiprocessing.Pool()
            results  =  pool.map(SpotsFeatureExtractorUtility, args_input)
            pool.close()

            self.spots_features  =  results[0].spots_features
            for k in range(1, len(results)):
                self.spots_features  =  np.concatenate((self.spots_features, results[k].spots_features), axis=0)

        else:
            self.spots_features  =  SpotsFeatureExtractorUtility([spots_trckd, spots_idxs, spots_3d_coords, raw_data])

        self.spots_idxs      =  spots_idxs
        self.min_act_frames  =  min_act_frames


class SpotsFeatureExtractorUtility:
    """Define spots cages and measure the background for each spot."""
    def __init__(self, arg_input):

        spots_trckd      =  arg_input[0]
        spots_idxs       =  arg_input[1]
        spots_3d_coords  =  arg_input[2]
        raw_data         =  arg_input[3]
        t_steps          =  arg_input[4]

        # tlen            =  spots_trckd.shape[0]
        spots_features  =  np.zeros((t_steps.size, 7, spots_idxs.size))         # initialize output matrix (for each spot at each time frame we have volume, intensity, background value, intensity divided by background and the 3D coordinate of the centroid)

        for ff, tt in enumerate(t_steps):
            spots_fr   =  reconstruc3d(spots_3d_coords, tt)                                                                        # reconstruct spot in 3d (it is b&w)
            spots_fr  *=  spots_trckd[ff]                                                                                          # give proper label to the 3d spots
            cage_fr    =  expand_labels(spots_fr, 5) - expand_labels(spots_fr, 3)                                                  # expand in 3d each label 5 times and subtract the same label expoanded 3 times to have cages around
            rgp_cage   =  regionprops_table(cage_fr, raw_data[ff], properties=["label", "area", "intensity_image"])                # regionprops of the cages
            rgp_spots  =  regionprops_table(spots_fr, raw_data[ff], properties=["label", "area", "intensity_image", "centroid"])   # regionprops of the spots (of course cage and spots label are the same)
            for cnt, lb in enumerate(rgp_cage["label"]):                                                                           # store info: volume, intensity, av_bckgd, ints by bckgd, centroid
                lb_idx                         =  np.where(lb == spots_idxs)[0]
                spots_features[ff, :, lb_idx]  =  rgp_spots["area"][cnt], np.sum(rgp_spots["intensity_image"][cnt]), np.sum(rgp_cage["intensity_image"][cnt]) / rgp_cage["area"][cnt], np.sum(rgp_spots["intensity_image"][cnt]) / (np.sum(rgp_cage["intensity_image"][cnt]) / rgp_cage["area"][cnt]), rgp_spots["centroid-0"][cnt], rgp_spots["centroid-1"][cnt], rgp_spots["centroid-2"][cnt]

        self.spots_features  =  spots_features
