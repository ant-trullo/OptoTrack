"""This function tracks spots in 2D keeping anyway all the 3D information.

Input is spots already detected by another function and distance threshold (user defined).
"""


import numpy as np
from skimage.measure import regionprops_table    # , label
# from skimage.morphology import binary_dilation
from PyQt5 import QtWidgets, QtCore


def dist_one_several(ctrs_ref_bff, t2search_bff, lb_step_cntrds_bff):
    """This function calculates the distance between one centroid and all the centroids of the following frame."""
    lb_step_cntrds_bff_sub  =  lb_step_cntrds_bff[lb_step_cntrds_bff[:, 1] == t2search_bff]         # isolate from the matrix all the centroids of the frame to search
    if lb_step_cntrds_bff_sub.size == 0:                                                            # if in the considered frame there are no spots, put just a super high distance value
        dists  =  [1000000, 100000, 100000, 100000]
    else:
        dists  =  []
        for k in range(lb_step_cntrds_bff_sub.shape[0]):                                            # calculate the distance between the reference point and all the centroids of the considered frame (to speed up the code other values are stored)
            dists.append([lb_step_cntrds_bff_sub[k, 0], lb_step_cntrds_bff_sub[k, 2], lb_step_cntrds_bff_sub[k, 3], np.sqrt((ctrs_ref_bff[0] - lb_step_cntrds_bff_sub[k, 2])**2 + (ctrs_ref_bff[1] - lb_step_cntrds_bff_sub[k, 3])**2)])

    dists  =  np.asarray(dists)                                                                 # easier to work with array for these tasks
    if len(dists.shape) == 1:
        dists  =  np.expand_dims(dists, axis=0)                                                     # dists must be a nx4 array, that's way we add a dimension
    return dists


class SpotsTracker:
    """Main class, does all the job."""
    def __init__(self, spts_2d_lbls, dist_thr):

        tlen           =  spts_2d_lbls.shape[0]
        lb_step_cntds  =  []                                                                            # list of spots properties: label, t step and centroids (2D)
        for tt in range(tlen):
            rgp_bff_clean  =  regionprops_table(spts_2d_lbls[tt], properties=["label", "centroid"])
            for cnt, ll in enumerate(rgp_bff_clean["label"]):
                lb_step_cntds.append([ll, tt, np.round(rgp_bff_clean["centroid-0"][cnt]).astype(int), np.round(rgp_bff_clean["centroid-1"][cnt]).astype(int)])

        lb_step_cntds  =  np.asarray(lb_step_cntds)                                                      # np array is easier to handle
        spts_trck_fin  =  np.zeros_like(spts_2d_lbls)                                                    # initialize the output matrix

        pb        =  QtWidgets.QProgressBar()
        pb.setRange(0, 0)
        pb.show()
        new_tags      =  0                                                                                   # initialize tag to associate for the final trzacking
        empty_frames  =  0
        while np.sum(lb_step_cntds) > 0:                                                                 # while loop to be sure all the spots are tracked
            QtCore.QCoreApplication.processEvents()
            new_tags                           +=  1                                                     # update the new tag
            lb_step_cntds                       =  lb_step_cntds[lb_step_cntds[:, 1].argsort()]          # since we start from the spot first listed in the coordinate, we sort with the time to be sure to not pick a spot from an arbitrary frame and not from the beginning
            t_start                             =  lb_step_cntds[0, 1]                                   # appearence frame for the spot
            ctrs_ref                            =  lb_step_cntds[0, 2:]                                  # centroid coordinate as a reference
            spts_trck_fin[lb_step_cntds[0, 1]] +=  (spts_2d_lbls[lb_step_cntds[0, 1]] == lb_step_cntds[0, 0]) * new_tags     # add the spot in the final matrix with the proper tag
            lb_step_cntds                       =  lb_step_cntds[1:]                                     # remove the added spot properties from the list
            while t_start < tlen - 1:
                dd  =  dist_one_several(ctrs_ref, t_start + 1, lb_step_cntds)                            # calculate the distance between the selected spot and all the other in the following frames
                # print(dd.shape)
                if dd[:, 3].min() <= dist_thr:                                                            # if there is a spot close enough we engage it
                    # print(dd[:, 3].min())
                    cc                      =  np.argmin(dd[:, 3])                                       # search the matrix index in the distance matrix of the closest spot
                    ctrs_ref                =  dd[cc, 1:3]                                               # update the centroid reference
                    t_start                +=  1                                                         # update the current time
                    spts_trck_fin[t_start] +=  (spts_2d_lbls[t_start] == int(dd[cc, 0])) * new_tags      # add the new spot with the proper tag
                    # print(np.where(np.sum(np.abs(lb_step_cntds - np.array([dd[cc, 0], t_start, dd[cc, 1], dd[cc, 2]])), axis=1) == 0))
                    lb_step_cntds           =  np.delete(lb_step_cntds, np.where(np.sum(np.abs(lb_step_cntds - np.array([dd[cc, 0], t_start, dd[cc, 1], dd[cc, 2]])), axis=1) == 0), axis=0)   # remove the properties of the selected spot
                    empty_frames            =  0
                elif dd[:, 3].min() > dist_thr:
                    if empty_frames < 8:
                        t_start       +=  1                                                               # in case the minimum is bigger than the distance threshold (or in the considered frame there are no spots at all) we check in the following frame keeping the same reference centroid
                        empty_frames  +=  1
                    elif empty_frames >= 8:
                        break

        pb.close()
        self.spts_trck_fin  =  spts_trck_fin
        self.orig_tags      =  []
