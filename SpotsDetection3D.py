"""This function detects spots in the 4D (time-x-y-z) stack.

This function is supposed to work in a multiprocessing pool. in_args are the
raw data (4D matrix) plus a threshold value. They are organized in a
list for multiprocessing purpose.
"""

import numpy as np
# from scipy.ndimage import filters
from scipy.ndimage import gaussian_filter, laplace
from scipy.stats import norm
from skimage.measure import label, regionprops  # , regionprops_table
from skimage.morphology import binary_dilation

import SpotsDetectionUtility


class SpotsDetection3D:
    """Class working on several time frames."""
    def __init__(self, in_args):                              # for multiprocessing purposes I need to define a single in_args variable which is a list. The relative class will act consequently

        green4D         =  in_args[0]
        thr_val         =  in_args[1]
        volume_thr_var  =  in_args[2]
        g_kern          =  np.load('gauss_kern_size.npy')

        steps, zlen, xlen, ylen  =  green4D.shape

        spots_ints    =  np.zeros((steps, xlen, ylen), dtype=np.int32)
        spots_vol     =  np.zeros((steps, xlen, ylen), dtype=np.int8)
        spots_coords  =  np.zeros((0, 4), dtype=np.int16)
        # spots_lbls    =  np.zeros((steps, zlen, xlen, ylen), dtype=np.int16)
        # spots_tzxy    =  np.zeros((0, 4), dtype=np.int16)

        for t in range(steps):
            # print(t)
            g21          =  green4D[t, :, :, :]                                           # for each time step, we Gaussian filter the 3D stack (x-y-z) and then Laplacian filter
            # g21g         =  filters.gaussian_filter(g21, float(g_kern))
            g21g         =  gaussian_filter(g21, float(g_kern))
            g21f         =  laplace(g21g.astype(np.float32))
            (mu, sigma)  =  norm.fit(np.abs(g21f))                                        # histogram is fitted with a Gaussian function
            g21f_thr     =  np.abs(g21f) > mu + thr_val * sigma                           # thresholding on the histogram
            g21f3dlbl    =  label(g21f_thr)                                               # labelling

            i_in       =  []                                                              # tags of spots that satisfies the conditions (volume and z planes)
            i_out      =  []                                                              # tags of spots that don't satisfy the conditions
            rgp_discr  =  regionprops(g21f3dlbl)
            for k in range(len(rgp_discr)):
                if rgp_discr[k]['area'] > volume_thr_var and np.diff(np.sort(rgp_discr[k]['coords'][:, 0])).sum() > 0:     # in 3D 'area' is volume; with 'coords' we plot all the values of the z coordinates of all the pixels.
                    i_in.append(rgp_discr[k]['label'])                                                                     # then we calculate the sum of the derivative: if it is zero, the spot is present only in una z-frame, so it is removed
                    # spots_tzxy  =  np.concatenate([spots_tzxy, [np.append(t, np.round(np.asarray(rgp_discr[k]['centroid'])).astype(np.int))]], axis=0)    # coordinates of the good spots
                else:
                    i_out.append(rgp_discr[k]['label'])

            zz              =  SpotsDetectionUtility.spts_int_vol(g21f3dlbl.astype(np.int64), g21.astype(np.int64), i_in)             # output are 3 matrices: spot intensity summed in z, volume summed in z and 3D spots to remove
            spots_ints[t]  +=  zz[1]
            spots_vol[t]   +=  zz[0].astype(np.int8)
            # spots_lbls[t]  +=  zz[2].astype(np.int16)
            rgp            =  regionprops(zz[2])
            # rgp_spts_lbls  =  regionprops_table(zz[2], properties=["label", "coors"])
            # coords_bff  =  np.zeros((0, 3), np.int16)
            for rr in rgp:
                # coords_bff  =   np.concatenate((coords_bff, rr['coords']), axis=0)

                spots_coords  =  np.concatenate((spots_coords, np.column_stack((t * np.ones(rr['coords'].shape[0], dtype=np.int16), rr['coords']))), axis=0)
            # spots_coords.append(coords_bff)

        self.spots_ints    =  spots_ints
        self.spots_vol     =  spots_vol
        # self.spots_tzxy    =  spots_tzxy
        # self.spots_lbls    =  spots_lbls
        self.spots_coords  =  spots_coords


class SpotsDetection3D_Single:
    """Class working on a single time frame."""
    def __init__(self, in_args):                                                    # for multiprocessing purposes I need to define a single in_args variable which is a list. The relative class will act consequently

        green4D         =  in_args[0]
        thr_val         =  in_args[1]
        volume_thr_var  =  in_args[2]

        zlen, xlen, ylen  =  green4D.shape

        spots_ints   =  np.zeros((xlen, ylen))
        spots_vol    =  np.zeros((xlen, ylen))
        g21          =  green4D                                                                    # for each time step, we Gaussian filter the 3D stack (x-y-z) and than Laplacian filter
        g21g         =  gaussian_filter(g21, 1)
        g21f         =  laplace(g21g.astype(float))
        (mu, sigma)  =  norm.fit(np.abs(g21f))                                                                      # histogram is fitted with a Gaussian function
        g21f_thr     =  np.abs(g21f) > mu + thr_val * sigma                                                    # thresholding on the histogram
        g21f3dlbl    =  label(g21f_thr)                                                                             # labelling
        g2show       =  np.zeros(g21f3dlbl.shape)

        i_in       =  []                                                              # tags of spots that satisfies the conditions (volume and z planes)
        i_out      =  []                                                              # tags of spots that don't satisfy the conditions
        rgp_discr  =  regionprops(g21f3dlbl)
        for k in range(len(rgp_discr)):
            if rgp_discr[k]['area'] > volume_thr_var and np.diff(np.sort(rgp_discr[k]['coords'][:, 0])).sum() > 0:     # in 3D 'area' is volume; with 'coords' we plot all the values of the z coordinates of all the pixels.
                i_in.append(rgp_discr[k]['label'])                                                                     # then we and calculate the sum of the derivative: if it is zero, and one z and and so on
                g2show  +=  g21f3dlbl == rgp_discr[k]['label']
            else:
                i_out.append(rgp_discr[k]['label'])

        zz           =  SpotsDetectionUtility.spts_int_vol(g21f3dlbl.astype(np.int64), g21.astype(np.int64), i_in)             # output are 3 matrices: spot intensity summed in z, volume summed in z and 3D spots to remove
        spots_ints  +=  zz[1]
        spots_vol   +=  zz[0]

        self.spots_ints  =  spots_ints
        self.spots_vol   =  spots_vol
        self.spots_lbls  =  np.sign(g2show)


class SpotsDetection3D_Single4Test:
    """Class working on a single time frame."""
    def __init__(self, in_args):                                                    # for multiprocessing purposes I need to define a single in_args variable which is a list. The relative class will act consequently

        green4D         =  in_args[0]
        thr_val         =  in_args[1]
        volume_thr_var  =  in_args[2]
        merge_radius    =  in_args[3]
        g_kern          =  np.load('gauss_kern_size.npy')
        # print("outside")
        zlen, xlen, ylen  =  green4D.shape

        spots_ints   =  np.zeros((xlen, ylen))
        spots_vol    =  np.zeros((xlen, ylen))
        g21          =  green4D                                                                    # for each time step, we Gaussian filter the 3D stack (x-y-z) and than Laplacian filter
        g21g         =  gaussian_filter(g21, float(g_kern))
        g21f         =  laplace(g21g.astype(float))
        (mu, sigma)  =  norm.fit(np.abs(g21f))                                                     # histogram is fitted with a Gaussian function
        g21f_thr     =  np.abs(g21f) > mu + thr_val * sigma                                        # thresholding on the histogram
        g21f3dlbl    =  label(g21f_thr)                                                            # labelling
        g2show       =  np.zeros(g21f3dlbl.shape)

        i_in       =  []                                                              # tags of spots that satisfies the conditions (volume and z planes)
        i_out      =  []                                                              # tags of spots that don't satisfy the conditions
        rgp_discr  =  regionprops(g21f3dlbl)
        for k in range(len(rgp_discr)):
            if rgp_discr[k]['area'] > volume_thr_var and np.diff(np.sort(rgp_discr[k]['coords'][:, 0])).sum() > 0:     # in 3D 'area' is volume; with 'coords' we plot all the values of the z coordinates of all the pixels.
                i_in.append(rgp_discr[k]['label'])                                                                     # then we and calculate the sum of the derivative: if it is zero, and one z and and so on
                g2show  +=  g21f3dlbl == rgp_discr[k]['label']
            else:
                i_out.append(rgp_discr[k]['label'])

        zz           =  SpotsDetectionUtility.spts_int_vol(g21f3dlbl.astype(np.int64), g21.astype(np.int64), i_in)             # output are 3 matrices: spot intensity summed in z, volume summed in z and 3D spots to remove
        spots_ints  +=  zz[1]
        spots_vol   +=  zz[0]

        spots_bw     =  np.sign(g2show)
        spots_lbls   =  label(spots_bw)
        if merge_radius > 0:
            spots_merged  =  np.sign(spots_lbls.sum(0))
            spots_merged  =  binary_dilation(spots_merged, np.ones((merge_radius, merge_radius)))
            spots_clean   =  spots_merged * label(np.sign(spots_lbls.sum(0)))

        else:
            spots_clean  =  label(np.sign(spots_lbls.sum(0)))

        self.spots_ints   =  spots_ints
        self.spots_vol    =  spots_vol
        self.spots_lbls   =  spots_lbls
        self.spots_clean  =  spots_clean
