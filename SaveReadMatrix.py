"""This function writes and reads a spots matrix as a 5xN array.

For each row of the array, there is the value of the tag and the t, z, x, y coordinates
of all the non-zero pixels. This is very useful in case of sparse matrix.
"""

import numpy as np
from skimage.measure import regionprops_table


class SpotsMatrixReader:
    """Function to reconstruct a spots matrix starting from its list of coordinates and tags."""
    def __init__(self, folder, fname):

        mtx2build  =  np.load(folder + fname)                                                                           # load the matrix
        spts_lbls  =  np.zeros((mtx2build[-1, 1], mtx2build[-1, 2], mtx2build[-1, 3]), dtype=np.uint32)                 # initialize the matrix image with the info in the matrix
        for kk in range(mtx2build.shape[0] - 1):
            spts_lbls[mtx2build[kk, 1], mtx2build[kk, 2], mtx2build[kk, 3]]  =  mtx2build[kk, 0]                        # fill the matrix with the info

        self.spts_lbls  =  spts_lbls


class SpotsMatrixSaver:
    """Function to write a ints spots matrix as a list of coordinates and tags."""
    def __init__(self, spts_lbls, folder, fname):

        rgp_spts  =  regionprops_table(np.sign(spts_lbls), properties=["coords"])                                       # regionprops to create a dictionary with al the info

        mtx2write  =  []                                                                                                # initialize list
        for oo in rgp_spts["coords"]:
            for cc in oo:
                mtx2write.append([spts_lbls[cc[0], cc[1], cc[2]], cc[0], cc[1], cc[2]])                                 # each element of the list is a list with label, z coord, x coord, y coord of each pixel

        mtx2write.append([0, spts_lbls.shape[0], spts_lbls.shape[1], spts_lbls.shape[2]])                               # last element is the size of the starting matrix image
        mtx2write  =  np.asarray(mtx2write)                                                                             # convert to array and write
        np.save(folder + fname, mtx2write)





# class SpotsMatrixSaver:
#     """Function to write a spots matrix as a list of coordinates and tags."""
#     def __init__(self, spts_lbls, folder, fname):
#
#         tlen       =  spts_lbls.shape[0]
#         mtx2write  =  list()                                                                                            # initialize list
#
#         for tt in range(tlen):
#             rgp_spts  =  regionprops_table(spts_lbls[tt], properties=["label", "coords"])                                   # regionprops to create a dictionary with al the info
#
#             for ll in range(rgp_spts["label"].size):
#                 for cc in range(rgp_spts["coords"][ll][:, 0].size):
#                     mtx2write.append([rgp_spts["label"][ll], tt, rgp_spts["coords"][ll][cc, 0], rgp_spts["coords"][ll][cc, 1], rgp_spts["coords"][ll][cc, 2]])  # each element of the list is a list with label, z coord, x coord, y coord of each pixel
#
#         mtx2write.append([0, tlen, spts_lbls.shape[1], spts_lbls.shape[2], spts_lbls.shape[3]])                         # last element is the size of the starting matrix image
#         mtx2write  =  np.asarray(mtx2write)                                                                             # convert to array and write
#         np.save(folder + fname, mtx2write)
#
#
# class SpotsMatrixReader:
#     """Function to reconstruct a spots matrix starting from its list of coordinates and tags."""
#     def __init__(self, folder, fname):
#
#         mtx2build  =  np.load(folder + fname)                                                                           # load the matrix
#         spts_lbls  =  np.zeros((mtx2build[-1, 1], mtx2build[-1, 2], mtx2build[-1, 3], mtx2build[-1, 4]), dtype=np.uint32)                 # initialize the matrix image with the info in the matrix
#         for kk in range(mtx2build.shape[0] - 1):
#             spts_lbls[mtx2build[kk, 1], mtx2build[kk, 2], mtx2build[kk, 3], mtx2build[kk, 4]]  =  mtx2build[kk, 0]                        # fill the matrix with the info
#
#         self.spts_lbls  =  spts_lbls
