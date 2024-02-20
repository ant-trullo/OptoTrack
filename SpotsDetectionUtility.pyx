import numpy as np
cimport numpy as np


cpdef spts_int_vol(np.ndarray[np.int64_t, ndim=3] a, np.ndarray[np.int64_t, ndim=3] raw, list i_in):

    cdef np.ndarray[np.int64_t, ndim=2] d = np.zeros([a.shape[1], a.shape[2]], dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=2] s = np.zeros([a.shape[1], a.shape[2]], dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=3] g = np.zeros([a.shape[0], a.shape[1], a.shape[2]], dtype=np.int64)

    cdef int j_in
    cdef int zlen  =  a.shape[0]
    cdef int xlen  =  a.shape[1]
    cdef int ylen  =  a.shape[2]
    cdef int x, y, z

    for j_in in i_in:
        for z in range(zlen):
            for x in range(xlen):
                for y in range(ylen):
                    if a[z, x, y] == j_in:
                        s[x, y]    += 1
                        d[x, y]    += raw[z, x, y]
                        g[z, x, y] += j_in

    return s, d, g
