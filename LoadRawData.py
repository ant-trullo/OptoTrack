"""This file loads .czi raw data files.
Loaded matrix is fliped and rotate to adapt to ImageJ visualization.

Input is a file path, output is a TxZxXxY file.
There is only one channel.
"""

from importlib import reload
import numpy as np
from aicsimageio import AICSImage
import tifffile
import czifile
from skimage.filters import gaussian

import ServiceWidgets


class LoadRawDataCzi:
    """Load and concatenate .czi files."""
    def __init__(self, fnames, ch_numb=-1):

        reload(ServiceWidgets)
        img  =  AICSImage(fnames[0])                                                               # read and load file
        try:
            pix_size_xy  =  img.physical_pixel_sizes.X
            pix_size_z   =  img.physical_pixel_sizes.Z

        except:
            pix_size_xy, pix_size_z  =  ServiceWidgets.InputPixSize.get_vals()

        with czifile.CziFile(str(fnames[0])) as czi:                                                # read the time step
            for attachment in czi.attachments():
                if attachment.attachment_entry.name == 'TimeStamps':
                    timestamps  =  attachment.data()
                    break
            else:
                raise ValueError('TimeStamps not found')
        time_step_value  =  np.round(timestamps[1] - timestamps[0], 2)                    # time step value

        if ch_numb == -1:
            if img.dims.C > 1:                                                                      # if the number of channels is higher than one there several channels
                ch_numb  =  ServiceWidgets.ChannelNumber.getNumb(img.channel_names) - 1             # choose the channel you want to work on
            else:
                ch_numb  =  0                                                                       # only one channel, work on the channel 0

        raw_data  =  img.get_image_data("TZXY", C=ch_numb)                     # get image

        pbar  =  ServiceWidgets.ProgressBar(total1=len(fnames))
        pbar.show()
        pbar.update_progressbar1(1)

        raw_data_mip  =  np.zeros((img.dims.T, img.dims.X, img.dims.Y), dtype=raw_data.dtype)   # initialize empty matrix for mip
        for uu in range(img.dims.T):
            for xx in range(img.dims.X):
                raw_data_mip[uu, xx, :]  =  raw_data[uu, :, xx, :].max(0)                               # mip

        for cn, fname in enumerate(fnames[1:]):                                                         # for all the other file (if any) repeat the same operations
            pbar.update_progressbar1(cn + 1)
            img_bff       =  AICSImage(fname)
            raw_data_bff  =  img_bff.get_image_data("TZXY", C=ch_numb)

            raw_data_bff_mip  =  np.zeros((img_bff.dims.T, img_bff.dims.X, img_bff.dims.Y), dtype=raw_data.dtype)
            for uu in range(img_bff.dims.T):
                for xx in range(img_bff.dims.X):
                    raw_data_bff_mip[uu, xx, :]  =  raw_data_bff[uu, :, xx, :].max(0)

            raw_data      =  np.concatenate((raw_data_bff, raw_data), axis=0)
            raw_data_mip  =  np.concatenate((raw_data_bff_mip, raw_data_mip), axis=0)

        pbar.close()

        self.raw_data         =  raw_data
        self.raw_data_mip     =  raw_data_mip
        self.pix_size_z       =  pix_size_z
        self.pix_size_xy      =  pix_size_xy
        self.ch_numb          =  ch_numb
        self.time_step_value  =  time_step_value


class LoadRawDataCziPresmooth:
    """Load and concatenate .czi files as in the previous class, but with a smoothing."""
    def __init__(self, fnames, ch_numb=-1):

        reload(ServiceWidgets)
        img  =  AICSImage(fnames[0])
        try:
            pix_size_xy  =  img.physical_pixel_sizes.X
            pix_size_z   =  img.physical_pixel_sizes.Z
        except:
            pix_size_xy, pix_size_z  =  ServiceWidgets.InputPixSize.get_vals()

        with czifile.CziFile(str(fnames[0])) as czi:
            for attachment in czi.attachments():
                if attachment.attachment_entry.name == 'TimeStamps':
                    timestamps  =  attachment.data()
                    break
            else:
                raise ValueError('TimeStamps not found')
        time_step_value  =  np.round(timestamps[1] - timestamps[0], 2)  # time step value

        if ch_numb == -1:
            if img.dims.C > 1:
                ch_numb   =  ServiceWidgets.ChannelNumber.getNumb(img.channel_names) - 1
            else:
                ch_numb  =  0

        pbar  =  ServiceWidgets.ProgressBar(total1=len(fnames))
        pbar.show()
        pbar.update_progressbar1(1)

        raw_data  =  img.get_image_data("TZXY", C=ch_numb)
        for tt in range(raw_data.shape[0]):
            bff           =  gaussian(raw_data[tt], 1.5)
            raw_data[tt]  =  (bff * 1000).astype(np.uint16)

        raw_data_mip  =  np.zeros((img.dims.T, img.dims.X, img.dims.Y), dtype=raw_data.dtype)
        for uu in range(img.dims.T):
            for xx in range(img.dims.X):
                raw_data_mip[uu, xx, :]  =  raw_data[uu, :, xx, :].max(0)

        for cn, fname in enumerate(fnames[1:]):
            pbar.update_progressbar1(cn + 1)
            img_bff       =  AICSImage(fname)
            raw_data_bff  =  img_bff.get_image_data("TZXY", C=ch_numb)

            for tt in range(img_bff.dims.T):
                raw_data_bff[tt]  =  (1000 * gaussian(raw_data_bff[tt], 1.5)).astype(np.uint16)

            raw_data_bff_mip  =  np.zeros((img_bff.dims.T, img_bff.dims.X, img_bff.dims.Y), dtype=raw_data.dtype)
            for uu in range(img_bff.dims.T):
                for xx in range(img_bff.dims.X):
                    raw_data_bff_mip[uu, xx, :]  =  raw_data_bff[uu, :, xx, :].max(0)

            raw_data      =  np.concatenate((raw_data_bff, raw_data), axis=0)
            raw_data_mip  =  np.concatenate((raw_data_bff_mip, raw_data_mip), axis=0)

        pbar.close()

        self.raw_data         =  raw_data
        self.raw_data_mip     =  raw_data_mip
        self.pix_size_z       =  pix_size_z
        self.pix_size_xy      =  pix_size_xy
        self.ch_numb          =  ch_numb
        self.time_step_value  =  time_step_value


class LoadRawDataTiff:
    """Load tiff files."""
    def __init__(self, fnames, ch_numb=-1):

        pos_ch   =  None
        ch_numb  =  None

        raw_data_bff  =  np.squeeze(tifffile.imread(fnames[0]))
        if len(raw_data_bff.shape) > 4:
            if ch_numb == -1:
                ch_numb  =  ServiceWidgets.ChannelNumber.getNumb()
            pos_ch   =  np.argmin(np.asarray(raw_data_bff.shape))
            if pos_ch == 0:
                raw_data_bff  =  raw_data_bff[ch_numb]
            elif pos_ch == 1:
                raw_data_bff  =  raw_data_bff[:, ch_numb]
            elif pos_ch == 2:
                raw_data_bff  =  raw_data_bff[:, :, ch_numb]

        tlen, zlen, ylen, xlen  =  raw_data_bff.shape
        raw_data                =  np.zeros((tlen, zlen, xlen, ylen))
        for tt in range(tlen):
            raw_data[tt]  =  np.rot90(raw_data_bff[tt, :, :, ::-1], axes=(1, 2))

        pbar  =  ServiceWidgets.ProgressBar(total1=len(fnames))
        pbar.show()
        pbar.update_progressbar1(1)

        raw_data_mip  =  np.zeros((tlen, xlen, ylen), dtype=raw_data_bff.dtype)
        for uu in range(tlen):
            for xx in range(xlen):
                raw_data_mip[uu, xx, :]  =  raw_data[uu, :, xx, :].max(0)

        for cn, fname in enumerate(fnames[1:]):
            pbar.update_progressbar1(cn + 1)
            raw_data_bff  =  np.squeeze(tifffile.imread(fname))
            if len(raw_data_bff.shape) > 4:
                if pos_ch == 0:
                    raw_data_bff  =  raw_data_bff[ch_numb]
                elif pos_ch == 1:
                    raw_data_bff  =  raw_data_bff[:, ch_numb]
                elif pos_ch == 2:
                    raw_data_bff  =  raw_data_bff[:, :, ch_numb]

            raw_data_bff2  =  np.zeros((tlen, zlen, xlen, ylen), dtype=raw_data_bff.dtype)
            for tt in range(tlen):
                raw_data_bff2[tt]  =  np.rot90(raw_data_bff[tt, :, :, ::-1], axes=(1, 2))

            raw_data_bff_mip  =  np.zeros((tlen, xlen, ylen))
            for uu in range(tlen):
                for xx in range(xlen):
                    raw_data_bff_mip[uu, xx, :]  =  raw_data_bff2[uu, :, xx, :].max(0)

            raw_data      =  np.concatenate((raw_data_bff2, raw_data), axis=0)
            raw_data_mip  =  np.concatenate((raw_data_bff_mip, raw_data_mip), axis=0)

        pbar.close()

        pix_size_xy, pix_size_z  =  ServiceWidgets.InputPixSize.get_vals()

        self.raw_data      =  raw_data
        self.raw_data_mip  =  raw_data_mip
        self.pix_size_z    =  pix_size_z
        self.pix_size_xy   =  pix_size_xy
