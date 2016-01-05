# pixeldata.py
"""Module for PixelData handling
"""
#
# Copyright (c) 2016 Korijn van Golen
# This file is part of pydicom, released under a modified MIT license.
#    See the file license.txt included with this distribution, also
#    available at https://github.com/darcymason/pydicom
#
import sys

from pydicom.compat import in_py2
from pydicom.charset import default_encoding
from pydicom.uid import is_compressed_transfer_syntax

sys_is_little_endian = (sys.byteorder == 'little')

have_numpy = True
try:
    import numpy as np
except ImportError:
    have_numpy = False

have_gdcm = True
try:
    import gdcm

    gdcm_photometric_interpretation_typemap = {
        'UNKNOW': gdcm.PhotometricInterpretation.UNKNOW,
        'MONOCHROME1': gdcm.PhotometricInterpretation.MONOCHROME1,
        'MONOCHROME2': gdcm.PhotometricInterpretation.MONOCHROME2,
        'PALETTE_COLOR': gdcm.PhotometricInterpretation.PALETTE_COLOR,
        'RGB': gdcm.PhotometricInterpretation.RGB,
        'HSV': gdcm.PhotometricInterpretation.HSV,
        'ARGB': gdcm.PhotometricInterpretation.ARGB,
        'CMYK': gdcm.PhotometricInterpretation.CMYK,
        'YBR_FULL': gdcm.PhotometricInterpretation.YBR_FULL,
        'YBR_FULL_422': gdcm.PhotometricInterpretation.YBR_FULL_422,
        'YBR_PARTIAL_422': gdcm.PhotometricInterpretation.YBR_PARTIAL_422,
        'YBR_PARTIAL_420': gdcm.PhotometricInterpretation.YBR_PARTIAL_420,
        'YBR_ICT': gdcm.PhotometricInterpretation.YBR_ICT,
        'YBR_RCT': gdcm.PhotometricInterpretation.YBR_RCT,
    }

    gdcm_transfer_syntax_typemap = {
        '1.2.840.10008.1.2': gdcm.TransferSyntax.ImplicitVRLittleEndian,
        '1.2.840.113619.5.2': gdcm.TransferSyntax.ImplicitVRBigEndianPrivateGE,
        '1.2.840.10008.1.2.1': gdcm.TransferSyntax.ExplicitVRLittleEndian,
        '1.2.840.10008.1.2.1.99': gdcm.TransferSyntax.DeflatedExplicitVRLittleEndian,
        '1.2.840.10008.1.2.2': gdcm.TransferSyntax.ExplicitVRBigEndian,
        '1.2.840.10008.1.2.4.50': gdcm.TransferSyntax.JPEGBaselineProcess1,
        '1.2.840.10008.1.2.4.51': gdcm.TransferSyntax.JPEGExtendedProcess2_4,
        '1.2.840.10008.1.2.4.52': gdcm.TransferSyntax.JPEGExtendedProcess3_5,
        '1.2.840.10008.1.2.4.53': gdcm.TransferSyntax.JPEGSpectralSelectionProcess6_8,
        '1.2.840.10008.1.2.4.55': gdcm.TransferSyntax.JPEGFullProgressionProcess10_12,
        '1.2.840.10008.1.2.4.57': gdcm.TransferSyntax.JPEGLosslessProcess14,
        '1.2.840.10008.1.2.4.70': gdcm.TransferSyntax.JPEGLosslessProcess14_1,
        '1.2.840.10008.1.2.4.80': gdcm.TransferSyntax.JPEGLSLossless,
        '1.2.840.10008.1.2.4.81': gdcm.TransferSyntax.JPEGLSNearLossless,
        '1.2.840.10008.1.2.4.90': gdcm.TransferSyntax.JPEG2000Lossless,
        '1.2.840.10008.1.2.4.91': gdcm.TransferSyntax.JPEG2000,
        '1.2.840.10008.1.2.4.92': gdcm.TransferSyntax.JPEG2000Part2Lossless,
        '1.2.840.10008.1.2.4.93': gdcm.TransferSyntax.JPEG2000Part2,
        '1.2.840.10008.1.2.5': gdcm.TransferSyntax.RLELossless,
        '1.2.840.10008.1.2.4.100': gdcm.TransferSyntax.MPEG2MainProfile,
        # '': gdcm.TransferSyntax.ImplicitVRBigEndianACRNEMA,
        # '': gdcm.TransferSyntax.CT_private_ELE,
        '1.2.840.10008.1.2.4.94': gdcm.TransferSyntax.JPIPReferenced,
        '1.2.840.10008.1.2.4.101': gdcm.TransferSyntax.MPEG2MainProfileHighLevel,
        '1.2.840.10008.1.2.4.102': gdcm.TransferSyntax.MPEG4AVCH264HighProfileLevel4_1,
        '1.2.840.10008.1.2.4.103': gdcm.TransferSyntax.MPEG4AVCH264BDcompatibleHighProfileLevel4_1,
    }

    gdcm_numpy_pixel_format_typemap = {
        gdcm.PixelFormat.INT8: np.int8,
        gdcm.PixelFormat.UINT8: np.uint8,
        # gdcm.PixelFormat.UINT12:   np.uint12,
        # gdcm.PixelFormat.INT12:    np.int12,
        gdcm.PixelFormat.UINT16: np.uint16,
        gdcm.PixelFormat.INT16: np.int16,
        gdcm.PixelFormat.UINT32: np.uint32,
        gdcm.PixelFormat.INT32: np.int32,
        # gdcm.PixelFormat.UINT64:   np.uint64,
        # gdcm.PixelFormat.INT64:    np.int64,
        # gdcm.PixelFormat.FLOAT16:  np.float16,
        gdcm.PixelFormat.FLOAT32: np.float32,
        gdcm.PixelFormat.FLOAT64: np.float64,
        # gdcm.PixelFormat.SINGLEBIT:  np.bit,
        # gdcm.PixelFormat.UNKNOWN:  np.unknown,
    }
except:
    have_gdcm = False


def numpy_dtype_from_dataset(ds):
    # Make NumPy format code, e.g. "uint16", "int32" etc
    # from two pieces of info:
    #    ds.PixelRepresentation -- 0 for unsigned, 1 for signed;
    #    ds.BitsAllocated -- 8, 16, or 32
    format_str = '%sint%d' % (('u', '')[ds.PixelRepresentation],
                              ds.BitsAllocated)
    try:
        numpy_dtype = np.dtype(format_str)
    except TypeError:
        msg = ("Data type not understood by NumPy: "
               "format='%s', PixelRepresentation=%d, BitsAllocated=%d")
        raise TypeError(msg % (format_str, ds.PixelRepresentation,
                               ds.BitsAllocated))

    if ds.is_little_endian != sys_is_little_endian:
        numpy_dtype.newbyteorder('S')

    return numpy_dtype


def construct_gdcm_pixel_data_element(ds):
    pixel_data_element = gdcm.DataElement(gdcm.Tag(0x7fe0, 0x0010))
    sequence_of_fragments = gdcm.SequenceOfFragments.New()

    fragment = gdcm.Fragment()
    if not in_py2:
        # the GDCM wrappers are generated by SWIG, which - under Python 3 - expects all passed C++ char * pointers
        # to be byte strings instead of bytearrays, so we have to try and decode it here
        byte_string = ds.PixelData.decode(default_encoding)
        fragment.SetByteValue(byte_string, gdcm.VL(len(byte_string)))
    else:
        fragment.SetByteValue(ds.PixelData, gdcm.VL(len(ds.PixelData)))

    sequence_of_fragments.AddFragment(fragment)
    pixel_data_element.SetValue(sequence_of_fragments.__ref__())

    return pixel_data_element


def construct_gdcm_image_from_dataset(ds):
    image = gdcm.Image()

    # construct GDCM PixelData element
    pixel_data_element = construct_gdcm_pixel_data_element(ds)
    image.SetDataElement(pixel_data_element)

    # set photometric interpretation
    image.SetPhotometricInterpretation(
        gdcm.PhotometricInterpretation(gdcm_photometric_interpretation_typemap[ds.PhotometricInterpretation]))

    # set pixel format
    pixel_format = gdcm.PixelFormat(ds.SamplesPerPixel, ds.BitsAllocated, ds.BitsStored, ds.HighBit,
                                       ds.PixelRepresentation)
    image.SetPixelFormat(pixel_format)

    # set transfer syntax
    transfer_syntax_uid = ds.file_meta.TransferSyntaxUID.title()
    image.SetTransferSyntax(gdcm.TransferSyntax(gdcm_transfer_syntax_typemap[transfer_syntax_uid]))

    # set dimensions (GDCM is column major)
    image.SetNumberOfDimensions(2)
    image.SetDimension(0, ds.Columns)
    image.SetDimension(1, ds.Rows)

    return image


def reshape_pixel_array(ds, pixel_array):
    # Note the following reshape operations return a new *view* onto pixel_array, but don't copy the data
    if 'NumberOfFrames' in ds and ds.NumberOfFrames > 1:
        if ds.SamplesPerPixel > 1:
            # TODO: Handle Planar Configuration attribute
            assert ds.PlanarConfiguration == 0
            pixel_array = pixel_array.reshape(ds.NumberOfFrames, ds.Rows, ds.Columns, ds.SamplesPerPixel)
        else:
            pixel_array = pixel_array.reshape(ds.NumberOfFrames, ds.Rows, ds.Columns)
    else:
        if ds.SamplesPerPixel > 1:
            if ds.BitsAllocated == 8:
                if ds.PlanarConfiguration == 0:
                    pixel_array = pixel_array.reshape(ds.Rows, ds.Columns, ds.SamplesPerPixel)
                else:
                    pixel_array = pixel_array.reshape(ds.SamplesPerPixel, ds.Rows, ds.Columns)
                    pixel_array = pixel_array.transpose(1, 2, 0)
            else:
                raise NotImplementedError("This code only handles SamplesPerPixel > 1 if Bits Allocated = 8")
        else:
            pixel_array = pixel_array.reshape(ds.Rows, ds.Columns)
    return pixel_array


def get_pixel_data(ds):
    """Return a NumPy array of the pixel data if NumPy is available.
    Falls back to GDCM in case of unsupported transfer syntaxes.

    Raises
    ------
    TypeError
        If there is no pixel data or not a supported data type
    ImportError
        If NumPy isn't found, or in the case of fallback, if GDCM isn't found.

    Returns
    -------
    NumPy array
    """
    compressed_pixeldata = is_compressed_transfer_syntax(ds.file_meta.TransferSyntaxUID)
    if compressed_pixeldata and not have_gdcm:
        raise NotImplementedError(
            "Pixel Data is compressed in a format pydicom does not yet handle. Cannot return array. Pydicom might be able to convert the pixel data using GDCM if it is installed.")
    if not have_numpy:
        raise ImportError("The Numpy package is required to use pixel_array, and numpy could not be imported.")
    if 'PixelData' not in ds:
        raise TypeError("No pixel data found in this dataset.")

    numpy_dtype = numpy_dtype_from_dataset(ds)
    pixel_bytearray = ds.PixelData

    if compressed_pixeldata and have_gdcm:
        # when pixeldata is compressed, we forward decompression handling to GDCM if it is available
        gdcm_image = construct_gdcm_image_from_dataset(ds)

        # get decompressed pixel array
        pixel_bytearray = gdcm_image.GetBuffer()
        if not in_py2:
            # the GDCM wrappers are generated by SWIG, which - under Python 3 - decodes all returned C++ char * pointers
            # as UTF-8 unicode strings with the surrogateescape error handler, so we can get the original bytearray
            # by encoding with the same parameters
            pixel_bytearray = pixel_bytearray.encode("utf-8", "surrogateescape")

        # determine the correct numpy datatype
        gdcm_pixel_format = gdcm_image.GetPixelFormat().GetScalarType()
        numpy_dtype = gdcm_numpy_pixel_format_typemap[gdcm_pixel_format]

        # if GDCM indicates that a byte swap is in order, make sure to inform numpy as well
        if gdcm_image.GetNeedByteSwap():
            numpy_dtype.newbyteorder('S')

    # finally, actually grab the bytearray through numpy
    # we use fromstring to copy the memory, so that all the GDCM variables can be garbage-collected
    pixel_array = np.fromstring(pixel_bytearray, dtype=numpy_dtype)

    # finally, reshape to match the image dimensions
    pixel_array = reshape_pixel_array(ds, pixel_array)

    return pixel_array


# Use by DataSet.pixel_array property
def get_pixel_data_cached(ds):
    # Check if already have converted to a NumPy array
    # Also check if ds.PixelData has changed. If so, get new NumPy array
    already_have = True
    if not hasattr(ds, "_pixel_array"):
        already_have = False
    elif ds._pixel_id != id(ds.PixelData):
        already_have = False
    if not already_have:
        ds._pixel_array = get_pixel_data(ds)
        ds._pixel_id = id(ds.PixelData)  # FIXME is this guaranteed to work if memory is re-used??
    return ds._pixel_array
