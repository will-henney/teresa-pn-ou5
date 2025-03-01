from __future__ import annotations
from typing import Union
import numpy as np
from reproject import reproject_interp  # type: ignore
from astropy.io import fits  # type: ignore
from astropy.wcs import WCS  # type: ignore
from astropy.table import Table  # type: ignore
from astropy.coordinates import SkyCoord  # type: ignore
import astropy.units as u  # type: ignore
from FITS_tools.hcongrid import hcongrid_hdu

HDU = Union[fits.PrimaryHDU, fits.CompImageHDU, fits.ImageHDU]


def regrid_pv(
    hdu_in: HDU,
    center: tuple[float, float] = (0.0, 0.0),
    shape: tuple[int, int] = (512, 512),
    pixscale: tuple[float, float] = (1.0, 0.3),
    copy_kwds: list = ["PA", "OFFSET", "WEIGHT"],
) -> fits.HDUList:
    """
    Regrid position-velocity images to a given frame by reprojection

    Processes all pvs in `hdulist_in`.  All 2-tuple arguments
    `center`, `shape`, and `pixscale` are in FITS axis order: (x, y)
    where x is velocity and y is position.  The units are assumed to
    be km/s and arcsec.
    """

    #
    # First set up WCS for the output image
    # We use capital letters for the output variables
    #
    NX, NY = shape
    dv, ds = pixscale
    v0, s0 = center
    W = WCS(naxis=2)
    W.wcs.cdelt = pixscale
    W.wcs.crpix = [0.5 * (1 + NX), 0.5 * (1 + NY)]
    W.wcs.crval = center
    W.wcs.ctype = ["VHEL", "LINEAR"]
    W.wcs.cunit = ["km s-1", "arcsec"]

    # hcongrid does not have a way to pass in the output shape
    # separately, so we must make a header that has that information
    # in it, which is not in the WCS
    hdr_out = fits.PrimaryHDU(
        header=W.to_header(),
        data=np.zeros((NY, NX)),
    ).header
    # Copy over information we want to preserve from the original header
    for kwd in copy_kwds:
        hdr_out[kwd] = hdu_in.header[kwd]

    # The interpolation uses hcongrid instead of reproject because the
    # latter only works with celestial coordinates
    return hcongrid_hdu(hdu_in, hdr_out)


def regrid_images(
    hdulist_in: fits.HDUList,
    center: SkyCoord = SkyCoord(0.0 * u.deg, 0.0 * u.deg),
    shape: tuple[float, float] = (512, 512),
    pixscale: u.Quantity = 0.3 * u.arcsec,
) -> fits.HDUList:
    """
    Regrid images to a particular frame by reprojection

    Processes all images in `hdulist_in`.  The WCS of each image must
    be in celestial coordinates.
    """

    #
    # First set up WCS for the output image
    # We use capital letters for the output variables
    #
    NX, NY = shape
    _pixscale = pixscale.to(u.deg).value
    dRA, dDec = -_pixscale, _pixscale
    RA0, Dec0 = center.ra.deg, center.dec.deg
    W = WCS(naxis=2)
    W.wcs.cdelt = [dRA, dDec]
    W.wcs.crpix = [0.5 * (1 + NX), 0.5 * (1 + NY)]
    W.wcs.crval = [RA0, Dec0]
    W.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    list_out = []
    for hdu_in in hdulist_in:
        _HDU = type(hdu_in)
        if _HDU in (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU):
            # Interpolate the data to the new frame
            data_out = reproject_interp(
                hdu_in, W, shape_out=shape, return_footprint=False
            )
            # Just nuke the previous header - the only way to be sure
            hdr_out = W.to_header()
            list_out.append(_HDU(header=hdr_out, data=data_out))
    return fits.HDUList(list_out)
