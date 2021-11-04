from __future__ import annotations
import numpy as np
from reproject import reproject_interp
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u


def regrid_images(
    hdulist_in: fits.HDUList,
    center: SkyCoord = SkyCoord(0.0 * u.deg, 0.0 * u.deg),
    shape: tuple[float, float] = (512, 512),
    pixscale: u.Quantity = 0.3 * u.arcsec,
) -> fits.HDUList:
    """
    Regrid images to a particular frame by reprojection

    Processes all images in `hdulist_in`
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
            # Carry over the header from the input ...
            hdr_out = hdu_in.header.copy()
            # ... except update it with the new WCS
            hdr_out.update(W.to_header())
            list_out.append(_HDU(header=hdr_out, data=data_out))
    return fits.HDUList(list_out)
