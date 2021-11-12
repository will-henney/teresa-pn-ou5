"""
Create spectral maps from the longslit spectra

Originally developed for Teresa Turtle project.  Re-written as library
routine instead of script.
"""


import sys
from pathlib import Path
import numpy as np
from astropy.io import fits  # type: ignore
from astropy.wcs import WCS  # type: ignore
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel  # type: ignore
import astropy.units as u  # type: ignore

from .helio_utils import helio_topo_from_header, vels2waves


def waves2pixels(waves: np.ndarray, w: WCS) -> np.ndarray:
    """
    Convert array of wavelengths (in m) to array indices (integers)

    The implementation is now much simpler than it used to be, thanks
    to the new API calls in astropy.wcs
    """
    return w.spectral.world_to_array_index_values(waves)


def make_vmap(
    vel0: float,
    ra0: float,
    dec0: float,
    dvel: float = 20.0,
    line_id: str = "ha",
    datapath: Path = Path.cwd().parent / "data" / "pvextract",
    shape: tuple[int, int] = (512, 512),
    pixel_scale: float = 0.2,
    slit_width: float = 1.0,
    verbose: bool = False,
) -> fits.HDUList:
    """
    Construct isovelocity channel map from a set of slit spectra

    Parameters:
    -----------
    vel0 : float
        Central velocity of channel (km/s, heliocentric)
    ra0 : float
        RA of center of map in degrees.
    dec0 : float
        Dec of center of map in degrees.
    dvel : float, optional
        Width of velocity channel in km/s. Default is 20 km/s
    line_id : str, optional
        Name of emission line. This is used to glob for the files containing the PV spectra.
        Default is "ha".
    datapath : `pathlib.Path`, optional
        Path to folder containing PV spectra files.  Default is ../data/pvextract with
        respect to current working directory.
    shape : 2-tuple of (int, int), optional
        Shape of output image array. Default is (512, 512).
    pixel_scale : float, optional
        Linear size in arcsec of each pixel in output array. Default is 0.2 arcsec.
    slit_width : float, optional
        Width of each slit in arcsec. Default is 1.0 arcsec.
    verbose : bool, default: False

    Returns:
    --------
    `fits.HDUList`
        List of 3 HDU images. The first ("slits") is the sum of all the slit
        brightness times the per-slit weight.  The second ("weight") is the
        sum of the weights.  These two have zero for pixels where there is no slit.
        The third ("scaled") is the first divided by the second, so this is the one
        that has the best estimate of the channel brightness in each pixel. This has
        NaN for pixels where there is no slit.
    """
    # First set up WCS for the output image
    #
    NX, NY = shape
    pixel_scale = 0.3  # arcsec
    dRA, dDec = -pixel_scale / 3600.0, pixel_scale / 3600.0
    w = WCS(naxis=2)
    w.wcs.cdelt = [dRA, dDec]
    w.wcs.crpix = [0.5 * (1 + NX), 0.5 * (1 + NY)]
    w.wcs.crval = [ra0, dec0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ["deg", "deg"]

    # Arrays to hold the output image
    outimage = np.zeros((NY, NX))
    outweights = np.zeros((NY, NX))

    # Use a slightly wider slit than is strictly accurate
    slit_pix_width = slit_width / pixel_scale

    speclist = datapath.glob(f"*-{line_id}.fits")

    # Window widths for line and BG
    dwline = 7.0 * u.Angstrom

    v1 = vel0 - 0.5 * dvel
    v2 = vel0 + 0.5 * dvel
    vrange = f"V_{int(vel0):+04d}_W_{int(dvel):03d}"
    for fn in speclist:
        if verbose:
            print("Processing", fn)
        (spechdu,) = fits.open(fn)
        wspec = WCS(spechdu.header, key="A")

        # Trim to good portion of the slit
        goodslice = slice(None, None)

        # Find per-slit weight
        slit_weight = spechdu.header["WEIGHT"]

        # Find sign of delta wavelength
        dwav = wspec.wcs.get_cdelt()[0] * wspec.wcs.get_pc()[0, 0]
        sgn = int(dwav / abs(dwav))  # Need to take slices backwards if this is negative

        # Eliminate degenerate 3rd dimension from data array and trim off bad bits
        spec2d = spechdu.data[0]

        # Rest wavelength from FITS header is in meters
        wavrest = wspec.wcs.restwav * u.m

        if verbose:
            print("Velocity window:", v1, "to", v2)
        waves = vels2waves([v1, v2], wavrest, spechdu.header, usewcs="A")
        if verbose:
            print("Wavelength window: {:.2f} to {:.2f}".format(*waves.to(u.Angstrom)))

        # Find pixel indices for line extraction window
        i1, i2 = waves2pixels(waves, wspec)
        if verbose:
            print("Pixel window:", i1, "to", i2, "in direction", sgn)

        # Extract profile for this wavelength or velocity window
        profile = spec2d[:, i1:i2:sgn].sum(axis=-1)

        # Find celestial coordinates for each pixel along the slit
        NS = len(profile)
        slit_coords = pixel_to_skycoord(range(NS), [0] * NS, wspec, 0)

        # Trim off bad parts of slit
        profile = profile[goodslice]
        slit_coords = slit_coords[goodslice]

        # Deal with NaNs in profile:
        # - Make an array of per-pixel weights
        wp = np.ones_like(profile) * slit_weight
        # - Set profile and weight to zeros wherever there are NaNs
        badmask = ~np.isfinite(profile)
        profile[badmask] = 0.0
        wp[badmask] = 0.0

        # Convert to pixel coordinates in output image
        xp, yp = skycoord_to_pixel(slit_coords, w, 0)

        for x, y, bright, wt in zip(xp, yp, profile, wp):
            # Find output pixels corresponding to corners of slit pixel
            # (approximate as square)
            i1 = int(0.5 + x - slit_pix_width / 2)
            i2 = int(0.5 + x + slit_pix_width / 2)
            j1 = int(0.5 + y - slit_pix_width / 2)
            j2 = int(0.5 + y + slit_pix_width / 2)
            # Make sure we don't go outside the output grid
            i1, i2 = max(0, i1), max(0, i2)
            i1, i2 = min(NX, i1), min(NX, i2)
            j1, j2 = max(0, j1), max(0, j2)
            j1, j2 = min(NY, j1), min(NY, j2)
            # Fill in the square
            outimage[j1:j2, i1:i2] += bright * wt
            outweights[j1:j2, i1:i2] += wt

    # Save everything as different images in a single fits file:
    # 1. The sum of the raw slits
    # 2. The weights
    # 3. The slits normalized by the weights
    if vrange is None:
        label = line_id + "-allvels"
    else:
        label = line_id + vrange

    return fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(header=w.to_header(), data=outimage, name="slits"),
            fits.ImageHDU(header=w.to_header(), data=outweights, name="weight"),
            fits.ImageHDU(
                header=w.to_header(), data=outimage / outweights, name="scaled"
            ),
        ]
    )
