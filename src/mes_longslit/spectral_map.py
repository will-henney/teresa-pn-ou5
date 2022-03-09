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
from astropy.coordinates import SkyCoord  # type: ignore
from .helio_utils import helio_topo_from_header, vels2waves, waves2vels


# This is derived from the data in Table 1 of Meaburn et al 2003RMxAA..39..185M
#
# For example 30 mm in focal plane is 6.5 arcmin = 390 arcsec
MES_ARCSEC_PER_MICRON = 0.013

# This is necessary so that astropy.unit will recognise the plural form
# "microns", which is often used in the SPM FITS headers
u.add_enabled_units([u.def_unit("microns", u.micron)])


def waves2pixels(waves: np.ndarray, w: WCS) -> np.ndarray:
    """
    Convert array of wavelengths (in m) to array indices (integers)

    The implementation is now much simpler than it used to be, thanks
    to the new API calls in astropy.wcs
    """
    return w.spectral.world_to_array_index_values(waves)


def make_vcube():
    ...


def make_vmap(
    vel0: float,
    ra0: float,
    dec0: float,
    dvel: float = 20.0,
    line_id: str = "ha",
    datapath: Path = Path.cwd().parent / "data" / "pvextract",
    shape: tuple[int, int] = (512, 512),
    pixel_scale: float = 0.2,
    slit_width_scale: float = 1.0,
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
        Name of emission line. This is used to glob for the files
        containing the PV spectra.  Default is "ha".
    datapath : `pathlib.Path`, optional
        Path to folder containing PV spectra files.  Default is
        ../data/pvextract with respect to current working directory.
    shape : 2-tuple of (int, int), optional
        Shape of output image array. Default is (512, 512).
    pixel_scale : float, optional
        Linear size in arcsec of each pixel in output array. Default
        is 0.2 arcsec.
    slit_width_scale : float, optional
        Scale factor to multiply the true width of slit. Default is
        1.0
    verbose : bool, default: False

    Returns:
    --------
    `fits.HDUList`
        List of 3 HDU images. The first ("slits") is the sum of all
        the slit brightness times the per-slit weight.  The second
        ("weight") is the sum of the weights.  These two have zero for
        pixels where there is no slit.  The third ("scaled") is the
        first divided by the second, so this is the one that has the
        best estimate of the channel brightness in each pixel. This
        has NaN for pixels where there is no slit.

    Notes
    -----

    Here are some limitations and quirks of the current algorithm.

    1. The transformation to pixels along the wavelength axis yields
    integer values.  This could produce inaccuracies and aliasing when
    `dvel` is less than or comparable to the original pixel size
    along the spectral axis of the PV image. One way of getting round
    this would be to subgrid-interpolate the spectral axis first, but
    this is not yet implemented.

    2. When each slit pixel is added to the output image, a square
    image pixel is used. This means that increasing the width of the
    slit will also produce a corresponding smoothing along the length
    of the slit.  This could even be considered as a positive feature
    ...

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

        # If the header does not specify the slit width, assume it is 150 um
        aperture = spechdu.header.get("APERTURE", "150 micron")
        # Use astropy.unit to parse the string
        slit_width_micron = u.Unit(aperture).to(u.micron)

        # Allow adjustment of the true slit width using the slit_width_scale parameter
        slit_pix_width = (
            MES_ARCSEC_PER_MICRON * slit_width_micron * slit_width_scale / pixel_scale
        )

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


def convert_pv_offset_vels(
    ra0: float,
    dec0: float,
    line_id: str = "ha",
    datapath: Path = Path.cwd().parent / "data" / "pvextract",
    outfolder: str = "pv-offset-vels",
    verbose: bool = False,
) -> None:
    """Convert PV spectra to heliocentric velocities, offset from star, and PA

    Based on the script turtle-pv-relative.py from Turtle project
    """

    # List of 2D spectra that we will process
    speclist = datapath.glob(f"*-{line_id}.fits")
    # Coordinates of central star. Radii and position angles will be
    # calculated from here.
    c0 = SkyCoord(ra0, dec0, unit="deg")
    # Write the new PV spectra to a sibling folder
    newpath = datapath.parent / outfolder
    # Make sure that path to output folder exists
    newpath.mkdir(exist_ok=True)

    for fn in speclist:
        if verbose:
            print("Processing", fn)
        (spechdu,) = fits.open(fn)
        wspec = WCS(spechdu.header, key="A")

        # Rest wavelength in m
        wav0 = wspec.wcs.restwav

        # Eliminate degenerate 3rd dimension from data array and trim off bad bits
        spec2d = spechdu.data[0]

        # Convert to heliocentric velocity
        [[wav1, _, _], [wav2, _, _]] = wspec.all_pix2world([[0, 0, 0], [1, 0, 0]], 0)
        [v1, v2] = waves2vels(np.array([wav1, wav2]), wav0, spechdu.header, usewcs="A")

        # sequence of pixels along the slit spatial axis
        ipixels = np.arange(wspec.array_shape[1])
        # sky coordinate that corresponds to each pixel
        coords = pixel_to_skycoord(ipixels, 0, wcs=wspec)
        # separation of each pixel from star
        radii = c0.separation(coords).to(u.arcsec).value
        # reference pixel has minimum separation from star
        iref = radii.argmin()
        cref = coords[iref]
        # slit offset is minimum of radii
        offset = radii.min()
        # but sign depends on PA
        pa_offset = c0.position_angle(cref)
        if np.sin(pa_offset) < 0.0:
            offset *= -1.0

        # separation from reference pixel gives coordinate along slit
        s = cref.separation(coords).to(u.arcsec).value
        # inter-pixel separation
        ds = np.abs(np.diff(s)).mean()
        # PA of slit
        pa = cref.position_angle(coords[-1])
        if np.cos(pa) < 0.0:
            # Make sure positive offset is to N
            ds *= -1.0
            # And flip slit PA to compensate
            pa += np.pi * u.rad

        wnew = WCS(naxis=2)
        wnew.wcs.ctype = ["VHEL", "LINEAR"]
        wnew.wcs.crpix = [1, iref + 1]
        wnew.wcs.cunit = ["km/s", "arcsec"]
        wnew.wcs.crval = [v1.to("km/s").value, 0.0]
        wnew.wcs.cdelt = [(v2 - v1).to("km/s").value, ds]

        newhdr = wnew.to_header()
        newhdr["PA"] = pa.to(u.deg).value, "Position angle of slit, degrees"
        newhdr["OFFSET"] = offset, "Perpendicular offset of slit from star, arcsec"
        newhdr["WEIGHT"] = spechdu.header["WEIGHT"]

        newsuffix = f"-PA{int(pa.to(u.deg).value)%360:03d}-sep{int(offset):+04d}"
        new_fn = newpath / (fn.stem + newsuffix + ".fits")
        if verbose:
            print("Writing", new_fn)
        fits.PrimaryHDU(data=spec2d, header=newhdr).writeto(new_fn, overwrite=True)
