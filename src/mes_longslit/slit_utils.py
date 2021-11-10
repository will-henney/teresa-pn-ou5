"""
Working with position-velocity 2D spectra from MES spectrograph

HISTORY:

2015-09 - Original version written for Alba Orion West project: alba-orion-west.org, 

2016-08 - Modified for Teresa Owl project: teresa-owl.org

2019-05 - Modified and extended for Teresa Turtle project: teresa-turtle.org, turtle_utils.py

2021-11 - Ported to this file. Added some type hints. 
          Added comments from teresa-turtle.org
"""

from pathlib import Path
import numpy as np
import numpy.typing as npt
from numpy.polynomial import Chebyshev
from collections.abc import Sequence, Iterable, Mapping, MutableMapping
from typing import Union
from astropy.io import fits  # type: ignore
from astropy.wcs import WCS  # type: ignore
from astropy.coordinates import SkyCoord  # type: ignore
from astropy.wcs.utils import pixel_to_skycoord  # type: ignore
from astropy import units as u  # type: ignore
from astropy.constants import c  # type: ignore
from astropy.modeling import models, fitting  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

VERBOSE = 0

HDU = Union[fits.PrimaryHDU, fits.CompImageHDU, fits.ImageHDU]


def slit_profile(
    ra: Sequence[float], dec: Sequence[float], image: np.ndarray, wcs: WCS
) -> np.ndarray:
    """
    Find the image intensity for a list of positions (ra and dec)

    * This uses nearest pixel algorithm - no interpolation at all
    * Slit pixels outside of the image are set to NaN
    """
    xi, yj = wcs.all_world2pix(ra, dec, 0)
    # Find nearest integer pixel
    ii, jj = np.floor(xi + 0.5).astype(int), np.floor(yj + 0.5).astype(int)
    if VERBOSE > 0:
        print(ra[::100], dec[::100])
        print(ii[::100], jj[::100])
    ny, nx = image.shape
    return np.array(
        [
            image[j, i] if (0 < i < nx and 0 < j < ny) else np.nan
            for i, j in list(zip(ii, jj))
        ]
    )


def slit_profile_circle(
    ra: Sequence[float], dec: Sequence[float], image: np.ndarray, wcs: WCS, r: float
) -> np.ndarray:
    """
    Find the image intensity for a list of positions (ra and dec)

    * This takes average intensity of all pixels within a distance `r`
    (in arcsec) of each position
    """
    # Coordinates of all pixels in the image
    ny, nx = image.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    c = wcs.pixel_to_world(X, Y)
    ns = len(ra)
    profile = np.zeros(shape=(ns,), dtype=float)
    for j, [_ra, _dec] in enumerate(zip(ra, dec)):
        c0 = SkyCoord(_ra, _dec, unit=u.deg)
        sep = c0.separation(c)
        m = sep <= r * u.arcsec
        profile[j] = np.mean(image[m])
    return profile


def find_slit_coords(db: Mapping, hdr: fits.Header, shdr: fits.Header) -> dict:
    """Find the coordinates of all the pixels along a spectrograph slit

    Input arguments are a dict-like 'db' of hand-measured values (must
    contain 'wa', 'ij', islit' and 'shift') and a FITS headers 'hdr' from
    the image+slit exposure and 'shdr' from a spectrum exposure

    Returns a dict of 'ds' (slit pixel scale), 'PA' (slit position
    angle), 'RA' (array of RA values in degrees along slit), 'Dec'
    (array of Dec values in degrees along slit)

    """

    # Decide on axis order for both spectrum and image. Note that
    # values of 'wa' and 'ij' give the axis that is perpendicular to
    # the slit length (wavelength or position, respectively). Hence we
    # subtract from 3 to get the slit length axis
    jstring_i = str(3 - db["ij"])  # which image (I+S) axis lies along slit
    jstring_s = str(3 - db["wa"])  # which spec (pv) axis lies along slit

    dRA_arcsec = hdr["CD1_" + jstring_i] * 3600 * np.cos(np.radians(hdr["CRVAL2"]))
    dDEC_arcsec = hdr["CD2_" + jstring_i] * 3600
    ds = np.hypot(dRA_arcsec, dDEC_arcsec)
    PA = np.degrees(np.arctan2(dRA_arcsec, dDEC_arcsec)) % 360.0

    # Deal with parameters that depend on orientation of the PV image
    if jstring_s == "1":
        # PV slit has spatial axis horizontal in IMAGE coords
        ns = shdr["NAXIS1"]
        # Mezcal has used two different ways of specifying on-chip binning
        try:
            # Older way
            spec_binning = shdr["CBIN"]
        except KeyError:
            try:
                # Newer way
                spec_binning = shdr["CCDXBIN"]
            except KeyError:
                # And the very old data don't have it at all
                spec_binning = 1
    elif jstring_s == "2":
        # PV slit has spatial axis vertical in IMAGE coords
        ns = shdr["NAXIS2"]
        try:
            spec_binning = shdr["RBIN"]
        except KeyError:
            try:
                spec_binning = shdr["CCDYBIN"]
            except KeyError:
                spec_binning = 1
    else:
        raise ValueError("PV slit axis (3 - wa) must be 1 or 2")

    # Pixel coords of each slit pixel on image (in 0-based convention)
    # Deal with parameters that depend on orientation of the I+S image
    if jstring_i == "1":
        # Slit is horizontal in IMAGE coords
        iarr = np.arange(ns) - float(db["shift"])
        jarr = np.ones(ns) * (float(db["islit"]) - 1)
        # Mezcal has used two different ways of specifying on-chip binning
        try:
            # Older way
            image_binning = hdr["CBIN"]
        except KeyError:
            try:
                # Newer way
                image_binning = hdr["CCDXBIN"]
            except KeyError:
                # And the very old data don't have it at all
                image_binning = 1
        # correct for difference in binning between the image+slit and the spectrum
        iarr *= spec_binning / image_binning
    elif jstring_i == "2":
        # Slit is vertical in IMAGE coords
        iarr = np.ones(ns) * (float(db["islit"]) - 1)
        jarr = np.arange(ns) - float(db["shift"])
        try:
            image_binning = hdr["RBIN"]
        except KeyError:
            try:
                image_binning = hdr["CCDYBIN"]
            except KeyError:
                image_binning = 1
        jarr *= spec_binning / image_binning
    else:
        raise ValueError("I+S slit axis (3 - ij) must be 1 or 2")

    if db["s"] < 0:
        # Slit pixel axis has opposite sense in I+S and PV
        iarr = iarr[::-1]
        jarr = jarr[::-1]

    print("iarr =", iarr[::100], "jarr =", jarr[::100])
    # Also correct the nominal slit plate scale
    ds *= spec_binning / image_binning

    # Convert to world coords, using the native frame
    w = WCS(hdr)
    coords = w.pixel_to_world(iarr, jarr)
    print("coords =", coords[::100])
    print("Binning along slit: image =", image_binning, "spectrum =", spec_binning)
    # Make sure to return the coords in the ICRS frame
    return {
        "ds": ds,
        "PA": PA,
        "RA": coords.icrs.ra.value,
        "Dec": coords.icrs.dec.value,
    }


def subtract_sky(
    data: np.ndarray,
    db: Mapping,
    degree: int = 1,
) -> np.ndarray:
    """Remove sky from spectra using explicit windows.
    Sky portions of the slit are given in `db["sky_windows"]`
    We fit a `degree` order polynomial at each wavelength in
    these windows to give the sky estimate, which we subtract.
    """
    # convert axis notation from FITS to python convention
    wav_axis = 2 - db["wa"]
    slit_axis = 1 - wav_axis
    # Array of points along the slit length
    s = np.arange(data.shape[slit_axis], dtype=float)
    # We only fit the sky_windows
    m = np.zeros_like(s, dtype=bool)
    for j1, j2 in db["sky_windows"]:
        m[j1 - 1 : j2] = True
    # Take a copy of the original array, which we will then modify
    newdata = data.copy()
    # If necessary, transpose the data so that axis 0 is wavelength
    if wav_axis == 1:
        newdata = newdata.T
    # Fit and remove the continuum from each row
    for row in newdata:
        p = Chebyshev.fit(s[m], row[m], degree)
        bg_row = p(s)
        row -= bg_row
    # Undo the transpose if necessary
    if wav_axis == 1:
        newdata = newdata.T
    return newdata


def subtract_sky_and_trim(
    data: np.ndarray,
    db: Mapping,
    trim: int = 3,
    margin: int = 10,
    degree: int = 1,
) -> np.ndarray:
    """Remove sky and extraneous border pixels.
    Assume that pixels within `trim` of edge might be bad, and use
    average sky within margin of edge in spatial direction to define
    the bg, fitting a `degree` order polynomial at each wavelength
    """
    # convert axis notation from FITS to python convention
    wav_axis = 2 - db["wa"]
    slit_axis = 1 - wav_axis
    # Array of points along the slit length
    s = np.arange(data.shape[slit_axis], dtype=float)
    # We only fit between trim and margin at each end, so set mask appropriately
    m = np.zeros_like(s, dtype=bool)
    m[trim:margin] = True
    m[-margin:-trim] = True
    # Take a copy of the original array, which we will then modify
    newdata = data.copy()
    # If necessary, transpose the data so that axis 0 is wavelength
    if wav_axis == 1:
        newdata = newdata.T
    # Fit and remove the continuum from each row
    for row in newdata:
        p = Chebyshev.fit(s[m], row[m], degree)
        bg_row = p(s)
        row -= bg_row
    # Undo the transpose if necessary
    if wav_axis == 1:
        newdata = newdata.T

    # And only then can we clean up the trim zone
    newdata[:trim, :] = 0.0
    newdata[-trim:, :] = 0.0
    newdata[:, :trim] = 0.0
    newdata[:, -trim:] = 0.0
    return newdata


def extract_full_profile_from_pv(
    spec_hdu: HDU,
    wavaxis: int,
    bandwidth: Union[float, None],
    linedict: Union[Mapping[str, float], None],
) -> np.ndarray:
    """Get profile along slit of PV image, summed over wavelength

    Optionally add extra continuum since the filter is roughly twice
    as wide as can fit in the PV spectrum, for which we need:

    - `bandwidth` full filter width in angstrom
    - `linedict` rest wavelengths of the lines so we know what to avoid
    """
    assert wavaxis in [1, 2]  # wavaxis is in FITS convention
    w = WCS(spec_hdu.header)
    if wavaxis == 1:
        nwav = spec_hdu.header["NAXIS1"]
        im = spec_hdu.data[:, :]
        wavs, _ = w.all_pix2world(np.arange(nwav), [0], 0)
    else:
        nwav = spec_hdu.header["NAXIS2"]
        im = spec_hdu.data[:, :].T
        _, wavs = w.all_pix2world([0], np.arange(nwav), 0)

    # im should have wavelength as last axis (python convention)
    assert nwav == im.shape[-1]
    full_profile = im.sum(axis=-1)

    if bandwidth is not None and linedict is not None:
        wavmask = np.ones((nwav,)).astype(bool)
        # remove from continuum mask +/- 150 km/s around each line
        for lineid, wav0 in linedict.items():
            vels = 3e5 * (wavs - wav0) / wav0
            wavmask = wavmask & (np.abs(vels) > 150.0)

        # broadcast to 2 dimensions
        imwts = np.ones_like(im) * wavmask[None, :]
        av_cont_profile = np.average(im, weights=imwts, axis=-1)
        # find how much extra continuum to add
        dwav = abs(wavs[1] - wavs[0])
        pv_bw = abs(wavs[-1] - wavs[0])
        missing_cont_profile = av_cont_profile * (bandwidth - pv_bw) / dwav
        # Add to the profile summed over the PV bandwidth
        full_profile += missing_cont_profile
    return full_profile


def extract_slit_profile_from_imslit(
    data: np.ndarray, db: Mapping, slit_width: int = 1
) -> np.ndarray:
    print(db["islit"])
    i1, i2 = int(db["islit"]) - slit_width, int(db["islit"]) + slit_width
    if db["ij"] == 1:
        return data[:, i1:i2].sum(axis=1)
    elif db["ij"] == 2:
        return data[i1:i2, :].sum(axis=0)
    else:
        raise ValueError("ij must be 1 or 2")


def wavs2slice(wavs: Sequence[float], wcs: WCS, db: Mapping) -> slice:
    """
    Convert a wavelength interval `wavs` (length-2 sequence) to a
    slice of the relevant axis`
    """
    assert len(wavs) == 2
    isT = db["wa"] == 2
    if isT:
        _, xpixels = wcs.all_world2pix([0, 0], wavs, 0)
    else:
        xpixels, _ = wcs.all_world2pix(wavs, [0, 0], 0)
    print("Wav:", wavs, "Pixel:", xpixels)
    i1, i2 = np.maximum(0, (xpixels + 0.5).astype(int))
    return slice(min(i1, i2), max(i1, i2))


def extract_line_and_regularize(
    data: np.ndarray,
    wcs: WCS,
    wavrest: float,
    db: Mapping,
    dw: float = 10.0,
    dwbg_in: float = 7.0,
    dwbg_out: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transpose data if necessary, and then subtract off the continuum
    (blue and red of line, inner width `dwbg_in`, outer width
    `dwbg_out`) and restrict to window (width `dw`) around line
    center.  Returns cont-subtracted PV array (2d), cont array (2d),
    and wavs array (1d)
    """
    isT = db["wa"] == 2
    # Make sure array axis order is (position, wavelength)
    if isT:
        data = data.T
        nwav = wcs.pixel_shape[1]
        _, wavs = wcs.all_pix2world([0], np.arange(nwav), 0)
    else:
        nwav = wcs.pixel_shape[0]
        wavs, _ = wcs.all_pix2world(np.arange(nwav), [0], 0)

    # pixel limits for blue, red bg extraction
    bslice = wavs2slice([wavrest - dwbg_out / 2, wavrest - dwbg_in / 2], wcs, db)
    rslice = wavs2slice([wavrest + dwbg_in / 2, wavrest + dwbg_out / 2], wcs, db)
    # extract backgrounds on blue and red sides
    bgblu = np.nanmean(data[:, bslice], axis=1)
    bgred = np.nanmean(data[:, rslice], axis=1)
    # take weighted average, accounting for cases where the bg region
    # does not fit in the image
    weight_blu = data[:, bslice].size
    weight_red = data[:, rslice].size
    print("Background weights:", weight_blu, weight_red)
    if weight_blu and weight_red:
        bg = (bgblu * weight_blu + bgred * weight_red) / (weight_blu + weight_red)
    elif weight_blu:
        bg = bgblu
    elif weight_red:
        bg = bgred
    else:
        raise ValueError("No valid red or blue BG found")

    # pixel limits for entire window
    wslice = wavs2slice([wavrest - dw / 2, wavrest + dw / 2], wcs, db)
    # restrict to just this window
    data = data[:, wslice]
    # and actually subtract the continuum
    bgdata = np.zeros_like(data)
    bgdata += bg[:, None]

    return data - bgdata, bgdata, wavs[wslice]


def make_slit_wcs(
    db: Mapping, slit_coords: Mapping, wavs: npt.NDArray[np.floating], j0: int
) -> tuple[WCS, WCS]:
    """Make WCS for the PV images

    Return two versions, one complex and one simple"""
    #
    # First, wavelength axis, which is easy
    #
    dwav = wavs[1] - wavs[0]
    wav0 = wavs[0]
    wavpix0 = 1

    #
    # Second, find the displacement scale and ref point from the slit_coords
    #
    # The slit_coords should already be in ICRS frame
    c = SkyCoord(slit_coords["RA"], slit_coords["Dec"], unit=u.deg)
    # Find vector of separations between adjacent pixels
    seps = c[:-1].separation(c[1:])
    # Ditto for the position angles
    PAs = c[:-1].position_angle(c[1:])
    # Check that they are all the same as the first one
    assert np.allclose(seps / seps[0], 1.0)
    # assert(np.allclose(PAs/PAs[0], 1.0, rtol=1.e-4))
    # Then use the first one as the slit pixel size and PA
    ds, PA, PA_deg = seps[0].deg, PAs.mean().rad, PAs.mean().deg
    # And for the reference values too
    RA0, Dec0 = c[0].ra.deg, c[0].dec.deg

    #
    # Now make a new shiny output WCS, constructed from scratch
    #
    w = WCS(naxis=3)

    # Make use of all the values that we calculated above
    w.wcs.crpix = [wavpix0, 1, 1]
    w.wcs.cdelt = [dwav, ds, ds]
    w.wcs.crval = [wav0, RA0, Dec0]
    # PC order is i_j = [[1_1, 1_2, 1_3], [2_1, 2_2, 2_3], [3_1, 3_2, 3_3]]
    w.wcs.pc = [
        [1.0, 0.0, 0.0],
        [0.0, np.sin(PA), -np.cos(PA)],
        [0.0, np.cos(PA), np.sin(PA)],
    ]

    #
    # Finally add in auxillary info
    #
    w.wcs.radesys = "ICRS"
    w.wcs.ctype = ["AWAV", "RA---TAN", "DEC--TAN"]
    w.wcs.specsys = "TOPOCENT"
    w.wcs.cunit = [u.Angstrom, u.deg, u.deg]
    w.wcs.name = "TopoWav"
    w.wcs.cname = ["Observed air wavelength", "Right Ascension", "Declination"]

    # Check the new pixel values
    npix = len(slit_coords["RA"])
    check_coords = pixel_to_skycoord(np.arange(npix), [0] * npix, w, 0)
    # These should be the same as the ICRS coords in slit_coords
    print("New coords:", check_coords[::100])
    print("Displacements in arcsec:", check_coords.separation(c).arcsec[::100])
    # 15 Sep 2015: They seem to be equal to within about 1e-2 arcsec

    #
    # And a simple version with slit offsets in arcsec
    #
    w2 = WCS(naxis=2)
    w2.wcs.crpix = [wavpix0, j0 + 1]
    w2.wcs.cdelt = [dwav, ds]
    w2.wcs.crval = [wav0, 0.0]
    w2.wcs.ctype = ["LINEAR", "LINEAR"]

    return w, w2


def fit_cheb(
    x: np.ndarray, y: np.ndarray, npoly: int = 3, mask: Union[np.ndarray, None] = None
) -> np.ndarray:
    """Fits a Chebyshev poly to y(x) and returns fitted y-values"""
    fitter = fitting.LinearLSQFitter()
    p_init = models.Chebyshev1D(npoly, domain=[x.min(), x.max()])
    if mask is None:
        mask = np.ones_like(x).astype(bool)
    p = fitter(p_init, x[mask], y[mask])
    if VERBOSE > 0:
        print(p)
    return p(x)


def make_three_plots(
    spec: np.ndarray,
    calib: np.ndarray,
    prefix: str,
    slit_points: Union[np.ndarray, None] = None,
    neighbors: Union[Mapping, None] = None,
    db: Union[Mapping, None] = None,
    sdb: Union[Mapping, None] = None,
    linelabel: str = "H$\alpha$",
    return_fig: bool = False,
    fig_fmt: str = "pdf",
) -> None:
    """Make diagnostic plots for slit normalization and alignment"""
    assert spec.shape == calib.shape
    fig, axes = plt.subplots(3, 1)

    if slit_points is None:
        ypix = np.arange(len(calib))
        xlabel = "Slit pixel"
        xlim = None
    else:
        ypix = slit_points
        xlabel = "Slit position, arcsec"
        xlim = -80, 80

    xlim = xlim or (ypix.min(), ypix.max())

    # vmax = np.percentile(calib, 95) + 2*calib.std()
    vmax = 20.0
    vmin = -0.01
    ratio = spec / calib

    alpha = 0.8

    # First, plot two profiles against each other to check for zero-point offsets
    # axes[0].plot(calib, spec/ratio_fit, '.', alpha=alpha)
    axes[0].plot(calib, spec, ".", alpha=alpha)
    axes[0].plot([vmin, vmax], [vmin, vmax], "-", alpha=alpha)
    axes[0].set_xlim(vmin, vmax)
    axes[0].set_ylim(vmin, vmax)
    axes[0].set_xlabel("Calibration Image")
    axes[0].set_ylabel("Uncorrected Integrated Spectrum")
    axes[0].set_xscale("symlog", linthresh=0.01)
    axes[0].set_yscale("symlog", linthresh=0.01)

    # Second, plot each against slit pixel to check spatial offset
    axes[1].plot(ypix, spec, alpha=alpha, lw=1, label="Integrated Spectrum")
    axes[2].plot(
        ypix, spec / np.nanmax(calib), alpha=alpha, lw=1, label="Integrated Spectrum"
    )
    axes[1].plot(ypix, calib, alpha=alpha, label="Calibration Image")
    axes[2].plot(ypix, calib / np.nanmax(calib), alpha=alpha, label="Calibration Image")
    if neighbors is not None:
        for nb, calib_nb in neighbors.items():
            lw = 0.5 + 0.1 * nb
            label = f"Slit $\Delta x = {nb:+1d}$"
            axes[1].plot(
                ypix, calib_nb, alpha=0.3 * alpha, lw=lw, color="k", label=label
            )
            axes[2].plot(
                ypix,
                calib_nb / np.nanmax(calib_nb),
                alpha=0.3 * alpha,
                lw=lw,
                color="k",
                label=label,
            )

    axes[1].set_xlim(*xlim)
    axes[1].set_ylim(vmin, vmax)
    axes[1].legend(fontsize="xx-small", loc="upper right")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Profile (absolute log scale)")
    axes[1].set_yscale("symlog", linthresh=0.01)

    axes[2].set_xlim(-40, 40)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_xlabel(xlabel)
    # axes[2].set_ylabel('Ratio: Spec / Calib')
    axes[2].set_ylabel("Profile (relative linear scale)")

    info = ""
    if db is not None:
        # Add some info to the graphs
        info += fr"{db['line_id']}" + "\n"
        info += f"PV: {db['slit_id']}" + "\n"
        info += f"I+S: {db['image_id']}" + "\n"
    if sdb is not None:
        info += (
            fr"Slit PA = ${sdb['PA']:.1f}^\circ$, ds = {sdb['ds']:.2f} arcsec" + "\n"
        )
    if info:
        axes[0].text(
            0.95,
            0.05,
            info,
            fontsize="small",
            ha="right",
            va="bottom",
            transform=axes[0].transAxes,
        )
    fig.set_size_inches(5, 8)
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.savefig(f"{prefix}.{fig_fmt}", dpi=300)

    if return_fig:
        # This option suitable for single use in notebook
        return fig
    else:
        # Default option to clean up, suitable for use in loop over many spectra
        plt.close(fig)
        return None


def trim_edges(
    image: np.ndarray, edges: Sequence[int], fill_value: float = np.nan
) -> np.ndarray:
    lx, ux, ly, uy = edges
    if ly > 0:
        image[:ly, :] = fill_value
    if uy < 0:
        image[uy:, :] = fill_value
    if lx > 0:
        image[:, :lx] = fill_value
    if ux < 0:
        image[:, ux:] = fill_value
    return image


def pv_extract(
    spec_hdu: HDU,
    im_hdu: HDU,
    photom_hdu: HDU,
    db: dict,
    restwavs: Mapping[str, float],
    pvpath: Path,
    neighbors: Sequence[int] = [-4, -2, 2, 4],
) -> None:
    """
    For a single spectrum, apply all steps to get calibrated line PV diagrams
    """

    if "trim" in db:
        # Replace non-linear part with NaNs
        spec_hdu.data = trim_edges(spec_hdu.data, db["trim"])
        # Fit and remove the linear trend along slit
        pvmed = np.nanmedian(spec_hdu.data, axis=2 - db["wa"])
        s = np.arange(len(pvmed))
        pvmed0 = np.nanmedian(pvmed)
        sig = np.nanstd(pvmed)
        m = np.abs(pvmed - pvmed0) <= db.get("bg_sig", 1) * sig
        p = Chebyshev.fit(s[m], pvmed[m], db.get("bg_deg", 1))
        if db["wa"] == 1:
            spec_hdu.data -= p(s)[:, None]
        else:
            spec_hdu.data -= p(s)[None, :]
        # And replace the NaNs with the median value
        spec_hdu.data = trim_edges(
            spec_hdu.data, db["trim"], np.nanmedian(spec_hdu.data)
        )

    # And take a mean-over-wavelength of the spectral slit
    # profile. This is so that we can use its median as a zero point
    # to subtract from the PV array
    spec_profile = np.nanmean(spec_hdu.data, axis=2 - db["wa"])
    imslit_profile = extract_slit_profile_from_imslit(im_hdu.data, db, slit_width=2)

    # Now subtract the median from each profile so that the cross correlation works
    spec_zero_point = np.median(spec_profile)
    spec_profile -= spec_zero_point
    imslit_profile -= np.median(imslit_profile)

    # We need to find the alignment along the slit.
    # New method uses cross-correlation
    ns = len(spec_profile)
    assert len(imslit_profile) == ns, "Incompatible lengths. Maybe different binning?"
    # The above assert would fail if the binning were different
    # between the spectrum and the im+slit, which is something I will
    # have to deal with later.

    # An array of pixel offsets that matches the result of `np.correlate` in "full" mode:
    jshifts = np.arange(-(ns - 1), ns)
    # Now calculate the correlation:
    xcorr = np.correlate(spec_profile, imslit_profile, mode="full")

    if "shift_range" in db:
        shmin, shmax = db["shift_range"]
        mm = (np.abs(jshifts) >= shmin) & (np.abs(jshifts) <= shmax)
        jshifts = jshifts[mm]
        xcorr = xcorr[mm]

    # Use the peak in xcorr to define the shift between the two profiles
    db["shift"] = jshifts[xcorr.argmax()]

    print("Shift along slit:", db["shift"])

    # The other thing we need is the approximate peak of the spectral profile
    jslit = np.arange(ns)
    jwin_slice = slice(*db["jwin"])
    # Take weighted average to define peak position
    j0_s = np.average(jslit[jwin_slice], weights=spec_profile[jwin_slice])

    # Now we have the shift, we can find the celestial coords along the slit in the spectrum
    slit_coords = find_slit_coords(db, im_hdu.header, spec_hdu.header)
    # And then find the profile along the calibration image
    wphot = WCS(photom_hdu.header)
    calib_profile = slit_profile(
        slit_coords["RA"],
        slit_coords["Dec"],
        photom_hdu.data,
        wphot,
    )

    # We have the ability to look at the profiles at neighboring slit
    # positions in the calibration image. This allows us to see if we
    # might have an error in the `islit` value:
    nb_calib_profiles = {}
    for nb in neighbors:
        nbdb = db.copy()
        nbdb["islit"] += nb
        nb_slit_coords = find_slit_coords(nbdb, im_hdu.header, spec_hdu.header)
        nb_calib_profiles[nb] = slit_profile(
            nb_slit_coords["RA"], nb_slit_coords["Dec"], photom_hdu.data, wphot
        )

    # Now estimate the ratio between the spectrum brightness and the calibration image
    jslice0 = slice(int(j0_s) - 20, int(j0_s) + 20)
    spec_sum = np.nansum(spec_profile[jslice0])
    calib_sum = np.nansum(calib_profile[jslice0])
    print("Normalizations:", spec_sum, calib_sum)
    print(spec_profile[jslice0])
    print(calib_profile[jslice0])

    rat0 = spec_sum / calib_sum
    # And normalize the spectral profile to match the calibration image
    spec_profile /= rat0

    # Make the diagnostic plot that shows the calibration
    slit_points = (np.arange(len(spec_profile)) - j0_s) * slit_coords["ds"]
    figpath = Path.cwd().parent / "figs"
    figpath.mkdir(exist_ok=True)
    plt_prefix = str(figpath / f"{db['slit_id']}-calib")
    make_three_plots(
        spec_profile,
        calib_profile,
        plt_prefix,
        slit_points=slit_points,
        neighbors=nb_calib_profiles,
        db=db,
        sdb=slit_coords,
        return_fig=False,
    )

    # Subtract zero point and flux-normalize the whole PV array
    spec_hdu.data -= spec_zero_point
    spec_hdu.data /= rat0
    # Write out the flux-calibrated spectra
    # The default header has minimal changes from the original
    pvheader = fits.Header(spec_hdu.header, copy=True)
    for lineid, wav0 in restwavs.items():
        pvdata, contdata, wavs = extract_line_and_regularize(
            spec_hdu.data, WCS(spec_hdu.header), wav0, db
        )
        pvdata = pvdata[None, :, :]
        contdata = contdata[None, :, :]

        # Create a fancy WCS object for slit coordinates (and a simple one too)
        wslit, wsimp = make_slit_wcs(db, slit_coords, wavs, j0_s)
        # Set the rest wavelength for this line
        wslit.wcs.restwav = (wav0 * u.Angstrom).to(u.m).value
        pvheader.update(wsimp.to_header())
        pvheader.update(wslit.to_header(key="A"))
        pvheader["WEIGHT"] = rat0

        pvfile = str(pvpath / f"{db['slit_id']}-{lineid}.fits")
        fits.PrimaryHDU(header=pvheader, data=pvdata).writeto(pvfile, overwrite=True)

        pvcfile = pvfile.replace(".fits", "-cont.fits")
        fits.PrimaryHDU(header=pvheader, data=contdata).writeto(pvcfile, overwrite=True)
