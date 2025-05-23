# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Compare slit profile with reference profile for [O III]
#
# Repeat of previous workbook but for a different line
#
# I first work through all the steps individually, looking at graphs of the intermediate results.  This was used while iterating on the algorithm, by swapping out the value of `db` below for different slits that showed problems.

# ## First try it out by hand for a single slit

# +
from pathlib import Path
import yaml
import numpy as np
from numpy.polynomial import Chebyshev
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from matplotlib import pyplot as plt
import seaborn as sns

import mes_longslit as mes

# -

dpath = Path.cwd().parent / "data"
pvpath = dpath / "pvextract"
pvpath.mkdir(exist_ok=True)

# List of data for each Ha slit exposure:

slit_db_list = yaml.safe_load((dpath / "slits-o3.yml").read_text())

# Photometric reference image:

(photom,) = fits.open(dpath / "regrid" / "ha-imslit-median.fits")
wphot = WCS(photom.header)

# To start off with, we will analyze a single slit. **This is what we change when we want to try a differnt slit**

db = slit_db_list[3]
db

# Get the HDUs for both the slit spectrum and the image+slit. The spectrum file names are very variable, so we have an `orig_file` entry in the database:

spec_hdu = fits.open(dpath / "originals" / db["orig_file"])[0]

# But the image file names are more regular and can be derived from the `image_id` entry:

(im_hdu,) = fits.open(dpath / "wcs" / f"cr{db['image_id']}_b-wcs.fits")

# There is no sign of any saturated pixels in any of the exposures, so we can miss out that step.

# Add in extra fields to database:
#
# - `wa` wavelength axis (1 or 2, fits order) in PV spectrum
# - `ij` slit orientation in I+S (1=vertical, 2=horizontal)

if db["slit_id"].startswith("N"):
    db["wa"] = 2
    db["ij"] = 2
else:
    db["wa"] = 1
    db["ij"] = 1
db["s"] = 1

if "cosmic_rays" in db:
    spec_hdu.data = mes.remove_cosmic_rays(
        spec_hdu.data, db["cosmic_rays"], np.nanmedian(spec_hdu.data),
    )

fig, ax = plt.subplots(figsize=(10, 5))
# ax.imshow(spec_hdu.data[250:700, :],
#          #vmin=-3, vmax=30,
#          origin="lower");
ax.imshow(spec_hdu.data[:, :], vmin=10, vmax=100, origin="lower")

# Try to correct for gradient:

if "trim" in db:
    # Replace non-linear part with NaNs
    spec_hdu.data = mes.trim_edges(spec_hdu.data, db["trim"])
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
    spec_hdu.data = mes.trim_edges(
        spec_hdu.data, db["trim"], np.nanmedian(spec_hdu.data)
    )

    fig, ax = plt.subplots()
    ax.plot(s[m], pvmed[m])
    ax.plot(s, p(s))

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(spec_hdu.data[:, :], vmin=-5, vmax=100, origin="lower")

# So we are no longer attempting to remove the sky at this stage, but we are trying to remove the light leak or whatever it is that adds a bright background at one end of the slit.  This is necessary so that the cross correlation works.

# Lines to avoid when calculating the continuum

restwavs = {"oiii": 5006.84, "hei": 5015.68}

spec_profile = mes.extract_full_profile_from_pv(
    spec_hdu,
    wavaxis=db["wa"],
    bandwidth=None,
    linedict=restwavs,
)

# This is the position of the slit in pixel coordinates.

imslit_profile = mes.extract_slit_profile_from_imslit(
    im_hdu.data,
    db,
    slit_width=2,
)


jslit = np.arange(len(spec_profile))

spec_profile.shape, imslit_profile.shape

spec_profile -= np.median(spec_profile)
imslit_profile -= np.median(imslit_profile)

# ### Find a better way to do the alignment

# I am going to experiment with using cross-correlation to estimate the along-slit offset between `spec_profile` and `imslit_profile`:

ns = len(spec_profile)
assert len(imslit_profile) == ns, "Incompatible lengths"

# The above assert would fail if the binning were different between the spectrum and the im+slit, which is something I will have to deal with later.

# An array of pixel offsets that matches the result of `np.correlate` in "full" mode:

jshifts = np.arange(-(ns - 1), ns)

# Now calculate the correlation:

xcorr = np.correlate(spec_profile, imslit_profile, mode="full")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(jshifts, xcorr)
# ax.set(xlim=[-300, 300]);

# That is a very clean result! One high narrow peak, at an offset of roughly 100, exactly where we expect it to be.

mm = (np.abs(jshifts) < 110) & (np.abs(jshifts) > 5)
jshift_peak = jshifts[mm][xcorr[mm].argmax()]
jshift_peak

# That is much better, at least for this example.

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(jslit + jshift_peak, imslit_profile / np.max(imslit_profile))
ax.plot(jslit, spec_profile / np.max(spec_profile), alpha=0.7)
ax.set(yscale="linear", ylim=[-0.4, 1])
ax.axhline(0)


jwin_slice = slice(*db["jwin"])
jwin_slice_shift = slice(
    jwin_slice.start - jshift_peak,
    jwin_slice.stop - jshift_peak,
)


fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(jslit[jwin_slice], imslit_profile[jwin_slice_shift])
ax.plot(jslit[jwin_slice], 100 * spec_profile[jwin_slice])
ax.set(
    yscale="linear",
    #    ylim=[0, 1000]
)

# We need to find the alignment along the slit.  Just use the initial guess for now.

j0_s = np.average(jslit[jwin_slice], weights=spec_profile[jwin_slice])
j0_i = np.average(
    jslit[jwin_slice_shift], weights=(10 + imslit_profile[jwin_slice_shift])
)
db["shift"] = jshift_peak
j0_s, j0_i, db["shift"]

slit_coords = mes.find_slit_coords(db, im_hdu.header, spec_hdu.header)

slit_coords["Dec"].shape, jslit.shape

calib_profile = mes.slit_profile(
    slit_coords["RA"],
    slit_coords["Dec"],
    photom.data,
    wphot,
    # r = slit_coords["ds"],
)

# Plot the calibration profile (green) compared with the spec and imslit profiles:

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(jslit + db["shift"], imslit_profile / imslit_profile.max())
ax.plot(jslit, spec_profile / spec_profile.max())
ax.plot(jslit, 0.05 * calib_profile)
ax.set(yscale="linear", ylim=[-0.1, 0.1])

# This is now working fine after I fixed the pixel scale in the calibration image.

# We have the ability to look at the profiles at neighboring slit positions in the calibration image. This allows us to see if we might have an error in the `islit` value:

neighbors = [-2, -1, 1, 2]
nb_calib_profiles = {}
for nb in neighbors:
    nbdb = db.copy()
    nbdb["islit"] += nb
    nb_slit_coords = mes.find_slit_coords(nbdb, im_hdu.header, spec_hdu.header)
    nb_calib_profiles[nb] = mes.slit_profile(
        nb_slit_coords["RA"],
        nb_slit_coords["Dec"],
        photom.data,
        wphot,
        # nb_slit_coords["ds"],
    )

fig, ax = plt.subplots(figsize=(12, 4))
# ax.plot(jslit + db["shift"], (imslit_profile + 20) * 10)
# ax.plot(jslit, spec_profile)
ax.plot(jslit, calib_profile, color="k", lw=2)
for nb in neighbors:
    ax.plot(jslit, nb_calib_profiles[nb], label=f"${nb:+d}$")
ax.legend()
ax.set(yscale="linear")


slit_points = (np.arange(len(spec_profile)) - j0_s) * slit_coords["ds"]

jslice0 = slice(int(j0_s) - 20, int(j0_s) + 20)

rat0 = np.nansum(spec_profile[jslice0]) / np.nansum(calib_profile[jslice0])
print("Coarse calibration: ratio =", rat0)

spec_profile[jslice0]

calib_profile[jslice0]

spec_profile /= rat0

figpath = Path.cwd().parent / "figs"
figpath.mkdir(exist_ok=True)

plt_prefix = figpath / f"{db['slit_id']}-calib"
mes.make_three_plots(
    spec_profile,
    calib_profile,
    plt_prefix,
    slit_points=slit_points,
    neighbors=nb_calib_profiles,
    db=db,
    sdb=slit_coords,
    return_fig=True,
);


# ## Now try the automated way

for db in slit_db_list:
    print(db)
    spec_hdu = fits.open(dpath / "originals" / db["orig_file"])[0]
    (im_hdu,) = fits.open(dpath / "wcs" / f"cr{db['image_id']}_b-wcs.fits")
    if db["slit_id"].startswith("N"):
        db["wa"] = 2
        db["ij"] = 2
    else:
        db["wa"] = 1
        db["ij"] = 1
    db["s"] = 1

    mes.pv_extract(
        spec_hdu,
        im_hdu,
        photom,
        db,
        restwavs,
        pvpath,
        neighbors=[-1, 1],
    )


db
