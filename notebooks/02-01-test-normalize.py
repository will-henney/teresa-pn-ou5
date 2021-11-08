# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Compare slit profile with reference profile
#
# First try it out by hand for a single slit

# +
from pathlib import Path
import yaml
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import pyplot as plt
import seaborn as sns

import mes_longslit as mes

# -

dpath = Path.cwd().parent / "data"

# List of data for each Ha slit exposure:

slit_db_list = yaml.safe_load(
    (dpath / "slits-ha.yml").read_text()
)

# Photometric reference image:

# + tags=[]
photom, = fits.open(dpath / "regrid" / "ha-imslit-median.fits")
wphot = WCS(photom.header)
# -

# To start off with, we will analyze a single slit:

db = slit_db_list[0]
db

# Get the HDUs for both the slit spectrum and the image+slit. The spectrum file names are very variable, so we have an `orig_file` entry in the database:

spec_hdu, = fits.open(dpath / "originals" / db["orig_file"])

# But the image file names are more regular and can be derived from the `image_id` entry:

im_hdu, = fits.open(dpath / "wcs" / f"cr{db['image_id']}_b-wcs.fits")

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

spec_hdu.data = mes.subtract_sky_and_trim(
    spec_hdu.data, db, trim=5, margin=50)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(spec_hdu.data[250:450, 200:700], origin="lower");

# So that does a reasonably good job of subtracting the background emission line

# Lines to avoid when calculating the continuum

restwavs = {'ha': 6562.79, 'nii': 6583.45, 'nii_s': 6548.05}

spec_profile = mes.extract_full_profile_from_pv(
    spec_hdu,
    wavaxis=db["wa"],
    bandwidth=90.0,
    linedict=restwavs,
)

# This is the position of the slit in pixel coordinates.

db["islit"] = 442

imslit_profile = mes.extract_slit_profile_from_imslit(
    im_hdu.data, db, slit_width=2,
)

jslit = np.arange(len(spec_profile))


spec_profile.shape, imslit_profile.shape

# ### Find a better way to do the alignment

jwin_slice = slice(340, 700)
shift_guess = 100
jwin_slice_shift = slice(
    jwin_slice.start - shift_guess, 
    jwin_slice.stop - shift_guess,
)



# +
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(jslit[jwin_slice], (imslit_profile[jwin_slice_shift] + 20) * 10)
ax.plot(jslit[jwin_slice], spec_profile[jwin_slice])
ax.set(yscale="linear", ylim=[0, 2000]);


# -

# We need to find the alignment along the slit.  Just use the initial guess for now.

j0_s = np.average(jslit[jwin_slice], weights=spec_profile[jwin_slice])
j0_i = np.average(jslit[jwin_slice_shift], weights=(20 + imslit_profile[jwin_slice_shift]))
db["shift"] = j0_s - j0_i
j0_s, j0_i, db["shift"]

slit_coords = mes.find_slit_coords(db, im_hdu.header, spec_hdu.header)

slit_coords['Dec'].shape, jslit.shape

calib_profile = mes.slit_profile(
    slit_coords['RA'], 
    slit_coords['Dec'],
    photom.data, 
    wphot,
)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(jslit + db["shift"], (imslit_profile + 20) * 10)
ax.plot(jslit, spec_profile)
ax.plot(jslit, calib_profile * 1000)
ax.set(yscale="linear", ylim=[0, 2000]);

calib_profile[::100]

spec_profile[::100]

slit_points = (np.arange(len(spec_profile)) - j0_s)*slit_coords["ds"]

jslice0 = slice(int(j0_s)-20, int(j0_s)+20)

rat0 = np.nansum(spec_profile[jslice0]) / np.nansum(calib_profile[jslice0])
print('Coarse calibration: ratio =', rat0)

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
    neighbors=None,
    db=db, 
    sdb=slit_coords,
)


