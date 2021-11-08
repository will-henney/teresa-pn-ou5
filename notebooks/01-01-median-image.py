# -*- coding: utf-8 -*-
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

# # Deal with the image+slit files

from pathlib import Path
import warnings
import yaml
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from mes_longslit import regrid_images

# ## Regrid the images to a common frame, centered on the source

datapath = Path.cwd().parent / "data"
inpath =  datapath / "wcs"
outpath = datapath / "regrid"
outpath.mkdir(exist_ok=True)

# We can use the database file that I wrote, so we can separate the Ha from O III exposures.

db = yaml.safe_load(
    (datapath / "image-slit-database.yml").read_text()
)
db

ha_list = [_["image_id"] for _ in db if _.get("line_id") == "Ha"]
o3_list = [_["image_id"] for _ in db if _.get("line_id") == "O III"]

# These are the coordinates of `PN Ou 5` [according to SIMBAD](http://simbad.u-strasbg.fr/simbad/sim-id?Ident=PN+Ou+5)

c0 = SkyCoord("21 14 20.03 +43 41 36.0", unit=(u.hourangle, u.deg))
c0

# We only need a small image around the center, since it is a small nebula.  We take 400 pixel square with 0.2 arcsec per pixel, so 80 arcsec along each side. The smallest pixels in the original data are 0.35 arcsec, so we are oversampling enough to hopefully avoid any aliasing issues.

ha_dict = {}
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for _id in ha_list:
        infile = inpath / f"cr{_id}_b-wcs.fits"
        hdulist = fits.open(infile)
        ha_dict[_id] = regrid_images(
            hdulist, 
            center=c0, 
            pixscale=0.2 * u.arcsec,
            shape=(400, 400),
        )    

ha_dict

o3_dict = {}
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for _id in o3_list:
        infile = inpath / f"cr{_id}_b-wcs.fits"
        hdulist = fits.open(infile)
        o3_dict[_id] = regrid_images(
            hdulist, 
            center=c0, 
            pixscale=0.2 * u.arcsec,
            shape=(400, 400),
        )  
o3_dict

# Look at one of the results to make sure things are OK

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("talk")

hdu = ha_dict["N20011"][0]
w = WCS(hdu.header)
fig, ax = plt.subplots(
    figsize=(4, 4),
    subplot_kw=dict(projection=w)
)
ax.imshow(hdu.data, cmap="gray_r", origin="lower", vmin=-5, vmax=12)
ax.set(
    xlabel="RA",
    ylabel="DEC",
)

# ## Normalize the images

# Define regions for normalizing the image.  Each window is 50x50 pixels
# * `peak_window` is the central portion
# * `bg_windows` is a list of the 4 corners

# +
peak_window = slice(170, 230), slice(170, 230)
bg_windows = [
    (slice(None, 50), slice(None, 50)),
    (slice(-50, None), slice(None, 50)),
    (slice(None, 50), slice(-50, None)),
    (slice(-50, None), slice(-50, None)),    
]

def normalize(im):
    """Subtract background and divide by peak"""
    bg = np.mean(
        [np.median(im[window]) for window in bg_windows]
    )
    im_new = im - bg
    peak = np.median(im_new[peak_window])
    im_new /= peak
    return im_new


# -

# Apply the normalization to the Ha and [O III] images

for hdulist in ha_dict.values():
    hdulist[0].data = normalize(hdulist[0].data)

for hdulist in o3_dict.values():
    hdulist[0].data = normalize(hdulist[0].data)

# Make median images:

ha_median = np.median(
    np.stack(
        [_[0].data for _ in ha_dict.values()],
    ),
    axis=0,
)
o3_median = np.median(
    np.stack(
        [_[0].data for _ in o3_dict.values()],
    ),
    axis=0,
)

# Look at all the Ha images

fig, axes = plt.subplots(
    3, 4, sharex=True, sharey=True,
    figsize=(12, 9),
)
for ax, (name, hdulist) in zip(axes.flat, ha_dict.items()):
    hdu = hdulist[0]
    ax.imshow(hdu.data, cmap="gray_r", origin="lower", vmin=-0.2, vmax=3.0)
    ax.set_title(name)
axes[-1, -1].imshow(ha_median, cmap="gray_r", origin="lower", vmin=-0.2, vmax=3.0)
axes[-1, -1].set_title("Median", fontweight="bold")
fig.suptitle("PN Ou 5 – Ha exposures")
fig.tight_layout()
fig.savefig("ha-image-slit-mosaic.pdf")

# They all look good.  The slit is clearly narrower in the first 3 images.
#
# Now write out the normalized images:

# +
for name, hdulist in ha_dict.items():
    savefile = outpath / f"ha-imslit-{name}.fits"
    hdulist.writeto(savefile, overwrite=True)

# And the median image
savefile = outpath / f"ha-imslit-median.fits"
fits.PrimaryHDU(
    header=hdulist[0].header,
    data=ha_median,
).writeto(savefile, overwrite=True)
# -

# Repeat for [O III]

fig, axes = plt.subplots(
    2, 3, sharex=True, sharey=True,
    figsize=(9, 6),
)
for ax, (name, hdulist) in zip(axes.flat, o3_dict.items()):
    hdu = hdulist[0]
    ax.imshow(normalize(hdu.data), cmap="gray_r", origin="lower", vmin=-0.2, vmax=3.0)
    ax.set_title(name)
axes[-1, -1].imshow(o3_median, cmap="gray_r", origin="lower", vmin=-0.2, vmax=3.0)
axes[-1, -1].set_title("Median", fontweight="bold")
fig.suptitle("PN Ou 5 – [O III] exposures")
fig.tight_layout()
fig.savefig("o3-image-slit-mosaic.pdf")

# +
for name, hdulist in o3_dict.items():
    savefile = outpath / f"o3-imslit-{name}.fits"
    hdulist.writeto(savefile, overwrite=True)
    
# And the median image
savefile = outpath / f"o3-imslit-median.fits"
fits.PrimaryHDU(
    header=hdulist[0].header,
    data=o3_median,
).writeto(savefile, overwrite=True)
# -




