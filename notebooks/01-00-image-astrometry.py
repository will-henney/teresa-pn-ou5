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

# We only need a small image around the center, since it is a small nebula.  We take 256 pixel square with 0.2 arcsec per pixel, so 51.2 arcsec along each side. The smallest pixels in the original data are 0.35 arcsec, so we are oversampling enough to hopefully avoid any aliasing issues.

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
            shape=(256, 256),
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
            shape=(256, 256),
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
#    subplot_kw=dict(projection=w)
)
ax.imshow(hdu.data, cmap="gray_r", origin="lower", vmin=-5, vmax=12)
# ax.set(
#     xlabel="RA",
#     ylabel="DEC",
# )

# ## Normalize the images

# Define regions for normalizing the image.  Each window is 50x50 pixels
# * `peak_window` is the central portion
# * `bg_windows` is a list of the 4 corners

# +
peak_window = slice(100, 150), slice(100, 150)
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

fig, axes = plt.subplots(
    3, 4, sharex=True, sharey=True,
    figsize=(12, 9),
)
for ax, (name, hdulist) in zip(axes.flat, ha_dict.items()):
    hdu = hdulist[0]
    ax.imshow(normalize(hdu.data), cmap="gray_r", origin="lower", vmin=-0.2, vmax=2.0)
    ax.set_title(name)
axes[-1, -1].remove()
fig.suptitle("PN Ou 5 – Ha exposures")
fig.tight_layout()
fig.savefig("ha-image-slit-mosaic.pdf")

fig, axes = plt.subplots(
    2, 3, sharex=True, sharey=True,
    figsize=(9, 6),
)
for ax, (name, hdulist) in zip(axes.flat, o3_dict.items()):
    hdu = hdulist[0]
    ax.imshow(normalize(hdu.data), cmap="gray_r", origin="lower", vmin=-0.2, vmax=2.0)
    ax.set_title(name)
axes[-1, -1].remove()
fig.suptitle("PN Ou 5 – [O III] exposures")
fig.tight_layout()
fig.savefig("o3-image-slit-mosaic.pdf")





w

tform = ax.get_transform("icrs")

tform.get_matrix()


