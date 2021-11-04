# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from pathlib import Path
import warnings
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from mes_longslit import regrid_images

inpath = Path.cwd().parent / "data" / "wcs"
outpath = inpath.parent / "regrid"
outpath.mkdir(exist_ok=True)

# These are the coordinates of `PN Ou 5` [according to SIMBAD](http://simbad.u-strasbg.fr/simbad/sim-id?Ident=PN+Ou+5)

c = SkyCoord("21 14 20.03 +43 41 36.0", unit=(u.hourangle, u.deg))
c

results = {}
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for infile in sorted(inpath.glob("*wcs.fits")):
        hdulist = fits.open(infile)
        results[infile.stem] = regrid_images(hdulist, center=c, shape=(256, 256))
    

results

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("talk")

hdu = results["crN20011_b-wcs"][0]
w = WCS(hdu.header)
fig, ax = plt.subplots(
    figsize=(10, 10),
    subplot_kw=dict(projection=w)
)
ax.imshow(hdu.data, cmap="gray_r", vmin=-5, vmax=12)
ax.set(
    xlabel="RA",
    ylabel="DEC",
)

w

tform = ax.get_transform("icrs")

tform.get_matrix()


