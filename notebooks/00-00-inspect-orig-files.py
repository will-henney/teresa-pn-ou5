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

# # PN Ou 5: Inspect original files

from pathlib import Path
from astropy.io import fits
from astropy.table import Table

dpath = Path("../data/originals/")

# Look and see what sort of files we have:

data = []
kwds = ["MEZMODE", "DATE-OBS", "FILTER", "RA", "DEC", "PA", "CCDTYPE", "CCDSUM"]
for _file in sorted(dpath.glob("*.fits")):
    hdu = fits.open(_file)[0]
    thisdata = {"File": _file.stem}
    for k in kwds:
        thisdata[k] = hdu.header.get(k)
    data.append(thisdata)
tab = Table(rows=data)
tab.show_in_notebook()

# So we have 2017 data with 70 micron slit and 2x2 binning, and then 2018, 2019 data with 150 micron slit and 3x3 binning.

# Select the image+slit or slit+image files that we will need to do astrometry of

m = ["slit" in _ for _ in tab["MEZMODE"]]
tab[m]

list(tab[m]["File"])

# ## Find the HEALpix coordinates of our source

from astropy.coordinates import SkyCoord, ICRS
import astropy.units as u

# All the positions should be about the same, so we just use the first one.

c = SkyCoord(tab[0]["RA"], tab[0]["DEC"], unit=(u.hourangle, u.deg))
c

from astropy_healpix import HEALPix

# In order to find which data files to download from http://data.astrometry.net/5000/, we need to translate the celestial coordinate to HEALpix index numbers:

hp_2 = HEALPix(nside=2, order="nested", frame=ICRS())
hp_1 = HEALPix(nside=1, order="nested", frame=ICRS())

# Levels 0 to 4 use the `nside=2` tiles. 

hp_2.cone_search_skycoord(c, radius=5 * u.arcminute)

# So that means `index500[0-4]-13.fits`

hp_1.cone_search_skycoord(c, radius=5 * u.arcminute)

# So that means `index500[5-7]-03.fits`


