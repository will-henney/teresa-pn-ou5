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

# +
from pathlib import Path
import yaml
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
import seaborn as sns

import mes_longslit as mes
# -

sns.set_context("talk")

c0 = SkyCoord("21 14 20.03 +43 41 36.0", unit=(u.hourangle, u.deg))
c0.ra.deg, c0.dec.deg

# + tags=[]
fig, axes = plt.subplots(
    1, 6, 
    figsize=(15, 10), 
    sharex=True,
    sharey=True,
    #subplot_kw=dict(projection=w)
)

for v0, ax in zip([-80, -60, -40, -20, 0, 20], axes):

    hdulist = mes.make_vmap(
        v0, c0.ra.deg, c0.dec.deg, slit_width=2.0, dvel=20.0, line_id="oiii",
    )
    hdu = hdulist["scaled"]
    im = ax.imshow(
        hdu.data[:, 200:-200], 
        norm=PowerNorm(gamma=0.5, vmin=-4.0, vmax=1000.0), 
        cmap="gray_r",
        origin="lower",
    )
fig.colorbar(im, ax=axes)
# -


