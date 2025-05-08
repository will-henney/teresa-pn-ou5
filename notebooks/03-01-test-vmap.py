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
import cmasher as cmr

import mes_longslit as mes
# -

np.seterr(all="ignore")

sns.set_context("talk")

c0 = SkyCoord("21 14 20.03 +43 41 36.0", unit=(u.hourangle, u.deg))
c0.ra.deg, c0.dec.deg

# +
fig, axes = plt.subplots(
    1, 6, 
    figsize=(15, 8), 
    sharex=True,
    sharey=True,
    #subplot_kw=dict(projection=w)
)

for v0, ax in zip([-80, -60, -40, -20, 0, 20], axes):

    hdulist = mes.make_vmap(
        v0, c0.ra.deg, c0.dec.deg, slit_width_scale=2.5, dvel=20.0, line_id="oiii",
    )
    hdu = hdulist["scaled"]
    im = ax.imshow(
        hdu.data[:, 200:-200], 
        norm=PowerNorm(gamma=0.5, vmin=-4.0, vmax=800.0), 
        cmap="magma_r",
        origin="lower",
    )
    ax.set_title(fr"${v0:+d}$ km/s")
fig.suptitle("PN Ou 5 - [O III]")
fig.colorbar(im, ax=axes)
# -

fig = plt.figure(figsize=(15, 8),)
axes = []
for j, v0 in enumerate([-80, -60, -40, -20, 0, 20]):
    hdulist = mes.make_vmap(
        v0, c0.ra.deg, c0.dec.deg, slit_width_scale=2.5, dvel=20.0, line_id="ha",
    )
    hdu = hdulist["scaled"]
    ax = fig.add_subplot(1, 6, j + 1, 
                         sharex=axes[0] if axes else None,
                         sharey=axes[0] if axes else None,
                         # projection=WCS(hdu.header),
                        )
    axes.append(ax)
    im = ax.imshow(
        hdu.data[:, 200:-200], 
        norm=PowerNorm(gamma=0.5, vmin=-4.0, vmax=800), 
        cmap="magma_r",
        origin="lower",
    )
    if j > 0:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_title(fr"${v0:+d}$ km/s")
fig.suptitle("PN Ou 5 - Ha")
fig.colorbar(im, ax=axes)

# It looks like there might be a flux calibration problem with one of the slits, which is the second from the left. 

# +
fig, axes = plt.subplots(
    1, 6, 
    figsize=(15, 8), 
    sharex=True,
    sharey=True,
    #subplot_kw=dict(projection=w)
)

for v0, ax in zip([-80, -60, -40, -20, 0, 20], axes):

    hdulist = mes.make_vmap(
        v0, c0.ra.deg, c0.dec.deg, slit_width_scale=2.5, dvel=20.0, line_id="heii",
    )
    hdu = hdulist["scaled"]
    im = ax.imshow(
        hdu.data[:, 200:-200], 
        norm=PowerNorm(gamma=0.5, vmin=-0.2, vmax=40.0), 
        cmap="magma_r",
        origin="lower",
    )
    ax.set_title(fr"${v0:+d}$ km/s")
fig.suptitle("PN Ou 5 - He II")
fig.colorbar(im, ax=axes)

# +
fig, axes = plt.subplots(
    1, 6, 
    figsize=(15, 8), 
    sharex=True,
    sharey=True,
    #subplot_kw=dict(projection=w)
)

for v0, ax in zip([-80, -60, -40, -20, 0, 20], axes):

    hdulist = mes.make_vmap(
        v0, c0.ra.deg, c0.dec.deg, slit_width_scale=2.5, dvel=20.0, line_id="nii",
    )
    hdu = hdulist["scaled"]
    im = ax.imshow(
        hdu.data[:, 200:-200], 
        norm=PowerNorm(gamma=0.5, vmin=-0.5, vmax=100.0), 
        cmap="magma_r",
        origin="lower",
    )
    ax.set_title(fr"${v0:+d}$ km/s")
fig.suptitle("PN Ou 5 - [N II]")
fig.colorbar(im, ax=axes)

# +
fig, axes = plt.subplots(
    1, 6, 
    figsize=(15, 8), 
    sharex=True,
    sharey=True,
    #subplot_kw=dict(projection=w)
)

for v0, ax in zip([-80, -60, -40, -20, 0, 20], axes):

    hdulist = mes.make_vmap(
        v0, c0.ra.deg, c0.dec.deg, slit_width_scale=2.5, dvel=20.0, line_id="hei",
    )
    hdu = hdulist["scaled"]
    im = ax.imshow(
        hdu.data[:, 200:-200], 
        norm=PowerNorm(gamma=0.5, vmin=-1, vmax=10), 
        cmap="magma_r",
        origin="lower",
    )
    ax.set_title(fr"${v0:+d}$ km/s")
fig.suptitle("PN Ou 5 - He I")
fig.colorbar(im, ax=axes)
# -

# ## Fine velocity slices of 6 km/s

vsys = -33
dv = 10
vels = vsys + (dv // 2) + np.arange(-50, 50, dv)
nv = len(vels)
vels[:nv//2] = vels[nv//2 - 1::-1]
vels, len(vels)

# Make one hdu just to set up the wcs
#

# +
hdulist = mes.make_vmap(
    vsys, c0.ra.deg, c0.dec.deg, slit_width_scale=2.5, dvel=dv, line_id="oiii",
)
hdu = hdulist["scaled"]

w = WCS(hdu.header)
w
# -

trimy, trimx = 80, 210
pixscale = 3600 * w.wcs.cdelt[1]

d_ra, d_dec = c0.spherical_offsets_to(w.pixel_to_world(trimx, trimy))
ww, hh = d_ra.to_value(u.arcsec), d_dec.to_value(u.arcsec)
ww, hh

extent = [ww, -ww, hh, -hh]

# +
fig, axes = plt.subplots(
    2, len(vels)//2, 
    figsize=(10, 10), 
    sharex=True,
    sharey=True,
)

for v0, ax in zip(vels, axes.flat):

    hdulist = mes.make_vmap(
        v0, c0.ra.deg, c0.dec.deg, slit_width_scale=2.5, dvel=dv, line_id="oiii",
    )
    hdu = hdulist["scaled"]
    hdu.data -= np.nanmedian(hdu.data)
    im = ax.imshow(
        hdu.data[80:-80, 210:-210], 
        norm=PowerNorm(gamma=0.5, vmin=-0.2, vmax=300.0), 
        cmap=cmr.rainforest_r,
        extent=extent,
        origin="lower",
    )
    ax.axhline(0.0, linestyle="dashed", color="k", linewidth=1)
    ax.axvline(0.0, linestyle="dashed", color="k", linewidth=1)
    ax.set_title(fr"$V_\odot = {v0:+d}$", fontsize="small")
#fig.suptitle("PN Ou 5 - [O III] 5007")
fig.colorbar(im, ax=axes)
fig.savefig("pn-ou5-isovel-oiii.pdf");

# +
fig, axes = plt.subplots(
    2, len(vels)//2, 
    figsize=(10, 10), 
    sharex=True,
    sharey=True,
    #subplot_kw=dict(projection=w)
)

for v0, ax in zip(vels, axes.flat):

    hdulist = mes.make_vmap(
        v0, c0.ra.deg, c0.dec.deg, slit_width_scale=2.5, dvel=5.0, line_id="ha",
    )
    hdu = hdulist["scaled"]
    hdu.data -= np.nanmedian(hdu.data, axis=0, keepdims=True)
    im = ax.imshow(
        hdu.data[80:-80, 210:-210], 
        norm=PowerNorm(gamma=0.5, vmin=-0.2, vmax=300.0), 
        cmap=cmr.sunburst_r,
        origin="lower",
        extent=extent,
    )
    ax.axhline(0.0, linestyle="dashed", color="k", linewidth=1)
    ax.axvline(0.0, linestyle="dashed", color="k", linewidth=1)
    ax.set_title(fr"$V_\odot = {v0:+d}$", fontsize="small")
# fig.suptitle("PN Ou 5 - Ha 6563")
fig.colorbar(im, ax=axes)
fig.savefig("pn-ou5-isovel-ha.pdf");

# +
fig, axes = plt.subplots(
    2, len(vels)//2, 
    figsize=(10, 10), 
    sharex=True,
    sharey=True,
    #subplot_kw=dict(projection=w)
)

for v0, ax in zip(vels, axes.flat):

    hdulist = mes.make_vmap(
        v0, c0.ra.deg, c0.dec.deg, slit_width_scale=2.0, dvel=5.0, line_id="heii",
    )
    hdu = hdulist["scaled"]
    hdu.data -= np.nanmedian(hdu.data)
    im = ax.imshow(
        hdu.data[80:-80, 210:-210], 
        norm=PowerNorm(gamma=0.5, vmin=-0.01, vmax=10.0), 
        cmap=cmr.flamingo_r,
        origin="lower",
        extent=extent,
    )
    ax.set_title(fr"$V_\odot = {v0:+d}$", fontsize="small")
    ax.axhline(0.0, linestyle="dashed", color="k", linewidth=1)
    ax.axvline(0.0, linestyle="dashed", color="k", linewidth=1)

# fig.suptitle("PN Ou 5 - He II 6560")
fig.colorbar(im, ax=axes)
fig.savefig("pn-ou5-isovel-heii.pdf");

# +
fig, axes = plt.subplots(
    2, 12, 
    figsize=(15, 7), 
    sharex=True,
    sharey=True,
    #subplot_kw=dict(projection=w)
)

for v0, ax in zip(vels, axes.flat):

    hdulist = mes.make_vmap(
        v0, c0.ra.deg, c0.dec.deg, slit_width_scale=2.0, dvel=5.0, line_id="nii",
    )
    hdu = hdulist["scaled"]
    hdu.data -= np.nanmedian(hdu.data, axis=0, keepdims=True)
    im = ax.imshow(
        hdu.data[80:-80, 210:-210], 
        norm=PowerNorm(gamma=0.5, vmin=-0.1, vmax=8.0), 
        cmap=cmr.voltage_r,
        origin="lower",
    )
    ax.set_title(fr"${v0:+d}$")
fig.suptitle("PN Ou 5 - [N II] 6583")
fig.colorbar(im, ax=axes)
fig.savefig("pn-ou5-isovel-nii.jpg");

# +
fig, axes = plt.subplots(
    2, 12, 
    figsize=(15, 7), 
    sharex=True,
    sharey=True,
    #subplot_kw=dict(projection=w)
)

for v0, ax in zip(vels, axes.flat):

    hdulist = mes.make_vmap(
        v0, c0.ra.deg, c0.dec.deg, slit_width_scale=2.0, dvel=5.0, line_id="nii_s",
    )
    hdu = hdulist["scaled"]
    hdu.data -= np.nanmedian(hdu.data, axis=0, keepdims=True)
    im = ax.imshow(
        hdu.data[80:-80, 210:-210], 
        norm=PowerNorm(gamma=0.5, vmin=-0.05, vmax=4.0), 
        cmap=cmr.voltage_r,
        origin="lower",
    )
    ax.set_title(fr"${v0:+d}$")
fig.suptitle("PN Ou 5 - [N II] 6548")
fig.colorbar(im, ax=axes)
fig.savefig("pn-ou5-isovel-nii_s.jpg");

# +
fig, axes = plt.subplots(
    2, 12, 
    figsize=(15, 7), 
    sharex=True,
    sharey=True,
    #subplot_kw=dict(projection=w)
)

for v0, ax in zip(vels, axes.flat):

    hdulist = mes.make_vmap(
        v0, c0.ra.deg, c0.dec.deg, slit_width_scale=2.0, dvel=5.0, line_id="hei",
    )
    hdu = hdulist["scaled"]
    hdu.data -= np.nanmedian(hdu.data, axis=0, keepdims=True)
    im = ax.imshow(
        hdu.data[80:-80, 210:-210], 
        norm=PowerNorm(gamma=0.5, vmin=-0.05, vmax=4.0), 
        cmap=cmr.horizon_r,
        origin="lower",
    )
    ax.set_title(fr"${v0:+d}$")
fig.suptitle("PN Ou 5 - He I 5016")
fig.colorbar(im, ax=axes)
fig.savefig("pn-ou5-isovel-hei.jpg");
# -


