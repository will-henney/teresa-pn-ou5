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

# # Co-add spectra along nebular axis

# +
from pathlib import Path
import yaml
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.convolution import Gaussian2DKernel, convolve_fft
from matplotlib import pyplot as plt
import seaborn as sns

import mes_longslit as mes
# -

dpath = Path.cwd().parent / "data"
pvpath = dpath / "pvextract"

slit_db_list = yaml.safe_load(
    (dpath / "slits-ha.yml").read_text()
)

# Find list of all ha slits that are close to axis. Use ones that mention "center" in their comments.

id_list = [_["slit_id"] for _ in slit_db_list if "center" in _["comment"]]

id_list

_id = id_list[0]

hdu, = fits.open(pvpath / f"{_id}-ha.fits")

w = WCS(hdu.header, key="A")
w

w.wcs.restwav

hdu.header["WEIGHT"]

w0 = WCS(hdu.header).sub(2)
w0

fig, ax = plt.subplots(
    figsize=(5, 12),
    subplot_kw=dict(projection=w0)
)
ax.imshow(hdu.data[0])
ax.set(
#    ylim=[-10, 10],
)

T = ax.get_transform("world")

T.transform_point([6563.0, 0])

c0 = SkyCoord("21 14 20.03 +43 41 36.0", unit=(u.hourangle, u.deg))
c0.ra.deg, c0.dec.deg

mes.convert_pv_offset_vels(c0.ra.deg, c0.dec.deg, line_id="ha", verbose=True)

mes.convert_pv_offset_vels(c0.ra.deg, c0.dec.deg, line_id="oiii", verbose=True)

# + tags=[]
mes.convert_pv_offset_vels(c0.ra.deg, c0.dec.deg, line_id="heii", verbose=True)

# + tags=[]
mes.convert_pv_offset_vels(c0.ra.deg, c0.dec.deg, line_id="nii", verbose=True)
# -

pvpath = dpath / "pv-offset-vels"


def fkey(filepath):
    "Extract integer slit offset for sorting"
    stem = str(filepath.stem).replace("-regrid", "")
    return int(stem[-4:])


file_list = sorted(pvpath.glob("*-oiii-*.fits"), key=fkey)
[_.stem for _ in file_list]

id_list

# Look at a single slit

hdu, = fits.open(file_list[3])

w = WCS(hdu.header)
w

hdu.header["OFFSET"], hdu.header["WEIGHT"]

# +
fig, ax = plt.subplots(
    figsize=(5, 12),
    subplot_kw=dict(projection=w)
)

vsys = -33
v1, v2 = vsys - 80, vsys + 80
s1, s2 = -30, 30
xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])

y1, y2 = [int(_) for _ in ylims]
bg1 = np.median(hdu.data[y1-10:y1+10], axis=0)
bg2 = np.median(hdu.data[y2-10:y2+10], axis=0)
im = hdu.data - 0.5 * (bg1 + bg2)
ax.imshow(im)
ax.set(
    xlim=xlims,
    ylim=ylims,
)
# -

# ## Look at all the [O III] slits

# +
file_list = sorted(pvpath.glob("*-oiii-*.fits"), key=fkey)

N = len(file_list)
ncols = 4
nrows = (N // ncols) + 1
fig = plt.figure(figsize=(4 * ncols, 8 * nrows))

vsys = -33
v1, v2 = vsys - 100, vsys + 100
s1, s2 = -35, 35

kernel = Gaussian2DKernel(x_stddev=2.0)
for i, filepath in enumerate(file_list):
    hdu, = fits.open(filepath)
    w = WCS(hdu.header)
    ax = plt.subplot(nrows, ncols, i + 1, projection=w)
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    y1, y2 = [int(_) for _ in ylims]
    bg1 = np.median(hdu.data[y1-10:y1+10], axis=0)
    bg2 = np.median(hdu.data[y2-10:y2+10], axis=0)
    im = hdu.data - 0.5 * (bg1 + bg2)
    im /= im.max()
    ax.imshow(im, vmin=-0.1, vmax=1.0)
    ims = convolve_fft(im, kernel)
    ax.contour(
        ims, 
        levels=[0.005, 0.01, 0.02, 0.04, 0.08], 
        colors="w",
        linewidths=[0.5, 1.0, 1.5, 2.0, 2.5],
    )

    ax.set(xlim=xlims, ylim=ylims)
    ax.set_title(filepath.stem)
...;
# -

# Regrid all the oiii slits and save to files

pvpath2 = pvpath.parent / "pv-common"
pvpath2.mkdir(exist_ok=True)

for filepath in file_list:
    hdu, = fits.open(filepath)
    hdu2 = mes.regrid_pv(hdu)
    filepath2 = pvpath2 / f"{filepath.stem}-regrid.fits" 
    hdu2.writeto(filepath2, overwrite=True)


# Take a look at the new regridded PV spectra

# +
file_list = sorted(pvpath2.glob("*-oiii-*.fits"), key=fkey)

N = len(file_list)
ncols = 4
nrows = (N // ncols) + 1
fig = plt.figure(figsize=(4 * ncols, 5 * nrows))

vsys = -33
v1, v2 = vsys - 100, vsys + 100
s1, s2 = -35, 35

kernel = Gaussian2DKernel(x_stddev=4.0)
for i, filepath in enumerate(file_list):
    hdu, = fits.open(filepath)
    w = WCS(hdu.header)
    ax = plt.subplot(nrows, ncols, i + 1, projection=w)
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    y1, y2 = [int(_) for _ in ylims]
    bg1 = np.median(hdu.data[y1-10:y1+10], axis=0)
    bg2 = np.median(hdu.data[y2-10:y2+10], axis=0)
    im = hdu.data - 0.5 * (bg1 + bg2)
    im /= im.max()
    ax.imshow(im, vmin=-0.1, vmax=1.0)
    ims = convolve_fft(im, kernel)
    ax.contour(
        ims, 
        levels=[0.005, 0.01, 0.02, 0.04, 0.08], 
        colors="w",
        linewidths=[0.5, 1.0, 1.5, 2.0, 2.5],
    )

    ax.set(xlim=xlims, ylim=ylims)
    ax.set_title(filepath.stem + f"\nweight = {hdu.header['WEIGHT']:.3f}")
...;
# -

# This looks good, except for the noise, which has gone a bit weird.  Now we choose the best spectra tho combine. 
#
# Looks like first 4

# +
selected = 0, 1, 2, 3
imlist = []
weightsum = 0.0

for isel in selected:
    hdu, = fits.open(file_list[isel])
    weight =  hdu.header["WEIGHT"]
    imlist.append(weight * hdu.data)
    weightsum += weight
    
hdu.data = np.sum(np.stack(imlist, axis=0), axis=0) / weightsum
hdu.writeto(pvpath2 / "oiii-pv-coadd.fits", overwrite=True)



# -





# ## Repeat for all the Ha slits

# + tags=[]
file_list = sorted(pvpath.glob("*-ha-*.fits"), key=fkey)

N = len(file_list)
ncols = 4
nrows = (N // ncols) + 1
fig = plt.figure(figsize=(4 * ncols, 8 * nrows))

vsys = -33
v1, v2 = vsys - 100, vsys + 100
s1, s2 = -35, 35

kernel = Gaussian2DKernel(x_stddev=2.0)
for i, filepath in enumerate(file_list):
    hdu, = fits.open(filepath)
    w = WCS(hdu.header)
    ax = plt.subplot(nrows, ncols, i + 1, projection=w)
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    y1, y2 = [int(_) for _ in ylims]
    bg1 = np.median(hdu.data[y1-10:y1], axis=0)
    bg2 = np.median(hdu.data[y2:y2+10], axis=0)
    im = hdu.data - 0.5 * (bg1 + bg2)
    im /= im.max()
    ax.imshow(im, vmin=-0.1, vmax=1.0)
    ims = convolve_fft(im, kernel)
    ax.contour(
        ims, 
        levels=[0.005, 0.01, 0.02, 0.04, 0.08], 
        colors="w",
        linewidths=[0.5, 1.0, 1.5, 2.0, 2.5],
    )

    ax.set(xlim=xlims, ylim=ylims)
    ax.set_title(filepath.stem)
...;
# -

for filepath in file_list:
    hdu, = fits.open(filepath)
    hdu2 = mes.regrid_pv(hdu)
    filepath2 = pvpath2 / f"{filepath.stem}-regrid.fits" 
    hdu2.writeto(filepath2, overwrite=True)

# +
file_list = sorted(pvpath2.glob("*-ha-*.fits"), key=fkey)

N = len(file_list)
ncols = 4
nrows = (N // ncols) + 1
fig = plt.figure(figsize=(4 * ncols, 5 * nrows))

vsys = -33
v1, v2 = vsys - 100, vsys + 100
s1, s2 = -35, 35

kernel = Gaussian2DKernel(x_stddev=4.0)
for i, filepath in enumerate(file_list):
    hdu, = fits.open(filepath)
    w = WCS(hdu.header)
    ax = plt.subplot(nrows, ncols, i + 1, projection=w)
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    y1, y2 = [int(_) for _ in ylims]
    bg1 = np.median(hdu.data[y1-10:y1], axis=0)
    bg2 = np.median(hdu.data[y2:y2+10], axis=0)
    im = hdu.data - 0.5 * (bg1 + bg2)
    im /= im.max()
    ax.imshow(im, vmin=-0.1, vmax=1.0)
    ims = convolve_fft(im, kernel)
    ax.contour(
        ims, 
        levels=[0.005, 0.01, 0.02, 0.04, 0.08], 
        colors="w",
        linewidths=[0.5, 1.0, 1.5, 2.0, 2.5],
    )

    ax.set(xlim=xlims, ylim=ylims)
    ax.set_title(filepath.stem + f"\nweight = {hdu.header['WEIGHT']:.3f}")
...;

# +
selected = 2, 4, 5, 6, 7, 8, 9
imlist = []
weightsum = 0.0

for isel in selected:
    hdu, = fits.open(file_list[isel])
    weight =  hdu.header["WEIGHT"]
    imlist.append(weight * hdu.data)
    weightsum += weight
    
hdu.data = np.sum(np.stack(imlist, axis=0), axis=0) / weightsum
hdu.writeto(pvpath2 / "ha-pv-coadd.fits", overwrite=True)


# -
# ## Repeat for He II and [N II]

# +
file_list = sorted(pvpath.glob("*-heii-*.fits"), key=fkey)

N = len(file_list)
ncols = 4
nrows = (N // ncols) + 1
fig = plt.figure(figsize=(4 * ncols, 8 * nrows))

vsys = -33
v1, v2 = vsys - 100, vsys + 100
s1, s2 = -35, 35

kernel = Gaussian2DKernel(x_stddev=2.0)
for i, filepath in enumerate(file_list):
    hdu, = fits.open(filepath)
    w = WCS(hdu.header)
    ax = plt.subplot(nrows, ncols, i + 1, projection=w)
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    y1, y2 = [int(_) for _ in ylims]
    bg1 = np.median(hdu.data[y1-10:y1], axis=0)
    bg2 = np.median(hdu.data[y2:y2+10], axis=0)
    im = hdu.data - 0.5 * (bg1 + bg2)
    im /= im.max()
    ax.imshow(im, vmin=-0.1, vmax=1.0)
    ims = convolve_fft(im, kernel)
    ax.contour(
        ims, 
        levels=[0.005, 0.01, 0.02, 0.04, 0.08], 
        colors="w",
        linewidths=[0.5, 1.0, 1.5, 2.0, 2.5],
    )

    ax.set(xlim=xlims, ylim=ylims)
    ax.set_title(filepath.stem)
...;
# -

for filepath in file_list:
    hdu, = fits.open(filepath)
    hdu2 = mes.regrid_pv(hdu)
    filepath2 = pvpath2 / f"{filepath.stem}-regrid.fits" 
    hdu2.writeto(filepath2, overwrite=True)

# +
file_list = sorted(pvpath2.glob("*-heii-*.fits"), key=fkey)

N = len(file_list)
ncols = 4
nrows = (N // ncols) + 1
fig = plt.figure(figsize=(6 * ncols, 8 * nrows))

vsys = -33
v1, v2 = vsys - 100, vsys + 100
s1, s2 = -35, 35

kernel = Gaussian2DKernel(x_stddev=4.0)
for i, filepath in enumerate(file_list):
    hdu, = fits.open(filepath)
    w = WCS(hdu.header)
    ax = plt.subplot(nrows, ncols, i + 1, projection=w)
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    y1, y2 = [int(_) for _ in ylims]
    x1, x2 = [int(_) for _ in xlims]
    x0, y0 = w.world_to_pixel_values(0.0, 0.0)
    x0 = int(x0)
    bg1 = np.median(hdu.data[y1-10:y1], axis=0)
    bg2 = np.median(hdu.data[y2:y2+10], axis=0)
    im = hdu.data - 0.5 * (bg1 + bg2)
    scale = np.percentile(im[y1:y2, x1:x0], 99)
    im /= scale
    ims = convolve_fft(im, kernel)
    ax.imshow(ims, vmin=-0.1, vmax=1.0)
    #ax.contour(
    #    ims, 
    #    levels=[0.005, 0.01, 0.02, 0.04, 0.08], 
    #    colors="w",
    #    linewidths=[0.5, 1.0, 1.5, 2.0, 2.5],
    #)

    ax.set(xlim=xlims, ylim=ylims)
    ax.set_title(filepath.stem + f"\nweight = {hdu.header['WEIGHT']:.3f}")
    ax.text(x1 + 5, y1 + 5, f"{i}", color="w", fontsize="x-large")
...;

# +
selected = 2, 4, 5, 6, 7, 8, 9
imlist = []
weightsum = 0.0

for isel in selected:
    hdu, = fits.open(file_list[isel])
    weight =  hdu.header["WEIGHT"]
    imlist.append(weight * hdu.data)
    weightsum += weight
    
hdu.data = np.sum(np.stack(imlist, axis=0), axis=0) / weightsum
hdu.writeto(pvpath2 / "heii-pv-coadd.fits", overwrite=True)

# +
file_list = sorted(pvpath.glob("*-nii-*.fits"), key=fkey)

N = len(file_list)
ncols = 4
nrows = (N // ncols) + 1
fig = plt.figure(figsize=(4 * ncols, 8 * nrows))

vsys = -33
v1, v2 = vsys - 100, vsys + 100
s1, s2 = -35, 35

kernel = Gaussian2DKernel(x_stddev=2.0)
for i, filepath in enumerate(file_list):
    hdu, = fits.open(filepath)
    w = WCS(hdu.header)
    ax = plt.subplot(nrows, ncols, i + 1, projection=w)
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    y1, y2 = [int(_) for _ in ylims]
    bg1 = np.median(hdu.data[y1-10:y1], axis=0)
    bg2 = np.median(hdu.data[y2:y2+10], axis=0)
    im = hdu.data - 0.5 * (bg1 + bg2)
    im /= im.max()
    ax.imshow(im, vmin=-0.1, vmax=1.0)
    ims = convolve_fft(im, kernel)
#    ax.contour(
#        ims, 
#        levels=[0.005, 0.01, 0.02, 0.04, 0.08], 
#        colors="w",
#        linewidths=[0.5, 1.0, 1.5, 2.0, 2.5],
#    )

    ax.set(xlim=xlims, ylim=ylims)
    ax.set_title(filepath.stem)
...;
# -

for filepath in file_list:
    hdu, = fits.open(filepath)
    hdu2 = mes.regrid_pv(hdu)
    filepath2 = pvpath2 / f"{filepath.stem}-regrid.fits" 
    hdu2.writeto(filepath2, overwrite=True)

# +
file_list = sorted(pvpath2.glob("*-nii-*.fits"), key=fkey)

N = len(file_list)
ncols = 4
nrows = (N // ncols) + 1
fig = plt.figure(figsize=(6 * ncols, 8 * nrows))

vsys = -33
v1, v2 = vsys - 100, vsys + 100
s1, s2 = -35, 35

kernel = Gaussian2DKernel(x_stddev=4.0)
for i, filepath in enumerate(file_list):
    hdu, = fits.open(filepath)
    w = WCS(hdu.header)
    ax = plt.subplot(nrows, ncols, i + 1, projection=w)
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    y1, y2 = [int(_) for _ in ylims]
    x1, x2 = [int(_) for _ in xlims]
    x0, y0 = w.world_to_pixel_values(0.0, 0.0)
    x0 = int(x0)
    bg1 = np.median(hdu.data[y1-10:y1], axis=0)
    bg2 = np.median(hdu.data[y2:y2+10], axis=0)
    im = hdu.data - 0.5 * (bg1 + bg2)
    scale = np.percentile(im[y1:y2, x1:x0], 99.9)
    im /= scale
    ims = convolve_fft(im, kernel)
    ax.imshow(ims, vmin=-0.1, vmax=1.0)
    #ax.contour(
    #    ims, 
    #    levels=[0.005, 0.01, 0.02, 0.04, 0.08], 
    #    colors="w",
    #    linewidths=[0.5, 1.0, 1.5, 2.0, 2.5],
    #)

    ax.set(xlim=xlims, ylim=ylims)
    ax.set_title(filepath.stem + f"\nweight = {hdu.header['WEIGHT']:.3f}")
    ax.text(x1 + 5, y1 + 5, f"{i}", color="w", fontsize="x-large")
...;

# +
selected = 2, 4, 5, 6, 7, 8, 9
imlist = []
weightsum = 0.0

for isel in selected:
    hdu, = fits.open(file_list[isel])
    weight =  hdu.header["WEIGHT"]
    imlist.append(weight * hdu.data)
    weightsum += weight
    
hdu.data = np.sum(np.stack(imlist, axis=0), axis=0) / weightsum
hdu.writeto(pvpath2 / "nii-pv-coadd.fits", overwrite=True)
# -

# ## Look at our co-added spectra


import seaborn as sns
sns.set_context("talk")

# +
file_list = sorted(pvpath2.glob("*-pv-coadd.fits"))

N = len(file_list)
ncols = 2
nrows = (N // ncols) + 1
fig = plt.figure(figsize=(8 * ncols, 10 * nrows))

vsys = -33
v1, v2 = vsys - 100, vsys + 100
s1, s2 = -35, 35

kernel = Gaussian2DKernel(x_stddev=2.0)
for i, filepath in enumerate(file_list):
    hdu, = fits.open(filepath)
    w = WCS(hdu.header)
    ax = plt.subplot(nrows, ncols, i + 1, projection=w)
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    y1, y2 = [int(_) for _ in ylims]
    x1, x2 = [int(_) for _ in xlims]
    x0, y0 = w.world_to_pixel_values(0.0, 0.0)
    x0 = int(x0)
    bg1 = np.median(hdu.data[y1-10:y1], axis=0)
    bg2 = np.median(hdu.data[y2:y2+10], axis=0)
    im = hdu.data - 0.5 * (bg1 + bg2)
    scale = np.percentile(im[y1:y2, x1:x0], 99.99)
    im /= scale
    # Save a FITS file of BG-subtracted and normalized image
    fits.PrimaryHDU(
        header=hdu.header,
        data=im,
    ).writeto(
        str(filepath).replace(".fits", "-bgsub.fits"),
        overwrite=True,
    )
    ax.imshow(im, vmin=-0.1, vmax=1.0)
    ims = convolve_fft(im, kernel)
    ax.contour(
        ims, 
        levels=[0.005, 0.01, 0.02, 0.04, 0.08], 
        colors="w",
        linewidths=[0.5, 1.0, 1.5, 2.0, 2.5],
    )

    ax.set(xlim=xlims, ylim=ylims)
    ax.set_title(filepath.stem + f"\nweight = {hdu.header['WEIGHT']:.3f}")
...;
# -

# ### Some comments on the results
#
# Components
#
# #### Inner lobes
#
# Red component is brighter than blue.
#
# #### Outer lobes
#
# Bend towards blue on both sides (N and S)
#
# #### Equatorial high velocity wings
# More obvious in oiii. 
#
# #### Polar knots
#
#
#

# ## Look at 1D spectra
#
# We can use the bg-subtracted files that we have just saved. Plot the Ha and oiii on the same graph. Show a series of spatial ranges:

# +
file_list = sorted(pvpath2.glob("*-pv-coadd-bgsub.fits"))

linehdus = {
    filepath.stem.split("-")[0]: fits.open(filepath)[0]
    for filepath in file_list
}


positions = (
    ("N knot", [20.0, 30.0]),
    ("N outer", [9.0, 15.0]),
    ("N inner", [3.0, 9.0]),
    ("Core", [-3.0, 3.0]),
    ("S inner", [-9.0, -3.0]),    
    ("S outer", [-15.0, -9.0]),
    ("S knot", [-30.0, -20.0]),
)

nlines = len(file_list)
npos = len(positions)
# -

linehdus

# +
fig, axes = plt.subplots(
    npos, 
    1, 
    sharex=True,
    figsize=(12, 16),
)

vsys = -33
v1, v2 = vsys - 100, vsys + 100

skip = ["heii", "nii"]
for ax, [pos_label, [s1, s2]] in zip(axes, positions):
    for line_label, hdu in linehdus.items():
        w = WCS(hdu.header)
        ns, nv = hdu.data.shape
        xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
        x1, x2 = [int(_) for _ in xlims]
        y1, y2 = [int(_) for _ in ylims]
        spec = hdu.data[y1:y2, x1:x2].mean(axis=0)
        vels, _ = w.pixel_to_world_values(np.arange(nv), [0]*nv)
        vels = vels[x1:x2]
        if line_label == "nii":
            spec[np.abs(vels) <= 10.0] = np.nan
        if line_label == "heii":
            spec[vels >= 30.0] = np.nan
        if line_label in skip and np.abs([s1, s2]).min() > 4:
            spec[:] = np.nan
        ax.plot(vels, spec, label=line_label)
    ax.axhline(0.0, linestyle="dashed", c="k", lw=1,)
    ax.axvline(vsys, linestyle="dashed", c="k", lw=1,)
    axtitle = f"  {pos_label}: [${int(s1):+d}$, ${int(s2):+d}$]"
    ax.set_title(axtitle, loc="left", y=0.7)
axes[3].legend(ncol=2)
axes[-1].set(
    xlabel="Heliocentric velocity, km/s",
)
fig.savefig("ou5-coadd-1dspec-all.pdf")
...;
# -

# That version has all the lines for the brighter parts.  But, to be honest, they do not really add anything. 

from astropy.modeling import models, fitting


def mark_component(model, color, ax):
    v = model.mean.value
    a = model.amplitude.value
    ax.plot([v, v], [0.3 * a, 0.7 * a], lw=3, color=color, alpha=0.7)


# +
fig, axes = plt.subplots(
    npos, 
    1, 
    sharex=True,
    figsize=(12, 16),
)

vsys = -33
v1, v2 = vsys - 100, vsys + 100
fitter = fitting.LevMarLSQFitter()
keep = ["ha", "oiii"]
gfits = {}
for ax, [pos_label, [s1, s2]] in zip(axes, positions):
    for line_label, hdu in linehdus.items():
        if not line_label in keep:
            continue
        w = WCS(hdu.header)
        ns, nv = hdu.data.shape
        xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
        x1, x2 = [int(_) for _ in xlims]
        y1, y2 = [int(_) for _ in ylims]
        spec = hdu.data[y1:y2, x1:x2].mean(axis=0)
        vels, _ = w.pixel_to_world_values(np.arange(nv), [0]*nv)
        vels = vels[x1:x2]
        dataline, = ax.plot(vels, spec, label=line_label)
        c = dataline.get_color()
        
        # Fit two Gaussians
        mask1 = vels < vsys
        mask2 = ~mask1
        sm1 = spec[mask1].max()
        vm1 = vels[spec[mask1].argmax()]
        sm2 = spec[mask2].max()
        vm2 = vels[spec[mask2].argmax() + mask1.sum()]
        g1 = models.Gaussian1D(amplitude=sm1, mean=vm1, stddev=10.0)
        g2 = models.Gaussian1D(amplitude=sm2, mean=vm2, stddev=10.0)
        if "knot" in pos_label:
            init_model = g1
            fac = -0.5
        elif "outer" in pos_label:
            init_model = g1 + g2
            fac = 0.02           
        else:
            init_model = g1 + g2
            fac = 0.3
        fitmask = spec > fac * spec.max()
        fitted_model = fitter(init_model, vels[fitmask], spec[fitmask])
        gfits[(line_label, pos_label)] = fitted_model
#        print(pos_label, line_label)
#        print(init_model)
#        print(fitted_model)
        ax.plot(
            vels, 
            fitted_model(vels), 
            linestyle="dashed", 
            lw=2, 
            c=c,
            label=f"{line_label} fit",
        )
        if "knot" in pos_label:
            mark_component(fitted_model, c, ax)
        else:
            mark_component(fitted_model[0], c, ax)
            mark_component(fitted_model[1], c, ax)

    ax.axhline(0.0, linestyle="dashed", c="k", lw=1,)
    ax.axvline(vsys, linestyle="dashed", c="k", lw=1,)
    axtitle = f"  {pos_label}: [${int(s1):+d}$, ${int(s2):+d}$]"
    ax.set_title(axtitle, loc="left", y=0.7)
axes[3].legend(ncol=2)
axes[-1].set(
    xlabel="Heliocentric velocity, km/s",
)
fig.savefig("ou5-coadd-1dspec-ha-oiii.pdf")
...;
# -

# Now zoom the y axis, to look at the wings, but just for the inner regions:

# + tags=[]
cpos = {
    k: v for k, v in positions
    if np.abs(v).min() < 4
}
cpos


# -

def mark_component_low(model, color, ax):
    v = model.mean.value
    a = model.amplitude.value
    ax.plot([v, v], [0.03, 0.07], lw=3, color=color, alpha=0.7)


from astropy.convolution import convolve_models

# +
fig, axes = plt.subplots(
    len(cpos), 
    1, 
    sharex=True,
    figsize=(12, 16),
)

vsys = -33
v1, v2 = vsys - 100, vsys + 100
fitter = fitting.LevMarLSQFitter()
keep = ["ha", "oiii"]
gfits2 = {}
for ax, [pos_label, [s1, s2]] in zip(axes, cpos.items()):
    for line_label, hdu in linehdus.items():
        if not line_label in keep:
            continue
        w = WCS(hdu.header)
        ns, nv = hdu.data.shape
        xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
        x1, x2 = [int(_) for _ in xlims]
        y1, y2 = [int(_) for _ in ylims]
        spec = hdu.data[y1:y2, x1:x2].mean(axis=0)
        vels, _ = w.pixel_to_world_values(np.arange(nv), [0]*nv)
        vels = vels[x1:x2]
        # Fit two Gaussians
        mask1 = vels < vsys
        mask2 = ~mask1
        sm1 = spec[mask1].max()
        vm1 = vels[spec[mask1].argmax()]
        sm2 = spec[mask2].max()
        vm2 = vels[spec[mask2].argmax() + mask1.sum()]
        g1 = models.Gaussian1D(amplitude=sm1, mean=vm1, stddev=10.0)
        g2 = models.Gaussian1D(amplitude=sm2, mean=vm2, stddev=10.0)
        g1b = models.Gaussian1D(amplitude=0.1 * sm1, mean=vm1, stddev=3.0)
        g2b = models.Gaussian1D(amplitude=0.1 * sm2, mean=vm2, stddev=3.0)

        init_model = g1 + g2 #+ g1b + g2b
        fitmask = spec > 0.3 * spec.max()
        fitted_model = fitter(init_model, vels[fitmask], spec[fitmask])
        #fitted_model = fitter(init_model, vels, spec)
        gfits2[(line_label, pos_label)] = fitted_model



        dataline, = ax.plot(
            vels, 
            spec - fitted_model(vels), 
            label=f"{line_label} residuals",
        )
        c = dataline.get_color()
        ax.fill_between(vels, 0.0, spec, color=c, alpha=0.1, label=f"{line_label} observed")
        ax.plot(
            vels, 
            fitted_model(vels), 
            linestyle="dashed", 
            lw=2, 
            c=c,
            label=f"{line_label} fit",
        )
        for component in fitted_model[:2]:
            mark_component_low(component, c, ax)
        for component in fitted_model:
            ax.plot(
                vels, 
                component(vels), 
                linestyle="dotted", 
                lw=2, 
                c=c,
                alpha=0.5,
            )

           

    ax.axhline(0.0, linestyle="dashed", c="k", lw=1,)
    ax.axvline(vsys, linestyle="dashed", c="k", lw=1,)
    axtitle = f"  {pos_label}: [${int(s1):+d}$, ${int(s2):+d}$]"
    ax.set_title(axtitle, loc="left", y=0.7)
    ax.set(ylim=[-0.04, 0.25])
axes[1].legend(ncol=1)
axes[-1].set(
    xlabel="Heliocentric velocity, km/s",
)
fig.savefig("ou5-coadd-1dspec-wings.pdf")
...;
# -

import pandas as pd

m = gfits[("oiii", "Core")]
dict(zip(m.param_names, m.parameters))

pd.DataFrame(
    {k: dict(zip(m.param_names, m.parameters)) for k, m in gfits.items()}
).T

pd.DataFrame(
    {k: dict(zip(m.param_names, m.parameters)) for k, m in gfits2.items()}
).T

# ## Exploratory material

# Try out reprojection of a PV onto a common grid.  We will test it with FITS_utils since reproject says it cannot handle non-celestial data

import FITS_tools

# +
hdu0, = fits.open(file_list[6])
hdu1, = fits.open(file_list[8])
weight0 = hdu0.header["WEIGHT"]
weight1 = hdu1.header["WEIGHT"]


hdu10 = fits.PrimaryHDU(
    header=hdu0.header,
    data=FITS_tools.hcongrid.hcongrid(hdu1.data, hdu1.header, hdu0.header),
)
hduav = fits.PrimaryHDU(
    header=hdu0.header,
    data=(hdu10.data * weight1 + hdu0.data * weight0) / (weight1 + weight0),
)

# +
ncols, nrows = 3, 1
fig = plt.figure(figsize=(4 * ncols, 8 * nrows))

vsys = -33
v1, v2 = vsys - 100, vsys + 100
s1, s2 = -35, 35

kernel = Gaussian2DKernel(x_stddev=2.0)
for i, hdu in enumerate([hdu0, hdu1, hduav]):
    w = WCS(hdu.header)
    ax = plt.subplot(nrows, ncols, i + 1, projection=w)
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    y1, y2 = [int(_) for _ in ylims]
    bg1 = np.median(hdu.data[y1-10:y1], axis=0)
    bg2 = np.median(hdu.data[y2:y2+10], axis=0)
    im = hdu.data - 0.5 * (bg1 + bg2)
    im /= im.max()
    ax.imshow(im, vmin=-0.1, vmax=1.0)
    ims = convolve_fft(im, kernel)
    ax.contour(
        ims, 
        levels=[0.0025, 0.005, 0.01, 0.02, 0.04, 0.08], 
        colors="w",
        linewidths=[0.3, 0.5, 1.0, 1.5, 2.0, 2.5],
    )

    ax.set(xlim=xlims, ylim=ylims)
...;
# -

# That seems to have worked fine.  Now we will try it with a pre-defined common grid. Say, 1 km/s and 0.2 arcsec.

# +
# FITS_tools.hcongrid.hcongrid_hdu??
# -

WCS(hdu1.header)

# Helium lines. Not very good.

# +
file_list = list(pvpath.glob("*-heii-*.fits"))
N = len(file_list)
ncols = 4
nrows = (N // ncols) + 1
fig = plt.figure(figsize=(4 * ncols, 8 * nrows))

vsys = -33
v1, v2 = vsys - 100, vsys + 100
s1, s2 = -35, 35

kernel = Gaussian2DKernel(x_stddev=2.0)
for i, filepath in enumerate(file_list):
    hdu, = fits.open(filepath)
    w = WCS(hdu.header)
    ax = plt.subplot(nrows, ncols, i + 1, projection=w)
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    y1, y2 = [int(_) for _ in ylims]
    x1, x2 = [int(_) for _ in xlims]
    bg1 = np.median(hdu.data[y1-10:y1], axis=0)
    bg2 = np.median(hdu.data[y2:y2+10], axis=0)
    im = hdu.data - 0.5 * (bg1 + bg2)
    x0, _ = w.world_to_pixel_values(0.0, 0.0)
    x0 = int(x0)
    im /= im[y1:y2, x1:x0].max()
    ax.imshow(im, vmin=-0.1, vmax=1.0)
    ims = convolve_fft(im, kernel)
    ax.contour(
        ims, 
        levels=[0.005, 0.01, 0.02, 0.04, 0.08], 
        colors="w",
        linewidths=[0.5, 1.0, 1.5, 2.0, 2.5],
    )

    ax.set(xlim=xlims, ylim=ylims)
    ax.set_title(filepath.stem)
...;
# -

x1, x2, x0

y1, y2

im[y1:y2, x1:x2].max()


