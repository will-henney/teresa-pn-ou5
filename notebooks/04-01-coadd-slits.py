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

# # Co-add spectra along nebular axis

# +
from pathlib import Path
import yaml
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.convolution import Gaussian2DKernel, Gaussian1DKernel, convolve_fft
from astropy.table import Table
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

mes.convert_pv_offset_vels(c0.ra.deg, c0.dec.deg, line_id="heii", verbose=True)

mes.convert_pv_offset_vels(c0.ra.deg, c0.dec.deg, line_id="nii", verbose=True)

mes.convert_pv_offset_vels(c0.ra.deg, c0.dec.deg, line_id="niis", verbose=True)

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

# Look at the offsets

file_list = sorted(pvpath.glob("*-oiii-*.fits"), key=fkey)
data = []
kwds = ["OFFSET", "WEIGHT"]
for _file in file_list:
    hdu = fits.open(_file)[0]
    thisdata = {"File": _file.stem}
    for k in kwds:
        thisdata[k] = hdu.header.get(k)
    data.append(thisdata)
tab = Table(rows=data)
tab

# Regrid all the oiii slits and save to files

pvpath2 = pvpath.parent / "pv-common"
pvpath2.mkdir(exist_ok=True)

for filepath in file_list:
    hdu, = fits.open(filepath)
    # Use output pixels of 2 km/s by 1 arcsec
    hdu2 = mes.regrid_pv(hdu, pixscale=(2.0, 1.0))
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
    ax.imshow(im, vmin=-0.1, vmax=1.0, aspect="auto")
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

# +
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

file_list = sorted(pvpath.glob("*-ha-*.fits"), key=fkey)
data = []
kwds = ["OFFSET", "WEIGHT"]
for _file in file_list:
    hdu = fits.open(_file)[0]
    thisdata = {"File": _file.stem}
    for k in kwds:
        thisdata[k] = hdu.header.get(k)
    data.append(thisdata)
tab = Table(rows=data)
tab

# ### Minor axis spatial profile
#
# Fit gaussian to profile along horizontal slit

w

filepath = file_list[4]
hdu, = fits.open(filepath)
w = WCS(hdu.header)
ny, nx = hdu.data.shape
_, yy = w.array_index_to_world(np.arange(ny), [0] * ny)
yy

_, [i1, i2] = w.world_to_array_index_values([-70, 0], [0, 0])
i1, i2, _

from astropy.modeling import models, fitting

# +
fig, ax = plt.subplots()
profile = np.mean(hdu.data[:, i1:i2], axis=1)
profile -= np.median(profile)
g1 = models.Gaussian1D(amplitude=50, mean=-4, stddev=4)
g2 = models.Gaussian1D(amplitude=70, mean=5, stddev=4)
init_model = g1 + g2
mask = np.abs(yy) < 25 * u.arcsec
fitter = fitting.LevMarLSQFitter()
fitted_model = fitter(init_model, yy[mask], profile[mask])
gg1, gg2 = fitted_model
ax.plot(yy, profile, ds="steps-mid")
ax.plot(yy, fitted_model(yy))
ax.plot(yy, gg1(yy), ls="dashed")
ax.plot(yy, gg2(yy), ls="dashed")

ax.axvline(0)
ax.axvline(gg1.mean.value)
ax.axvline(gg2.mean.value)

ax.set_xlim(25, -25)
# -

# Calculate offset between the two peaks:

gg2.mean - gg1.mean

gg1.mean, gg2.mean

# Width of peaks

gg1.fwhm, gg2.fwhm

gg1.stddev


# #### Fit with thick shell

def shell_bright(x, r, h, f = 0):
    """Projected brightness of cylindrical hollow shell
    
    Parameters:

    x : array of projected positions perpendicular to cylindrical axis
    r : shell outer radius
    h : shell relative thickness
    f : asymmetry
    """
    # line of sight depth through cylinder
    z_out = np.where(
        x**2 <= r**2,
        np.sqrt(r**2 - x**2),
        0.0
    )
    # radius of inner hole
    r_in = r * (1 - h)
    # line of sight depth through inner hole
    z_in = np.where(
        x**2 <= r_in**2,
        np.sqrt(r_in**2 - x**2),
        0.0
    )
    # emissivity is linear function of x for simplicity
    e = np.where(
        x**2 <= r**2,
        1 + f * x / (2 * r),
        0.0,
    )
    return e * (z_out - z_in)



# +
fig, ax = plt.subplots()
rshell, hshell = 9, 0.5
sprofile = 12 * shell_bright(yy.value, rshell, hshell, f=0.5)
kernel = Gaussian1DKernel(stddev=3)
sprofile = convolve_fft(sprofile, kernel)

sg1 = models.Gaussian1D(amplitude=50, mean=-4, stddev=4)
sg2 = models.Gaussian1D(amplitude=70, mean=5, stddev=4)
init_model = sg1 + sg2
mask = np.abs(yy) < 25 * u.arcsec
sfitted_model = fitter(init_model, yy[mask], sprofile[mask])
sgg1, sgg2 = sfitted_model
ax.plot(yy, sprofile, ds="steps-mid")
ax.plot(yy, sfitted_model(yy))
ax.plot(yy, sgg1(yy), ls="dashed")
ax.plot(yy, sgg2(yy), ls="dashed")

ax.axvline(0)
ax.axvline(sgg1.mean.value)
ax.axvline(sgg2.mean.value)

ax.axvspan(hshell * rshell, rshell, alpha=0.1, color="k", zorder=100)
ax.axvspan(-hshell * rshell, -rshell, alpha=0.1, color="k", zorder=100)
ax.set_xlim(25, -25)
# -

sgg2.mean - sgg1.mean

# + jupyter={"source_hidden": true}
sgg1.mean, sgg2.mean

# + jupyter={"source_hidden": true}
sgg1.fwhm, sgg2.fwhm
# -

hshell * rshell / np.mean([sgg1.fwhm.value, sgg2.fwhm.value])

hshell * rshell / sgg2.fwhm.value

2 * (1 - hshell) * rshell / (sgg2.mean - sgg1.mean).value

# So we find that the peak-peak distance is equal to the *inner* diameter of the shell. 
#
# Whereas the shell thickness is about 0.6 times FWHM. 

# Conclusions, the inner lobes have larger diameter than I thought. 
#
# Inner radius 4.3. Thickness 4.5 arcsec, so outer radius about 9 arcsec. 
#
# If we take half-way point, we have 6.8 +/- 2.2 arcsec. 

# We also reproduce the asymmetry in the peak positions with a simple linear gradient for the emissivity. The weaker side is displaced inward by about 10% and the stronger side displaced outward by the same amount

# #### Fit with power-law shell

# +
from scipy import integrate
def shell_bright_powerlaw(x, r_in, a=4, f=0):
    """Projected brightness of cylindrical hollow shell
    
    Parameters:

    x : array of projected positions perpendicular to cylindrical axis
    r_in : shell inner radius
    a : power-law index for emissivity
    f : asymmetry
    """

    # emissivity is power law in radius
    rgrid = np.geomspace(r_in, 10 * r_in, 200)
    egrid = (rgrid/r_in) ** -a

    # integrate to get surface brightness
    bgrid, sbgrid = brightness_discrete(rgrid, egrid)

    # double up grid to include pos and neg sides
    bb = np.concatenate([-bgrid[::-1], bgrid])
    sbb = np.concatenate([sbgrid[::-1], sbgrid])

    # interpolate onto requested grid
    e = np.interp(x, bb, sbb)
    print(e.shape, x.shape)
    # add in a linear asymmetry
    e = e * (1 + f * x / (2 * r_in))
    return e

    
    
    
def brightness_discrete(r, e, n_inner=50, verbose=False, integrator=np.trapz):
    """Perform integral of surface brightness along line of sight

    Suitable values for `integrator` are numpy.trapz or scipy.integrate.simpson
    """

    # Use the Cloudy radial points with additional uniform grid from origin to inner boundary
    b_inner = np.linspace(0.0, r.min(), num=n_inner, endpoint=False)
    b = np.concatenate((b_inner, r))
    
    sb = np.zeros_like(b)
    # For each impact parameter
    for i, _b in enumerate(b):
        # Select all radii greater than impact parameter
        m = r >= _b
        # Array of LOS positions for each of these radii
        z = np.sqrt(r[m]**2 - _b**2)
        _e = e[m]
        # Integrate along z to find brightness
        sb[i] = 2 * integrator(_e, z)
    return b, sb


# +
fig, ax = plt.subplots()
r_in = 5.3
s6profile = 30 * shell_bright_powerlaw(yy.value, r_in, a=6, f=0.35)
kernel = Gaussian1DKernel(stddev=3)
s6profile = convolve_fft(s6profile, kernel)

sg1 = models.Gaussian1D(amplitude=50, mean=-4, stddev=4)
sg2 = models.Gaussian1D(amplitude=70, mean=5, stddev=4)
init_model = sg1 + sg2
mask = np.abs(yy) < 25 * u.arcsec
sfitted_model = fitter(init_model, yy[mask], s6profile[mask])
sgg1, sgg2 = sfitted_model
ax.plot(yy, s6profile, ds="steps-mid")
ax.plot(yy, sfitted_model(yy))
ax.plot(yy, sgg1(yy), ls="dashed")
ax.plot(yy, sgg2(yy), ls="dashed")

ax.axvline(0)
ax.axvline(sgg1.mean.value)
ax.axvline(sgg2.mean.value)

ax.axvspan(r_in, 25, alpha=0.1, color="k", zorder=100)
ax.axvspan(-25, -r_in, alpha=0.1, color="k", zorder=100)
ax.set_xlim(25, -25)
# -

sgg2.mean - sgg1.mean

sgg1.mean, sgg2.mean

sgg1.fwhm, sgg2.fwhm

# +
fig, ax = plt.subplots()
r_in = 4.5
s3profile = 13 * shell_bright_powerlaw(yy.value, r_in, a=3, f=0.2)
kernel = Gaussian1DKernel(stddev=3)
s3profile = convolve_fft(s3profile, kernel)

sg1 = models.Gaussian1D(amplitude=50, mean=-4, stddev=2)
sg2 = models.Gaussian1D(amplitude=70, mean=5, stddev=2)
init_model = sg1 + sg2
mask = np.abs(yy) < 15 * u.arcsec
sfitted_model = fitter(init_model, yy[mask], s3profile[mask])
sgg1, sgg2 = sfitted_model
ax.plot(yy, s3profile, ds="steps-mid")
ax.plot(yy, sfitted_model(yy))
ax.plot(yy, sgg1(yy), ls="dashed")
ax.plot(yy, sgg2(yy), ls="dashed")

ax.axvline(0)
ax.axvline(sgg1.mean.value)
ax.axvline(sgg2.mean.value)

ax.axvspan(r_in, 25, alpha=0.1, color="k", zorder=100)
ax.axvspan(-25, -r_in, alpha=0.1, color="k", zorder=100)
ax.set_xlim(25, -25)
# -

sgg2.mean - sgg1.mean

sgg1.mean, sgg2.mean

sgg1.fwhm, sgg2.fwhm

# So the power law model with a=6 does work ok

# #### Summary of minor axis results
#
# Both the thick shell and the steep power law work tolerably well at fitting the profile along the horizontal slit.

# +
fig, ax = plt.subplots()

ax.fill_between(yy.value, profile, step="mid", alpha=0.2, color="k", zorder=100)
ax.plot(yy, sprofile, label=r"thick shell $r_\mathrm{in} = 4.5$, $r_\mathrm{out} = 9.0$")
ax.plot(yy, 0.8 * s6profile, label=r"power law $n = 6$, $r_\mathrm{in} = 5.5$")
ax.plot(yy, s3profile, label=r"power law $n = 4$, $r_\mathrm{in} = 4.5$")

ax.axvline(0, ls="dashed", color="k", lw=1)
# ax.axvline(gg1.mean.value)
# ax.axvline(gg2.mean.value)
ax.set_xlabel("Displacement along slit")
ax.set_ylabel("Brightness")
ax.legend(fontsize="x-small")
ax.set_xlim(25, -25)
# -

#



# ### Regrid the Ha spectra

for filepath in file_list:
    hdu, = fits.open(filepath)
    hdu2 = mes.regrid_pv(hdu, pixscale=(2.0, 1.0))
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
    ax.imshow(im, vmin=-0.1, vmax=1.0, aspect="auto")
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
    hdu2 = mes.regrid_pv(hdu, pixscale=(2.0, 1.0))
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
    ax.imshow(ims, vmin=-0.1, vmax=1.0, aspect="auto")
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
    bg1 = np.median(hdu.data[y1:y1+30], axis=0)
    bg2 = np.median(hdu.data[y2-30:y2], axis=0)
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
    hdu2 = mes.regrid_pv(hdu, pixscale=(2.0, 1.0))
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
    bg1 = np.median(hdu.data[y1:y1+30], axis=0)
    bg2 = np.median(hdu.data[y2-30:y2], axis=0)
    im = hdu.data - 0.5 * (bg1 + bg2)
    scale = np.percentile(im[y1:y2, x1:x0], 99.9)
    im /= scale
    ims = convolve_fft(im, kernel)
    ax.imshow(ims, vmin=-0.1, vmax=1.0, aspect="auto")
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

# Update this to be more similar to what we do for oiii in the 04-02 notebook. Write the normalized bg-subtracted images first 

# +
file_list = sorted(pvpath2.glob("*-pv-coadd.fits"))

N = len(file_list)

vsys = -33
v1, v2 = vsys - 100, vsys + 100
s1, s2 = -35, 35

contours = "oiii", "ha"
kernel = Gaussian2DKernel(x_stddev=2.0)
for i, filepath in enumerate(file_list):
    hdu, = fits.open(filepath)
    w = WCS(hdu.header)
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    y1, y2 = [int(_) for _ in ylims]
    x1, x2 = [int(_) for _ in xlims]
    x0, y0 = w.world_to_pixel_values(0.0, 0.0)
    x0 = int(x0)
    if filepath.stem.startswith("nii"):
        bg1 = np.mean(hdu.data[y1:y1+30], axis=0)
        bg2 = np.mean(hdu.data[y2-30:y2], axis=0)
    else:
        bg1 = np.median(hdu.data[y1-10:y1], axis=0)
        bg2 = np.median(hdu.data[y2:y2+10], axis=0)
    im = hdu.data - 0.5 * (bg1 + bg2)
    scale = np.percentile(im[y1:y2, x1:x0], 99.99)
    im /= scale
    print(filepath.stem, scale)
    # Save a FITS file of BG-subtracted and normalized image
    fits.PrimaryHDU(
        header=hdu.header,
        data=im,
    ).writeto(
        str(filepath).replace(".fits", "-bgsub.fits"),
        overwrite=True,
    )
# -

# And then do the plot separately

# +
file_list = sorted(pvpath2.glob("*-pv-coadd-bgsub.fits"))

N = len(file_list)
ncols = 2
nrows = (N // ncols)
fig = plt.figure(figsize=(8 * ncols, 10 * nrows))

vsys = -33
v1, v2 = vsys - 100, vsys + 100
s1, s2 = -35, 35

contours = "oiii", "ha"
kernel = Gaussian2DKernel(x_stddev=0.1)
for i, filepath in enumerate(file_list):
    hdu, = fits.open(filepath)
    im = hdu.data
    w = WCS(hdu.header)
    ax = plt.subplot(nrows, ncols, i + 1, projection=w)
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    y1, y2 = [int(_) for _ in ylims]
    x1, x2 = [int(_) for _ in xlims]
    ax.imshow(im, vmin=-0.1, vmax=1.0, aspect="auto")
    # ims = convolve_fft(im, kernel)
    if filepath.stem.startswith(contours):
        ax.contour(
            im, 
            levels=[0.01, 0.02, 0.04, 0.08, 0.16], 
            colors="w",
            linewidths=[0.5, 1.0, 1.5, 2.0, 2.5],
        )
    x0, y0 = w.world_to_pixel_values(vsys, 0.0)
    ax.axhline(y0, color="orange", ls="dashed", lw=4, alpha=0.3)
    ax.axvline(x0, color="orange", ls="dashed", lw=4, alpha=0.3)
    ax.set(xlim=xlims, ylim=ylims)
    ax.set_title(filepath.stem, pad=16)
figfile = "ou5-coadd-2dspec.pdf"
fig.savefig(figfile)
fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")
...;
# -

# ### And do the spatial profiles along the slit

sns.husl_palette(l=0.5, h=0.04, n_colors=3)

# We want to use the same colors that we used for the image-derived profiles in 04-05 notebook. With the addition of orange for N II

coldict = {}
coldict["heii"], coldict["ha"], coldict["oiii"] = sns.husl_palette(l=0.5, h=0.06, n_colors=3)
coldict["nii"] = "tab:orange"
coldict

# +
file_list = sorted(pvpath2.glob("*-pv-coadd-bgsub.fits"))

fig, ax = plt.subplots(
    figsize=(10, 6),
)
labels = {
    "oiii": r"[O III] $\lambda 5007$",
    "ha": r"H$\alpha$  $\lambda 6563$",
    "nii": "[N II]  $\lambda 6583$",
    "heii": "He II  $\lambda 6560$",
}
vsys = -33
v1, v2 = -75, 10
s1, s2 = -35, 35
offset = 0.0
profiles = {}
positions = {}
for i, filepath in enumerate(file_list):
    hdu, = fits.open(filepath)
    line_label = filepath.stem.split("-")[0]
    im = hdu.data
    w = WCS(hdu.header)
    ns, nv = hdu.data.shape
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    x1, x2 = [int(_) for _ in xlims]
    y1, y2 = [int(_) for _ in ylims]
    profile = hdu.data[y1:y2, x1:x2].mean(axis=1)
    _, pos = w.pixel_to_world_values([0]*ns, np.arange(ns))
    pos = pos[y1:y2]
    profiles[line_label] = profile.copy()
    positions[line_label] = pos.copy()
    maxscale = np.max(profile)
    sumscale = np.sum(profile[np.abs(pos) <= 10])
    profile *= 1 / maxscale
    print(f"{filepath.stem=}, {maxscale=}, {sumscale=}")
    line, = ax.plot(pos, profile + offset, 
                    label=line_label, 
                    color=coldict[line_label], ds="steps-mid")
    ax.text(-30, offset + 0.15, labels[line_label], color=line.get_color())
    ax.axhline(offset, linestyle="dashed", c="k", lw=1,)
    offset += 1
ax.axvline(0.0, linestyle="dashed", c="k", lw=1,)
#ax.legend(ncol=2)
ax.set(
    xlabel="Displacement along South–North slit, arcsec",
    xlim=[-35, 35],
)
figfile = "ou5-coadd-spatial-profiles-1d.pdf"
fig.savefig(figfile, bbox_inches="tight")
fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")
...;
# -

# Repeat but for line ratios. However, these are not so useful since we have already normalised all the spectra

# +
fig, ax = plt.subplots(
    figsize=(10, 6),
)

pos = positions["ha"]
offset = 0.0
for line_label in "heii", "oiii", "nii":
    ratio = profiles[line_label] / profiles["ha"]
    smax = 30 if line_label == "oiii" else 11
    ratio[np.abs(pos) > smax] = np.nan
    label = fr"{labels[line_label]} / H$\alpha$"
    line, = ax.plot(
        pos, ratio + offset, label=label, 
        # ds="steps-mid",
    )
    # ax.text(-30, offset + 0.15, label, color=line.get_color())
    ax.axhline(offset, linestyle="dashed", c="k", lw=1,)
    #offset += 1
ax.plot(pos, np.log10(profiles["ha"]), label=r"log$_{10}$ H$\alpha$", color="k")
ax.plot(pos, np.log10(profiles["oiii"]), label=None, color="0.5")
ax.plot(pos, np.log10(profiles["heii"]), label=None, color="c")
ax.axvline(0.0, linestyle="dashed", c="k", lw=1,)
ax.legend(ncol=2)
ax.set(
    xlabel="Displacement along slit, arcsec",
    ylim=[-3, 3],
)
figfile = "ou5-coadd-spatial-ratio-profiles-1d.pdf"
fig.savefig(figfile, bbox_inches="tight")
fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")
...;
# -

# ### Summed spectrum of entire inner lobes
#

# +
file_list = sorted(pvpath2.glob("*-pv-coadd-bgsub.fits"))

fig, ax = plt.subplots(
    figsize=(10, 6),
)
labels = {
    "oiii": "[O III]",
    "ha": r"H$\alpha$",
    "nii": "[N II]",
    "heii": "He II",
}
vprofiles_all = {}
vsys = -33
v1, v2 = -120, 40
s1, s2 = -6, 6
offset = 0.0
for i, filepath in enumerate(file_list):
    hdu, = fits.open(filepath)
    line_label = filepath.stem.split("-")[0]
    im = hdu.data
    w = WCS(hdu.header)
    ns, nv = hdu.data.shape
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    x1, x2 = [int(_) for _ in xlims]
    y1, y2 = [int(_) for _ in ylims]
    profile = hdu.data[y1:y2, x1:x2].mean(axis=0)
    vels, _ = w.pixel_to_world_values(np.arange(nv), [0]*nv)
    vels = vels[x1:x2]
    profile *= 1/ np.max(profile)
    vprofiles_all[line_label] = profile
    line, = ax.plot(vels, profile + offset, label=line_label, ds="steps-mid")
    ax.text(-100, offset + 0.15, labels[line_label], color=line.get_color())
    ax.axhline(offset, linestyle="dashed", c="k", lw=1,)
    offset += 1
ax.axvline(0.0, linestyle="dashed", c="k", lw=1,)
#ax.legend(ncol=2)
ax.set(
    xlabel="Heliocentric velocity",
)
figfile = "ou5-coadd-velocity-profiles-1d.pdf"
fig.savefig(figfile, bbox_inches="tight")
fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")
...;
# -

# ### Summed spectrum for $3.5 < |z| < 8.0$

# Repeat but excluding the equatorial plane, in order to avoid the high-velocity wings. 

# +
file_list = sorted(pvpath2.glob("*-pv-coadd-bgsub.fits"))

fig, ax = plt.subplots(
    figsize=(10, 6),
)
labels = {
    "oiii": "[O III]",
    "ha": r"H$\alpha$",
    "nii": "[N II]",
    "heii": "He II",
}
vprofiles_ilobes = {}
vsys = -33
v1, v2 = -120, 40
s1, s2 = 3.5, 8
offset = 0.0
for i, filepath in enumerate(file_list):
    hdu, = fits.open(filepath)
    line_label = filepath.stem.split("-")[0]
    im = hdu.data
    w = WCS(hdu.header)
    ns, nv = hdu.data.shape

    # North lobe
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    x1, x2 = [int(_) for _ in xlims]
    y1, y2 = [int(_) for _ in ylims]
    profile = hdu.data[y1:y2, x1:x2].mean(axis=0)

    # South lobe
    xlims, ylims = w.world_to_pixel_values([v1, v2], [-s2, -s1])
    x1, x2 = [int(_) for _ in xlims]
    y1, y2 = [int(_) for _ in ylims]
    profile += hdu.data[y1:y2, x1:x2].mean(axis=0)

    vels, _ = w.pixel_to_world_values(np.arange(nv), [0]*nv)
    vels = vels[x1:x2]
    profile *= 1/ np.max(profile)
    vprofiles_ilobes[line_label] = profile
    line, = ax.plot(vels, profile + offset, label=line_label, ds="steps-mid")
    ax.text(-100, offset + 0.15, labels[line_label], color=line.get_color())
    ax.axhline(offset, linestyle="dashed", c="k", lw=1,)
    offset += 1
ax.axvline(0.0, linestyle="dashed", c="k", lw=1,)
#ax.legend(ncol=2)
ax.set(
    xlabel="Heliocentric velocity",
)
figfile = "ou5-coadd-velocity-profiles-1d-avoid-equator.pdf"
fig.savefig(figfile)
fig.savefig(figfile.replace(".pdf", ".jpg"))
...;
# -

# ### Convolve [O III] with Gaussian to match Ha

from astropy.convolution import convolve, Gaussian1DKernel

# Choose whether to excluide equatorial region or not

vprofiles = vprofiles_all
#vprofiles = vprofiles_ilobes

# Small correction to the [O III] zero level

vprofiles["oiii"] -= 0.01

# Shift the Ha profile to match the mean velocities

# + editable=true slideshow={"slide_type": ""}
vmean_h = np.average(vels, weights=vprofiles["ha"])
vmean_o = np.average(vels, weights=vprofiles["oiii"])
vmean_h, vmean_o
# -

vshift = vmean_h - vmean_o
vshift

shifted = np.interp(vels + vshift, vels, vprofiles["ha"])
fig, ax = plt.subplots(figsize=(6, 2))
ax.plot(vels, vprofiles["ha"])
ax.plot(vels, shifted)

# That seems to have worked correctly by shifting it to the right.

vprofiles["ha"] = shifted

# Ensure that total flux is same for [O III] and Ha

boost = np.sum(vprofiles["ha"]) / np.sum(vprofiles["oiii"])
vprofiles["oiii"] *= boost
boost

# Estimate noise from the continuum level

outside = np.abs(vels - vsys) >= 50
noise_rms_ha = np.std(vprofiles["ha"][outside])
noise_rms_oiii = np.std(vprofiles["oiii"][outside])
noise_rms_diff = np.sqrt(noise_rms_ha**2 + noise_rms_oiii**2)
noise_rms_ha, noise_rms_oiii, noise_rms_diff

# S/N ratio

snr_ha = vprofiles["ha"].max() / noise_rms_ha
snr_oiii = vprofiles["oiii"].max() / noise_rms_oiii
snr_ha, snr_oiii

# So this is very high, which is why our reduced chi squared are always in the hundreds

# +
fig, ax = plt.subplots()

ax.plot(vels, vprofiles["ha"], lw=4, color="k")
ax.plot(vels, vprofiles["oiii"])

sprofile = convolve(vprofiles["oiii"], Gaussian1DKernel(stddev=3.8))
ax.plot(vels, sprofile)

# -

sprofile.sum() / np.sum(vprofiles["ha"])

dv = np.diff(vels)[0]

ss_list = np.linspace(0.0, 8.0, 25)
skip = 3
with sns.color_palette("rocket", 1 + len(ss_list) // skip):
    fig, ax = plt.subplots(figsize=(8, 5))
    sum_square_residuals = []
    sum_square_residuals_core = []
    sum_square_residuals_wings = []
    core_width = 20
    core = np.abs(vels + 33) < core_width
    wings = (~core) & (~outside)
    for index, ss in enumerate(ss_list):
        if ss > 0:
            sprofile = convolve(vprofiles["oiii"], Gaussian1DKernel(stddev=ss, mode="oversample"))
        else:
            sprofile = vprofiles["oiii"]
        residuals = (sprofile - vprofiles["ha"]) / noise_rms_diff
        if index % skip == 0:
            if index % (skip*2) == 0:
                label = f"$\sigma = {dv * ss:.0f}$ km/s"
                lw = 2.5
            else:
                label = None
                lw = 1.0
            ax.plot(vels, residuals, label=label, lw=lw)
        sum_square_residuals.append(np.sum(residuals**2))
        sum_square_residuals_core.append(np.sum(residuals[core]**2))
        sum_square_residuals_wings.append(np.sum(residuals[wings]**2))
    ax.axvline(-33, ls="dashed", color="k", lw=1)
    ax.axhline(0, ls="dashed", color="k", lw=1)
    ax.axvspan(-33 - core_width, -33 + core_width, color="b", alpha=0.1, zorder=-100)
    ax.legend(fontsize="small", title="Extra broadening")
    ax.set_ylim(None, 190)
    ax.set_xlabel("Heliocentric velocity, km/s")
    ax.set_ylabel(r"$ (\mathrm{[O III]}^\bigstar - \mathrm{H\alpha})\ / \ s_\mathrm{noise}$")
    figfile = "pn-ou5-convolution-residuals.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


with sns.color_palette("colorblind", n_colors=3):
    fig, ax = plt.subplots(figsize=(8, 5))
    ss_opt = np.interp(0.0, np.gradient(sum_square_residuals), ss_list)
    label = f"All velocities, $\sigma_\mathrm{{opt}} = {dv * ss_opt:.1f}$ km/s"
    line, = ax.plot(dv * ss_list, sum_square_residuals, label=label)
    c = line.get_color()
    ax.axvline(ss_opt * dv, linestyle="dotted", color=c, ymax=0.25)
    
    ss_opt_core = np.interp(0.0, np.gradient(sum_square_residuals_core), ss_list)
    label = f"Core only, $\sigma_\mathrm{{opt}} = {dv * ss_opt_core:.1f}$ km/s"
    line, = ax.plot(dv * ss_list, sum_square_residuals_core, label=label)
    c = line.get_color()
    ax.axvline(ss_opt_core * dv, linestyle="dotted", color=c, ymax=0.25)

    ss_opt_wings = np.interp(0.0, np.gradient(sum_square_residuals_wings), ss_list)
    label = f"Wings only, $\sigma_\mathrm{{opt}} = {dv * ss_opt_wings:.1f}$ km/s"
    line, = ax.plot(dv * ss_list, sum_square_residuals_wings, label=label)
    c = line.get_color()
    ax.axvline(ss_opt_wings * dv, linestyle="dotted", color=c, ymax=0.25)
    
    ax.set_ylim(0.0, None)
    ax.set_xlabel(r"Extra broadening, $\sigma$, km/s")
    ax.set_ylabel("Sum squared residuals: $\chi^2$")
    ax.legend(fontsize="small")
    figfile = "pn-ou5-convolution-optimum.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


# Repeat but with the reduced chi-squared. First, calculate the degrees of freedom:

npts_core = np.sum(core).astype(int)
npts_wings = np.sum(wings).astype(int)
nu_core = npts_core - 1
nu_wings = npts_wings - 1
nu_full = npts_core + npts_wings - 1
nu_core, nu_wings, nu_full

with sns.color_palette("colorblind", n_colors=3):
    fig, ax = plt.subplots(figsize=(8, 5))
    ss_opt = np.interp(0.0, np.gradient(sum_square_residuals), ss_list)
    label = f"All velocities, $\sigma_\mathrm{{opt}} = {dv * ss_opt:.1f}$ km/s"
    line, = ax.plot(dv * ss_list, sum_square_residuals / nu_full, label=label)
    c = line.get_color()
    ax.axvline(ss_opt * dv, linestyle="dotted", color=c, ymax=0.25)
    
    ss_opt_core = np.interp(0.0, np.gradient(sum_square_residuals_core), ss_list)
    label = f"Core only, $\sigma_\mathrm{{opt}} = {dv * ss_opt_core:.1f}$ km/s"
    line, = ax.plot(dv * ss_list, sum_square_residuals_core / nu_core, label=label)
    c = line.get_color()
    ax.axvline(ss_opt_core * dv, linestyle="dotted", color=c, ymax=0.25)

    ss_opt_wings = np.interp(0.0, np.gradient(sum_square_residuals_wings), ss_list)
    label = f"Wings only, $\sigma_\mathrm{{opt}} = {dv * ss_opt_wings:.1f}$ km/s"
    line, = ax.plot(dv * ss_list, sum_square_residuals_wings / nu_wings, label=label)
    c = line.get_color()
    ax.axvline(ss_opt_wings * dv, linestyle="dotted", color=c, ymax=0.25)
    
    ax.set_ylim(0.0, 900)
    ax.set_xlabel(r"Extra broadening, $\sigma$, km/s")
    ax.set_ylabel(r"Goodness of fit: $\chi^2 / \nu$")
    ax.legend(fontsize="small")
    figfile = "pn-ou5-convolution-optimum-chi2.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


sns.set_color_codes()
with sns.color_palette("dark"):

    fig, ax = plt.subplots(figsize=(8, 5))
    #ds = "default"
    ds = "steps-mid"
    olw, hlw = 4, 2
    ocolor = (1, 0.2, 0.2)
    offset_step = 0.5
    offset = 0
    ax.plot(vels, offset + vprofiles["oiii"], color="k", lw=olw, ds=ds, label="[O III]")
    ax.plot(vels, offset + vprofiles["ha"], lw=hlw, color=ocolor, ds=ds, label=r"H$\alpha$")
    text = "No broadening\n" + r"$\sigma = 0$ km/s"
    ax.text(-120, offset + 0.1, text, fontsize="small")
    ax.axhline(offset, ls="dashed", color="k", lw=1)

    ds = "default"
    dsha = "steps-mid"
    offset += offset_step
    sprofile = convolve(vprofiles["oiii"], Gaussian1DKernel(stddev=ss_opt_wings, mode="oversample"))
    ax.plot(vels, offset + sprofile, color="k", lw=olw)
    ax.plot(vels, offset + vprofiles["ha"], lw=hlw, ds=dsha, color=ocolor)
    text = "Optimize wings\n" + rf"$\sigma = {ss_opt_wings * dv:.1f}$ km/s"
    ax.text(-120, offset + 0.1, text, fontsize="small")
    ax.text(20, offset + 0.1, "A", fontsize="small")
    ax.axhline(offset, ls="dashed", color="k", lw=1)
    
    offset += offset_step
    sprofile = convolve(vprofiles["oiii"], Gaussian1DKernel(stddev=ss_opt_core, mode="oversample"))
    ax.plot(vels, offset + sprofile, color="k", lw=olw)
    ax.plot(vels, offset + vprofiles["ha"], lw=hlw, ds=dsha, color=ocolor)
    text = "Optimize core\n" + rf"$\sigma = {ss_opt_core * dv:.1f}$ km/s"
    ax.text(-120, offset + 0.1, text, fontsize="small")
    ax.text(20, offset + 0.1, "B", fontsize="small")
    ax.axhline(offset, ls="dashed", color="k", lw=1)

    ax.axvspan(-33 - core_width, -33 + core_width, color="b", alpha=0.1, zorder=-100)
    ax.axvline(-33, ls="dashed", color="k", lw=1)
    ax.set_xlabel("Heliocentric velocity, km/s")
    ax.legend()
    
    figfile = "pn-ou5-convolution-fits.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


# +
T4_opt = ((ss_opt * dv)**2 - 10.233) / 77.34
T4_opt_core = ((ss_opt_core * dv)**2 - 10.233) / 77.34
T4_opt_wings = ((ss_opt_wings * dv)**2 - 10.233) / 77.34

T4_opt_wings, T4_opt, T4_opt_core
# -

# So this gives $T = 6000_{-3000}^{+600}$ K

# # Velocities in the horizontal slit

# +

file_dict = {
    "ha": "spm0440o-ha-PA085-sep+000-regrid.fits",
    "nii": "spm0440o-nii-PA085-sep+000-regrid.fits",
    "heii": "spm0440o-heii-PA085-sep+000-regrid.fits",
}

N = len(file_dict)

vsys = -33
v1, v2 = vsys - 100, vsys + 100
s1, s2 = -15, 15

for i, (lineid, filename) in enumerate(file_dict.items()):
    filepath = pvpath2 / filename
    hdu, = fits.open(filepath)
    w = WCS(hdu.header)
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    y1, y2 = [int(_) for _ in ylims]
    x1, x2 = [int(_) for _ in xlims]
    x0, y0 = w.world_to_pixel_values(0.0, 0.0)
    x0 = int(x0)
    if filepath.stem.startswith("nii"):
        bg1 = np.mean(hdu.data[y1:y1+30], axis=0)
        bg2 = np.mean(hdu.data[y2-30:y2], axis=0)
    else:
        bg1 = np.median(hdu.data[y1-10:y1], axis=0)
        bg2 = np.median(hdu.data[y2:y2+10], axis=0)
    im = hdu.data - 0.5 * (bg1 + bg2)
    if not lineid in []:
        # Re-use ha scale for nii 
        scale = np.percentile(im[y1:y2, x1:x0], 99.99)
    im /= scale
    print(lineid, scale)
    # Save a FITS file of BG-subtracted and normalized image
    fits.PrimaryHDU(
        header=hdu.header,
        data=im,
    ).writeto(
        pvpath2 / f"{lineid}-pv-horizontal-bgsub.fits",
        overwrite=True,
    )

# +
file_list = sorted(pvpath2.glob("*-pv-horizontal-bgsub.fits"))

N = len(file_list)
ncols = 2
nrows = ((N + 1) // ncols)
fig = plt.figure(figsize=(8 * ncols, 10 * nrows))

vsys = -33
v1, v2 = vsys - 100, vsys + 100
s1, s2 = -35, 35

kernel = Gaussian2DKernel(x_stddev=0.1)
for i, filepath in enumerate(file_list):
    hdu, = fits.open(filepath)
    im = hdu.data
    w = WCS(hdu.header)
    ax = plt.subplot(nrows, ncols, i + 1, projection=w)
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    y1, y2 = [int(_) for _ in ylims]
    x1, x2 = [int(_) for _ in xlims]
    ax.imshow(im, vmin=-0.1, vmax=1.0, aspect="auto")
    x0, y0 = w.world_to_pixel_values(vsys, 0.0)
    ax.axhline(y0, color="orange", ls="dashed", lw=4, alpha=0.3)
    ax.axvline(x0, color="orange", ls="dashed", lw=4, alpha=0.3)
    ax.set(xlim=xlims, ylim=ylims)
    ax.set_title(filepath.stem, pad=16)
figfile = "ou5-horizontal-2dspec.pdf"
fig.savefig(figfile)
fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")
...;
# -

# Mean velocity as function of position. *No need to do this* It looks like the bright border has a peak very close to vsys

# ## Spatial profiles along the horizontal slit

# +
file_list = sorted(pvpath2.glob("*-pv-horizontal-bgsub.fits"))

fig, ax = plt.subplots(
    figsize=(10, 6),
)
vsys = -33
v1, v2 = -75, 10
s1, s2 = -35, 35
offset = 0.0
for i, filepath in enumerate(file_list):
    hdu, = fits.open(filepath)
    line_label = filepath.stem.split("-")[0]
    im = hdu.data
    w = WCS(hdu.header)
    ns, nv = hdu.data.shape
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    x1, x2 = [int(_) for _ in xlims]
    y1, y2 = [int(_) for _ in ylims]
    profile = hdu.data[y1:y2, x1:x2].mean(axis=1)
    _, pos = w.pixel_to_world_values([0]*ns, np.arange(ns))
    pos = pos[y1:y2]
    profile *= 1/ np.max(profile)
    line, = ax.plot(-pos, profile + offset, 
                    label=line_label, color=coldict[line_label], ds="steps-mid")
    ax.text(20, offset + 0.25, labels[line_label], color=line.get_color())
    ax.axhline(offset, linestyle="dashed", c="k", lw=1,)
    offset += 1
ax.axvline(0.0, linestyle="dashed", c="k", lw=1,)
#ax.legend(ncol=2)
ax.set(
    xlabel="Displacement along East–West slit, arcsec",
    xlim=[-35, 35],
)
figfile = "ou5-horizontal-spatial-profiles-1d.pdf"
fig.savefig(figfile, bbox_inches="tight")
fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")
...;
# -

# Unfortunately, the s/n is too low for this to be very useful. Hopefully the NOT images will be better.
#
# However, it is consistent with He II being less extended along the equatorial axis







# # Look at the 70 micron slit
#
# This has higher spectral resolution. And we have one position around +5 arcsec, which hits the tangent point of the inner shell, where the expansion broadening should be minimised

# ## Subtract the background ISM line 

# +

file_dict = {
    "oiii": "N10045-oiii-PA359-sep+005-regrid.fits",
    "ha": "N10043-ha-PA359-sep+004-regrid.fits",
    "heii": "N10043-heii-PA359-sep+004-regrid.fits",
    "nii": "N10043-nii-PA359-sep+004-regrid.fits",
}

N = len(file_dict)

vsys = -33
v1, v2 = vsys - 100, vsys + 100
s1, s2 = -15, 15

for i, (lineid, filename) in enumerate(file_dict.items()):
    filepath = pvpath2 / filename
    hdu, = fits.open(filepath)
    w = WCS(hdu.header)
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    y1, y2 = [int(_) for _ in ylims]
    x1, x2 = [int(_) for _ in xlims]
    x0, y0 = w.world_to_pixel_values(0.0, 0.0)
    x0 = int(x0)
    if filepath.stem.startswith("nii"):
        bg1 = np.mean(hdu.data[y1:y1+30], axis=0)
        bg2 = np.mean(hdu.data[y2-30:y2], axis=0)
    else:
        bg1 = np.median(hdu.data[y1-10:y1], axis=0)
        bg2 = np.median(hdu.data[y2:y2+10], axis=0)
    im = hdu.data - 0.5 * (bg1 + bg2)
    if not lineid in ["nii", "heii"]:
        # Re-use ha scale for nii and heii
        scale = np.percentile(im[y1:y2, x1:x0], 99.99)
    im /= scale
    print(lineid, scale)
    # Save a FITS file of BG-subtracted and normalized image
    fits.PrimaryHDU(
        header=hdu.header,
        data=im,
    ).writeto(
        pvpath2 / f"{lineid}-pv-tangent-point-70-bgsub.fits",
        overwrite=True,
    )
# -

# ## Plot the bg-subtracted PV arrays

# +
file_list = sorted(pvpath2.glob("*-pv-tangent-*-bgsub.fits"))

N = len(file_list)
ncols = 2
nrows = (N // ncols)
fig = plt.figure(figsize=(8 * ncols, 10 * nrows))

vsys = -33
v1, v2 = vsys - 100, vsys + 100
s1, s2 = -35, 35

kernel = Gaussian2DKernel(x_stddev=0.1)
for i, filepath in enumerate(file_list):
    hdu, = fits.open(filepath)
    im = hdu.data
    w = WCS(hdu.header)
    ax = plt.subplot(nrows, ncols, i + 1, projection=w)
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    y1, y2 = [int(_) for _ in ylims]
    x1, x2 = [int(_) for _ in xlims]
    ax.imshow(im, vmin=-0.1, vmax=1.0, aspect="auto")
    x0, y0 = w.world_to_pixel_values(vsys, 0.0)
    ax.axhline(y0, color="orange", ls="dashed", lw=4, alpha=0.3)
    ax.axvline(x0, color="orange", ls="dashed", lw=4, alpha=0.3)
    ax.set(xlim=xlims, ylim=ylims)
    ax.set_title(filepath.stem, pad=16)
figfile = "ou5-tp70-2dspec.pdf"
fig.savefig(figfile)
fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")
...;
# -

# ## Summed spectra over equatorial region

# +
file_list = sorted(pvpath2.glob("*-pv-tangent-*-bgsub.fits"))

fig, ax = plt.subplots(
    figsize=(10, 6),
)
labels = {
    "oiii": "[O III]",
    "ha": r"H$\alpha$",
    "nii": "[N II]",
    "heii": "He II",
}
vprofiles_tp70 = {}
vsys = -33
v1, v2 = -120, 40
s1, s2 = -2.5, 2.5
#s1, s2 = -5, 5
offset = 0.0
for i, filepath in enumerate(file_list):
    hdu, = fits.open(filepath)
    line_label = filepath.stem.split("-")[0]
    im = hdu.data
    w = WCS(hdu.header)
    ns, nv = hdu.data.shape
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    x1, x2 = np.rint(xlims).astype(int)
    y1, y2 = np.rint(ylims).astype(int)
    profile = hdu.data[y1:y2, x1:x2].mean(axis=0)
    vels, _ = w.pixel_to_world_values(np.arange(nv), [0]*nv)
    vels = vels[x1:x2]
    #profile *= 1/ np.max(profile)
    vprofiles_tp70[line_label] = profile
    line, = ax.plot(vels, profile + offset, label=line_label, ds="steps-mid")
    ax.text(-100, offset + 0.15, labels[line_label], color=line.get_color())
    ax.axhline(offset, linestyle="dashed", c="k", lw=1,)
    offset += 1
ax.axvline(0.0, linestyle="dashed", c="k", lw=1,)
ax.axvline(vsys, linestyle="dashed", c="k", lw=1,)
#ax.legend(ncol=2)
ax.set(
    xlabel="Heliocentric velocity",
)
figfile = "ou5-tp70-velocity-profiles-1d.pdf"
fig.savefig(figfile, bbox_inches="tight")
fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")
...;
# -

# ### Remove continuum gradients
#
# We want to make sure that the zero line is flat and actually at zero. Later on, we create a window around the mean velocity, but we do not know what that is yet, so we will use a fixed window in velocity. In order to make absolutely sure we are not overlapping with the Ha wings, we will use from -85 to +10 as the line window, and everything outside of that will be used for the continuum fit. 

#is_continuum = (vels <= -85) | (vels > 10)
is_continuum = (vels <= -75) | (vels > 0)

p_ha = np.polynomial.Polynomial.fit(
    vels[is_continuum], vprofiles_tp70["ha"][is_continuum], deg=1,
)
p_ha.coef

p_oiii = np.polynomial.Polynomial.fit(
    vels[is_continuum], vprofiles_tp70["oiii"][is_continuum], deg=1,
)
p_oiii.coef

fig, ax = plt.subplots()
line, = ax.plot(vels, p_ha(vels))
ax.plot(
    vels[is_continuum], 
    vprofiles_tp70["ha"][is_continuum], 
    ".",
    alpha=0.5,
    color=line.get_color(),
)
line, = ax.plot(vels, p_oiii(vels))
ax.plot(
    vels[is_continuum], 
    vprofiles_tp70["oiii"][is_continuum], 
    ".",
    alpha=0.5,
    color=line.get_color(),
)


# Apply the linear continuum correction

vprofiles_tp70["ha"] -= p_ha(vels)
vprofiles_tp70["oiii"] -= p_oiii(vels)

# ### Match the mean velocities of the two lines
#
# We want to avoid interpolation if we can, since we do not want to be broadening either of the lines

# + editable=true slideshow={"slide_type": ""}
vmean_h = np.average(vels, weights=vprofiles_tp70["ha"])
vmean_o = np.average(vels, weights=vprofiles_tp70["oiii"])
vmean_h, vmean_o
# -

vshift = vmean_h - vmean_o
vshift

ivshift = np.rint(vshift / dv) * dv
ivshift

shifted = np.interp(vels + ivshift, vels, vprofiles_tp70["ha"])
fig, ax = plt.subplots(figsize=(6, 2))
ax.plot(vels, vprofiles_tp70["ha"])
ax.plot(vels, shifted)

# That seems to have worked correctly by shifting it to the right.

vprofiles_tp70["ha"] = shifted

# Ensure that total flux is same for [O III] and Ha

boost = np.sum(vprofiles_tp70["ha"]) / np.sum(vprofiles_tp70["oiii"])
vprofiles_tp70["oiii"] *= boost
boost

# Save the mean velocity

vmean_o_tp70 = vmean_o

# ### Check the plots of the profiles again

fig, ax = plt.subplots(
    figsize=(10, 6),
)
v1, v2 = -120, 40
offset = 0.0
for lineid in "ha", "oiii":
    line, = ax.plot(vels, vprofiles_tp70[lineid] + offset, ds="steps-mid")
    ax.text(-100, offset + 0.15, labels[lineid], color=line.get_color())
    ax.axhline(offset, linestyle="dashed", c="k", lw=1,)
    offset += 1
ax.axvline(0.0, linestyle="dashed", c="k", lw=1,)
# ax.axvline(vsys, linestyle="dashed", c="k", lw=1,)
ax.axvline(vmean_o_tp70, linestyle="dashed", c="k", lw=1,)
#ax.legend(ncol=2)
ax.set(
    xlabel="Heliocentric velocity",
)
figfile = "ou5-tp70-velocity-profiles-corrected-1d.pdf"
fig.savefig(figfile, bbox_inches="tight")
fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")
...;

# # Two-phase model
#
# We will now repeat this with the two-phase model from the other notebook, with $\alpha \equiv T_c / T_w = 0.1$ and $\omega \equiv I_c / (I_c + I_w) = 0.5$.

# ## Data on the fine structure components

# Read the Clegg 1999 data for $n = 100$ pcc

tab = Table.read(Path.cwd().parent / "docs" / "h-case-b-n2.csv")

tab

np.log10(list(map(float, tab.colnames[3:])))

# + editable=true slideshow={"slide_type": ""}
itable = Table(tab.columns[3:])
np.array(itable)
# -

np.array(tab["d v"])

# ## Calculate profiles with astropy.modeling
#

from astropy import constants
from astropy.modeling import models

# Normalization for the rms widths

sig0 = np.sqrt( 
    constants.k_B * 10_000 * u.K / constants.m_p 
).to(u.km / u.s)
sig0

# Fine structure profile for T = 1000 K

vmean_1000 = np.average(tab["d v"], weights=tab["1000"])
sig_1000 = sig0.value * np.sqrt(0.1 * (1 - 1/16))
vmean_1000, sig_1000

comps1000 = [
    models.Gaussian1D(
        amplitude=_amp / sig_1000, 
        mean=_mean - vmean_1000,
        stddev=sig_1000,
    )
    for _mean, _amp in zip(tab["d v"], tab["1000"])
]

kernel1000 = comps1000[0]
for _comp in comps1000[1:]:
    kernel1000 += _comp
kernel1000

fig, ax = plt.subplots()
finevels = np.linspace(-50, 50, 500)
ax.plot(finevels, kernel1000(finevels))

# Fine structure profile for T = 10,000 K

vmean_10000 = np.average(tab["d v"], weights=tab["10000"])
sig_10000 = sig0.value * np.sqrt(1.0 * (1 - 1/16))
vmean_10000, sig_10000

comps10000 = [
    models.Gaussian1D(
        amplitude=_amp / sig_10000, 
        mean=_mean - vmean_10000,
        stddev=sig_10000,
    )
    for _mean, _amp in zip(tab["d v"], tab["10000"])
]

kernel10000 = comps10000[0]
for _comp in comps10000[1:]:
    kernel10000 += _comp
kernel10000

fig, ax = plt.subplots()
finevels = np.linspace(-50, 50, 500)
ax.plot(finevels, kernel10000(finevels))

kernel1000(finevels).sum(), kernel10000(finevels).sum()

# Now combine the two temperature components

fine_dv = np.diff(finevels)[0]

omega = 0.5
profile_2temp = (
    omega * kernel1000(finevels) / np.sum(kernel1000(finevels))
    + (1 - omega) * kernel10000(finevels) / np.sum(kernel10000(finevels))
) / fine_dv

# And compare with the one-temperature version with 6000 K

sig_6000 = sig0.value * np.sqrt(0.6 * (1 - 1/16))
kernel6000 = models.Gaussian1D(
    amplitude=1, 
    mean=0,
    stddev=sig_6000,
)
profile_1temp = kernel6000(finevels) / np.sum(kernel6000(finevels))
profile_1temp /= fine_dv

fig, ax = plt.subplots(figsize=(8, 5))
finevels = np.linspace(-50, 50, 500)
ax.plot(
    finevels, profile_2temp, 
    label=r"Two-phase: $T_\mathrm{w} = 10\,000$ K, $T_\mathrm{c} = 1000$ K"
)
ax.plot(
    finevels, profile_1temp + 0.01, 
    label=r"One-phase: $T = 6000$ K",
)
ax.legend(fontsize="small", title=r"Thermal + fine-structure H$\alpha$ profiles")
ax.set_ylim(None, 0.11)
ax.set_xlabel("Velocity, km/s")
figfile = "pn-ou5-two-temperature-kernel.pdf"
fig.savefig(figfile, bbox_inches="tight")
fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


# ## Calculate profiles with discrete gaussian model
#
# The idea here is to get away from the astropy.modeling framework, since it was making it hard to move over to the convolution part. 
#
# Also, we want to generalise it so we can use different values of $\alpha$ (cool to warm temperature ratio) and $\omega$ (emission fraction from cool phase).
#
# Finally, we will make sure that we use the cdf of the profile to account for the finite bin width, although that means specifying the bin width in advance

# ### Class to encapsulate a single phase

# +
import scipy.stats

_cdf = scipy.stats.norm.cdf

def discrete_gaussian(x, amplitude, mean, stddev, bin_width):
    "Gaussian profile integrated over finite bins"
    return amplitude * (
        _cdf(x + 0.5 * bin_width, loc=mean, scale=stddev)
        - _cdf(x - 0.5 * bin_width, loc=mean, scale=stddev)
    )

class PhaseProfile():
    """Fine structure plus thermal line profile for single temperature

    Currently H alpha only. Fine structure components from Clegg 1999
    """
    DATA_PATH = Path.cwd().parent / "docs" 
    WAV_REF = 6562.8812
    # RMS thermal broadening at 1e4 K
    SIG0 = np.sqrt( 
        constants.k_B * 10_000 * u.K / constants.m_p
    ).to(u.km / u.s).value
    
    def __init__(
        self, 
        temperature,
        datafile="h-case-b-n2.csv",
        A_other=16,
        dv=0.1,
        vmax=50,
    ):
        """
        
        """
        self.temperature = temperature
        self.A_other = A_other
        self.initialize_components(Table.read(self.DATA_PATH / datafile))
        # Mean velocity over components
        self.vmean = np.average(self.vcomps, weights=self.icomps)
        # Centroid lab wavelength (in air)
        self.wav0 = self.WAV_REF * (1 + self.vmean / 3e5)
        # RMS excess thermal sigma of each component
        self.sigma = self.SIG0 * np.sqrt(
            (self.temperature / 1e4) * (1 - 1/self.A_other)
        )
        # Velocity grid for evaluating profile
        self.dv = dv
        self.nvgrid = 1 + int(2 * vmax / dv)
        self.vgrid = np.linspace(-vmax, vmax, self.nvgrid)
        self.initialize_igrid()

    def initialize_components(self, table):
        # Velocity shifts of fs components
        self.vcomps = np.array(table["d v"])
        self.ncomps = len(self.vcomps)
        # Interpolate intensities in log10(T)
        itable = Table(table.columns[3:])
        # Array of log T from columns of intensity grid
        logTs = np.log10(list(map(float, itable.colnames)))
        # Interpolate the intensity of each component at desired T
        self.icomps = np.array([
            np.interp(
                np.log10(self.temperature),
                logTs,
                irow,
            )
            for irow in np.array(itable).tolist()
        ])
        
    def initialize_igrid(self):
        self.igrid = np.zeros_like(self.vgrid)
        for _icomp, _vcomp in zip(self.icomps, self.vcomps):
            self.igrid += discrete_gaussian(
                self.vgrid,
                _icomp,
                _vcomp,
                self.sigma,
                self.dv,
            ) / self.dv
        


# -

prof = PhaseProfile(1e2)

prof.vgrid

# ### Plot the profiles for different T

# Tlist = [100, 200, 500, 1000, 2000, 5000, 10000]
Tlist = [300, 1000, 3000, 10000]
nT = len(Tlist)
with sns.color_palette("plasma", n_colors=nT):
    fig, ax = plt.subplots()
    for _i, T in enumerate(Tlist):
        p = PhaseProfile(T, dv=0.1)
        line, = ax.plot(p.vgrid, p.igrid, label=f"{T} K")
        ax.axvline(p.vmean, color=line.get_color(), 
                   lw=2, ymax=1 - (_i/nT), ymin=1 - (1+_i)/nT,
                   ls="dashed",
                  )
    ax.set_xlim(-33, 27)
    ax.legend(ncol=1, title="Temperature")
    ax.set_xlabel("Velocity, km/s")
    figfile = "pn-ou5-fine-structure-ha.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


# ### Class to encapsulate two phases

# +
class TwoPhaseProfile():
    def __init__(self, alpha, omega, Twarm=1e4, dv=0.1):
        self.Twarm = Twarm
        self.alpha = alpha
        self.omega = omega
        self.Tcool = alpha * Twarm
        self.warm = PhaseProfile(self.Twarm, dv=dv)
        self.cool = PhaseProfile(self.Tcool, dv=dv)
        self.vgrid = self.warm.vgrid
        self.igrid = (
            (1 - self.omega) * self.warm.igrid 
            + self.omega * self.cool.igrid
        )
        self.vmean = np.average(self.vgrid, weights=self.igrid)

    def __call__(self, v):
        """Interpolated profile at velocity shift `v` centered on mean
        """
        return np.interp(v + self.vmean, self.vgrid, self.igrid)




# + editable=true slideshow={"slide_type": ""}
alist = 0.03, 0.1, 0.3
with sns.color_palette("mako", n_colors=len(alist)):
    fig, ax = plt.subplots()
    for alpha in reversed(alist):
        omega = 0.4 + alpha
        p = TwoPhaseProfile(alpha, omega, dv=0.1)
        label = fr"$T_\mathrm{{cool}} = {1e4 * alpha:.0f}$, $\omega = {omega:.2f}$"
        line, = ax.plot(p.vgrid - p.vmean, p.igrid, label=label)
    ax.legend(fontsize="x-small", loc="upper left", title="2-phase kernel")
    ax.set_xlim(-35, 35)
    ax.set_ylim(None, 0.12)
    ax.set_xlabel("Velocity shift, km/s")
    figfile = "pn-ou5-two-phase-alpha.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")

# -

# Test that the interpolation function works. Plot on a coarser grid

# + editable=true slideshow={"slide_type": ""}
alist = 0.03, 0.1, 0.3
vgrid = np.linspace(-50, 50, 101)
with sns.color_palette("mako", n_colors=len(alist)):
    fig, ax = plt.subplots()
    for alpha in alist:
        omega = 0.4 + alpha
        p = TwoPhaseProfile(alpha, omega, dv=0.1)
        line, = ax.plot(vgrid, p(vgrid),
                        label=fr"$\alpha = {alpha:.2f}$, $\omega = {omega:.2f}$")
    ax.legend(fontsize="x-small", loc="upper left", title="2-phase kernel")
    ax.set_xlim(-35, 35)
    ax.set_ylim(None, 0.11)
    ax.set_xlabel("Velocity, km/s")
    # figfile = "pn-ou5-two-phase-alpha.pdf"
    # fig.savefig(figfile, bbox_inches="tight")
    # fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")

# -

# ### Test the affect of the typical instrumental plus non-thermal broadening on the 2-phase profile
#
# The typical sigma from the [O III] gaussian fits is 7 km/s, so we can convolve the 2-phase profile with this to get an idea of what the individual Ha velocity components are predicted to look like

# + editable=true slideshow={"slide_type": ""}
def std_dev(x, y):
    xm = np.average(x, weights=y)
    xvar = np.average((x - xm)**2, weights=y)
    return np.sqrt(xvar)
    


alist = 0.03, 0.1, 0.3
vgrid = np.linspace(-50, 50, 1001)
vgrid_spacing = np.diff(vgrid)[0]
with sns.color_palette("mako", n_colors=len(alist)):
    fig, ax = plt.subplots()
    for alpha in alist:
        omega = 0.4 + alpha
        p = TwoPhaseProfile(alpha, omega, dv=0.1)
        std0 = std_dev(vgrid, p(vgrid))
        sprofile = convolve(
            p(vgrid),
            Gaussian1DKernel(stddev=7.0 / vgrid_spacing, mode="oversample"),
        )
        std = std_dev(vgrid, sprofile)
        label = (
            fr"$\alpha = {alpha:.2f}$, $\omega = {omega:.2f}$"
            fr": $\sigma_0 = {std0:.2f}$, $\sigma = {std:.2f}$"
        )
        line, = ax.plot(vgrid, sprofile, label=label, lw=0.5)
    ax.legend(fontsize="x-small", loc="upper left", title="2-phase kernel")
    ax.set_xlim(-35, 35)
    ax.set_ylim(None, 0.11)
    ax.set_xlabel("Velocity, km/s")


# + [markdown] editable=true slideshow={"slide_type": ""}
# So this shows that there is almost no perceptible difference between the different 2-phase models once we convolve them with the instrumental + non-thermal profile. Everything comes out looking like a gaussian with a total sigma of about 10 km/s, which is consistent with the component widths of the Ha gaussian decomposition. (See 04-02 notebook)

# + [markdown] editable=true slideshow={"slide_type": ""}
# ### Make a smoothing kernel from 2-phase model
# -

# Convenience wrapper function for the class

def two_phase_profile(v, alpha=0.1, omega=0.5, Twarm=1e4):
    _profile = TwoPhaseProfile(alpha, omega, Twarm)
    return _profile(v)


# Try to use `astropy.convolution.utils.discretize_model`, which supposedly can be used with any callable. If we make an instance of `TwoPhaseProfile` first, then that will be our callable. 
#
# *We still need to worry about pixel verus velocity units*

from astropy.convolution.utils import discretize_model

p = TwoPhaseProfile(alpha=0.03, omega=0.43)
PIXEL_SIZE = 2
def pixel_profile(x):
    return p(PIXEL_SIZE * x)
x_range = (-20, 21)
kernel_ce = discretize_model(pixel_profile, x_range, mode="center")
kernel_li = discretize_model(pixel_profile, x_range, mode="linear_interp")
kernel_os = discretize_model(pixel_profile, x_range, mode="oversample")
kernel_in = discretize_model(pixel_profile, x_range, mode="integrate")


fig, ax = plt.subplots()
x_arr = np.arange(*x_range)
ax.plot(x_arr, kernel_ce, label="center")
ax.plot(x_arr, kernel_li, label="linear interpolation")
ax.plot(x_arr, kernel_os, label="oversample")
ax.plot(x_arr, kernel_in, label="integrate")
ax.legend(fontsize="x-small", loc="upper left")
ax.set_ylim(None, 0.11)
ax.set_xlabel("Pixel")

from astropy.convolution import CustomKernel

kernel = CustomKernel(kernel_os)

# +
fig, ax = plt.subplots()

ax.plot(vels, vprofiles["ha"], lw=4, color="k")
ax.plot(vels, vprofiles["oiii"])

sprofile = convolve(vprofiles["oiii"], kernel)
ax.plot(vels, sprofile)
# -

# Well that seems to fit pretty well. Which is a shame if we want to rule out a low-T phase











# ## Repeat convolution with [O III] but for two-phase model
#
# We will generate the same graphs that we did for the gaussian case

# ### Optimum cool fraction $\omega$ for different cool temperatures

# #### $\alpha = 0.03$

om_list = np.linspace(0.0, 1.0, 41)[::-1]
skip = 5
alpha = 0.03
with sns.color_palette("rocket", 1 + len(om_list) // skip):
    fig, ax = plt.subplots(figsize=(8, 5))
    sum_square_residuals = []
    sum_square_residuals_core = []
    sum_square_residuals_wings = []
    core_width = 20
    core = np.abs(vels + 33) < core_width
    ax.plot(vels,  vprofiles["oiii"]- vprofiles["ha"], label="Original", lw=2.5, color="b")
    for index, omega in enumerate(om_list):
        if omega is not None:
            p = TwoPhaseProfile(alpha=alpha, omega=omega)
            kernel = CustomKernel(
                discretize_model(pixel_profile, x_range, mode="oversample")
            )
            sprofile = convolve(vprofiles["oiii"], kernel)
        else:
            sprofile = vprofiles["oiii"]
        residuals = sprofile - vprofiles["ha"]
        if index % skip == 0:
            if index % (skip*2) == 0:
                label = f"$\omega = {omega:.2f}$"
                lw = 2.5
            else:
                label = None
                lw = 1.0
            ax.plot(vels, residuals, label=label, lw=lw)
        sum_square_residuals.append(np.sum(residuals**2))
        sum_square_residuals_core.append(np.sum(residuals[core]**2))
        sum_square_residuals_wings.append(np.sum(residuals[~core]**2))
    ax.axvline(-33, ls="dashed", color="k", lw=1)
    ax.axhline(0, ls="dashed", color="k", lw=1)
    ax.axvspan(-33 - core_width, -33 + core_width, color="b", alpha=0.1, zorder=-100)
    ax.legend(fontsize="small", title="Cool fraction")
    ax.set_ylim(None, 0.9)
    ax.set_xlabel("Heliocentric velocity, km/s")
    ax.set_ylabel(rf"$ \left\{{ I(\mathrm{{[O III]}}) \circ K({alpha:.2f}, \omega) \right\}} - I(\mathrm{{H\alpha}})$")
    figfile = "pn-ou5-2phase-a003-convolution-residuals.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


with sns.color_palette("colorblind", n_colors=3):
    fig, ax = plt.subplots(figsize=(8, 5))
    om_opt = np.interp(0.0, np.gradient(sum_square_residuals), om_list)
    label = f"All velocities, $\omega_\mathrm{{opt}} = {om_opt:.2f}$"
    line, = ax.plot(om_list, sum_square_residuals, label=label)
    c = line.get_color()
    ax.axvline(om_opt, linestyle="dotted", color=c, ymax=0.25)
    
    om_opt_core = np.interp(0.0, np.gradient(sum_square_residuals_core), om_list)
    label = f"Core only, $\omega_\mathrm{{opt}} = {om_opt_core:.2f}$"
    line, = ax.plot(om_list, sum_square_residuals_core, label=label)
    c = line.get_color()
    ax.axvline(om_opt_core, linestyle="dotted", color=c, ymax=0.25)

    om_opt_wings = np.interp(0.0, np.gradient(sum_square_residuals_wings), om_list)
    label = f"Wings only, $\omega_\mathrm{{opt}} = {om_opt_wings:.2f}$"
    line, = ax.plot(om_list, sum_square_residuals_wings, label=label)
    c = line.get_color()
    ax.axvline(om_opt_wings, linestyle="dotted", color=c, ymax=0.25)
    
    ax.set_ylim(0.0, 0.8)
    ax.set_xlabel(r"Cool fraction, $\omega$")
    ax.set_ylabel("Sum squared residuals")
    ax.legend(fontsize="small", title=rf"$\alpha = {alpha:.2f}$")
    figfile = "pn-ou5-2phase-a003-convolution-optimum.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


min(sum_square_residuals)

# #### $\alpha = 0.1$

om_list = np.linspace(0.0, 1.0, 41)[::-1]
skip = 5
alpha = 0.1
with sns.color_palette("rocket", 1 + len(om_list) // skip):
    fig, ax = plt.subplots(figsize=(8, 5))
    sum_square_residuals = []
    sum_square_residuals_core = []
    sum_square_residuals_wings = []
    core_width = 20
    core = np.abs(vels + 33) < core_width
    ax.plot(vels,  vprofiles["oiii"]- vprofiles["ha"], label="Original", lw=2.5, color="b")
    for index, omega in enumerate(om_list):
        if omega is not None:
            p = TwoPhaseProfile(alpha=alpha, omega=omega)
            kernel = CustomKernel(
                discretize_model(pixel_profile, x_range, mode="oversample")
            )
            sprofile = convolve(vprofiles["oiii"], kernel)
        else:
            sprofile = vprofiles["oiii"]
        residuals = sprofile - vprofiles["ha"]
        if index % skip == 0:
            if index % (skip*2) == 0:
                label = f"$\omega = {omega:.2f}$"
                lw = 2.5
            else:
                label = None
                lw = 1.0
            ax.plot(vels, residuals, label=label, lw=lw)
        sum_square_residuals.append(np.sum(residuals**2))
        sum_square_residuals_core.append(np.sum(residuals[core]**2))
        sum_square_residuals_wings.append(np.sum(residuals[~core]**2))
    ax.axvline(-33, ls="dashed", color="k", lw=1)
    ax.axhline(0, ls="dashed", color="k", lw=1)
    ax.axvspan(-33 - core_width, -33 + core_width, color="b", alpha=0.1, zorder=-100)
    ax.legend(fontsize="small", title="Cool fraction")
    ax.set_ylim(None, 0.9)
    ax.set_xlabel("Heliocentric velocity, km/s")
    ax.set_ylabel(rf"$ \left\{{ I(\mathrm{{[O III]}}) \circ K({alpha:.2f}, \omega) \right\}} - I(\mathrm{{H\alpha}})$")
    figfile = "pn-ou5-2phase-a010-convolution-residuals.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


with sns.color_palette("colorblind", n_colors=3):
    fig, ax = plt.subplots(figsize=(8, 5))
    om_opt = np.interp(0.0, np.gradient(sum_square_residuals), om_list)
    label = f"All velocities, $\omega_\mathrm{{opt}} = {om_opt:.2f}$"
    line, = ax.plot(om_list, sum_square_residuals, label=label)
    c = line.get_color()
    ax.axvline(om_opt, linestyle="dotted", color=c, ymax=0.25)
    
    om_opt_core = np.interp(0.0, np.gradient(sum_square_residuals_core), om_list)
    label = f"Core only, $\omega_\mathrm{{opt}} = {om_opt_core:.2f}$"
    line, = ax.plot(om_list, sum_square_residuals_core, label=label)
    c = line.get_color()
    ax.axvline(om_opt_core, linestyle="dotted", color=c, ymax=0.25)

    om_opt_wings = np.interp(0.0, np.gradient(sum_square_residuals_wings), om_list)
    label = f"Wings only, $\omega_\mathrm{{opt}} = {om_opt_wings:.2f}$"
    line, = ax.plot(om_list, sum_square_residuals_wings, label=label)
    c = line.get_color()
    ax.axvline(om_opt_wings, linestyle="dotted", color=c, ymax=0.25)
    
    ax.set_ylim(0.0, 0.8)
    ax.set_xlabel(r"Cool fraction, $\omega$")
    ax.set_ylabel("Sum squared residuals")
    ax.legend(fontsize="small", title=rf"$\alpha = {alpha:.2f}$")
    figfile = "pn-ou5-2phase-a010-convolution-optimum.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


min(sum_square_residuals)

# Optimum is very slightly better for $\alpha = 0.1$ than for $\alpha = 0.03$

# #### $\alpha = 0.3$

om_list = np.linspace(0.0, 1.0, 41)[::-1]
skip = 5
alpha = 0.3
with sns.color_palette("rocket", 1 + len(om_list) // skip):
    fig, ax = plt.subplots(figsize=(8, 5))
    sum_square_residuals = []
    sum_square_residuals_core = []
    sum_square_residuals_wings = []
    core_width = 20
    core = np.abs(vels + 33) < core_width
    ax.plot(vels,  vprofiles["oiii"]- vprofiles["ha"], label="Original", lw=2.5, color="b")
    for index, omega in enumerate(om_list):
        if omega is not None:
            p = TwoPhaseProfile(alpha=alpha, omega=omega)
            kernel = CustomKernel(
                discretize_model(pixel_profile, x_range, mode="oversample")
            )
            sprofile = convolve(vprofiles["oiii"], kernel)
        else:
            sprofile = vprofiles["oiii"]
        residuals = sprofile - vprofiles["ha"]
        if index % skip == 0:
            if index % (skip*2) == 0:
                label = f"$\omega = {omega:.2f}$"
                lw = 2.5
            else:
                label = None
                lw = 1.0
            ax.plot(vels, residuals, label=label, lw=lw)
        sum_square_residuals.append(np.sum(residuals**2))
        sum_square_residuals_core.append(np.sum(residuals[core]**2))
        sum_square_residuals_wings.append(np.sum(residuals[~core]**2))
    ax.axvline(-33, ls="dashed", color="k", lw=1)
    ax.axhline(0, ls="dashed", color="k", lw=1)
    ax.axvspan(-33 - core_width, -33 + core_width, color="b", alpha=0.1, zorder=-100)
    ax.legend(fontsize="small", title="Cool fraction")
    ax.set_ylim(None, 0.9)
    ax.set_xlabel("Heliocentric velocity, km/s")
    ax.set_ylabel(rf"$ \left\{{ I(\mathrm{{[O III]}}) \circ K({alpha:.2f}, \omega) \right\}} - I(\mathrm{{H\alpha}})$")
    figfile = "pn-ou5-2phase-a030-convolution-residuals.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


with sns.color_palette("colorblind", n_colors=3):
    fig, ax = plt.subplots(figsize=(8, 5))
    om_opt = np.interp(0.0, np.gradient(sum_square_residuals), om_list)
    label = f"All velocities, $\omega_\mathrm{{opt}} = {om_opt:.2f}$"
    line, = ax.plot(om_list, sum_square_residuals, label=label)
    c = line.get_color()
    ax.axvline(om_opt, linestyle="dotted", color=c, ymax=0.25)
    
    om_opt_core = np.interp(0.0, np.gradient(sum_square_residuals_core), om_list)
    label = f"Core only, $\omega_\mathrm{{opt}} = {om_opt_core:.2f}$"
    line, = ax.plot(om_list, sum_square_residuals_core, label=label)
    c = line.get_color()
    ax.axvline(om_opt_core, linestyle="dotted", color=c, ymax=0.25)

    om_opt_wings = np.interp(0.0, np.gradient(sum_square_residuals_wings), om_list)
    label = f"Wings only, $\omega_\mathrm{{opt}} = {om_opt_wings:.2f}$"
    line, = ax.plot(om_list, sum_square_residuals_wings, label=label)
    c = line.get_color()
    ax.axvline(om_opt_wings, linestyle="dotted", color=c, ymax=0.25)
    
    ax.set_ylim(0.0, 0.8)
    ax.set_xlabel(r"Cool fraction, $\omega$")
    ax.set_ylabel("Sum squared residuals")
    ax.legend(fontsize="small", title=rf"$\alpha = {alpha:.2f}$")
    figfile = "pn-ou5-2phase-a030-convolution-optimum.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


min(sum_square_residuals)

# Optimum is even better for $\alpha = 0.3$, but I am not sure I believe it



# #### $\alpha = 0.5$

om_list = np.linspace(0.0, 1.0, 41)[::-1]
skip = 5
alpha = 0.5
with sns.color_palette("rocket", 1 + len(om_list) // skip):
    fig, ax = plt.subplots(figsize=(8, 5))
    sum_square_residuals = []
    sum_square_residuals_core = []
    sum_square_residuals_wings = []
    core_width = 20
    core = np.abs(vels + 33) < core_width
    ax.plot(vels,  vprofiles["oiii"]- vprofiles["ha"], label="Original", lw=2.5, color="b")
    for index, omega in enumerate(om_list):
        if omega is not None:
            p = TwoPhaseProfile(alpha=alpha, omega=omega)
            kernel = CustomKernel(
                discretize_model(pixel_profile, x_range, mode="oversample")
            )
            sprofile = convolve(vprofiles["oiii"], kernel)
        else:
            sprofile = vprofiles["oiii"]
        residuals = sprofile - vprofiles["ha"]
        if index % skip == 0:
            if index % (skip*2) == 0:
                label = f"$\omega = {omega:.2f}$"
                lw = 2.5
            else:
                label = None
                lw = 1.0
            ax.plot(vels, residuals, label=label, lw=lw)
        sum_square_residuals.append(np.sum(residuals**2))
        sum_square_residuals_core.append(np.sum(residuals[core]**2))
        sum_square_residuals_wings.append(np.sum(residuals[~core]**2))
    ax.axvline(-33, ls="dashed", color="k", lw=1)
    ax.axhline(0, ls="dashed", color="k", lw=1)
    ax.axvspan(-33 - core_width, -33 + core_width, color="b", alpha=0.1, zorder=-100)
    ax.legend(fontsize="small", title="Cool fraction")
    ax.set_ylim(None, 0.9)
    ax.set_xlabel("Heliocentric velocity, km/s")
    ax.set_ylabel(rf"$ \left\{{ I(\mathrm{{[O III]}}) \circ K({alpha:.2f}, \omega) \right\}} - I(\mathrm{{H\alpha}})$")
    figfile = "pn-ou5-2phase-a050-convolution-residuals.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


with sns.color_palette("colorblind", n_colors=3):
    fig, ax = plt.subplots(figsize=(8, 5))
    om_opt = np.interp(0.0, np.gradient(sum_square_residuals), om_list)
    label = f"All velocities, $\omega_\mathrm{{opt}} = {om_opt:.2f}$"
    line, = ax.plot(om_list, sum_square_residuals, label=label)
    c = line.get_color()
    ax.axvline(om_opt, linestyle="dotted", color=c, ymax=0.25)
    
    om_opt_core = np.interp(0.0, np.gradient(sum_square_residuals_core), om_list)
    label = f"Core only, $\omega_\mathrm{{opt}} = {om_opt_core:.2f}$"
    line, = ax.plot(om_list, sum_square_residuals_core, label=label)
    c = line.get_color()
    ax.axvline(om_opt_core, linestyle="dotted", color=c, ymax=0.25)

    om_opt_wings = np.interp(0.0, np.gradient(sum_square_residuals_wings), om_list)
    label = f"Wings only, $\omega_\mathrm{{opt}} = {om_opt_wings:.2f}$"
    line, = ax.plot(om_list, sum_square_residuals_wings, label=label)
    c = line.get_color()
    ax.axvline(om_opt_wings, linestyle="dotted", color=c, ymax=0.25)
    
    ax.set_ylim(0.0, 0.8)
    ax.set_xlabel(r"Cool fraction, $\omega$")
    ax.set_ylabel("Sum squared residuals")
    ax.legend(fontsize="small", title=rf"$\alpha = {alpha:.2f}$")
    figfile = "pn-ou5-2phase-a050-convolution-optimum.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


min(sum_square_residuals)

# #### $\alpha = 0.7$
#
# This one is ridiculous really, but I just want the residuals to start increasing

om_list = np.linspace(0.0, 1.0, 41)[::-1]
skip = 5
alpha = 0.7
with sns.color_palette("rocket", 1 + len(om_list) // skip):
    fig, ax = plt.subplots(figsize=(8, 5))
    sum_square_residuals = []
    sum_square_residuals_core = []
    sum_square_residuals_wings = []
    core_width = 20
    core = np.abs(vels + 33) < core_width
    ax.plot(vels,  vprofiles["oiii"]- vprofiles["ha"], label="Original", lw=2.5, color="b")
    for index, omega in enumerate(om_list):
        if omega is not None:
            p = TwoPhaseProfile(alpha=alpha, omega=omega)
            kernel = CustomKernel(
                discretize_model(pixel_profile, x_range, mode="oversample")
            )
            sprofile = convolve(vprofiles["oiii"], kernel)
        else:
            sprofile = vprofiles["oiii"]
        residuals = sprofile - vprofiles["ha"]
        if index % skip == 0:
            if index % (skip*2) == 0:
                label = f"$\omega = {omega:.2f}$"
                lw = 2.5
            else:
                label = None
                lw = 1.0
            ax.plot(vels, residuals, label=label, lw=lw)
        sum_square_residuals.append(np.sum(residuals**2))
        sum_square_residuals_core.append(np.sum(residuals[core]**2))
        sum_square_residuals_wings.append(np.sum(residuals[~core]**2))
    ax.axvline(-33, ls="dashed", color="k", lw=1)
    ax.axhline(0, ls="dashed", color="k", lw=1)
    ax.axvspan(-33 - core_width, -33 + core_width, color="b", alpha=0.1, zorder=-100)
    ax.legend(fontsize="small", title="Cool fraction")
    ax.set_ylim(None, 0.9)
    ax.set_xlabel("Heliocentric velocity, km/s")
    ax.set_ylabel(rf"$ \left\{{ I(\mathrm{{[O III]}}) \circ K({alpha:.2f}, \omega) \right\}} - I(\mathrm{{H\alpha}})$")
    figfile = "pn-ou5-2phase-a070-convolution-residuals.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


with sns.color_palette("colorblind", n_colors=3):
    fig, ax = plt.subplots(figsize=(8, 5))
    om_opt = np.interp(0.0, np.gradient(sum_square_residuals), om_list)
    label = f"All velocities, $\omega_\mathrm{{opt}} = {om_opt:.2f}$"
    line, = ax.plot(om_list, sum_square_residuals, label=label)
    c = line.get_color()
    ax.axvline(om_opt, linestyle="dotted", color=c, ymax=0.25)
    
    om_opt_core = np.interp(0.0, np.gradient(sum_square_residuals_core), om_list)
    label = f"Core only, $\omega_\mathrm{{opt}} = {om_opt_core:.2f}$"
    line, = ax.plot(om_list, sum_square_residuals_core, label=label)
    c = line.get_color()
    ax.axvline(om_opt_core, linestyle="dotted", color=c, ymax=0.25)

    om_opt_wings = np.interp(0.0, np.gradient(sum_square_residuals_wings), om_list)
    label = f"Wings only, $\omega_\mathrm{{opt}} = {om_opt_wings:.2f}$"
    line, = ax.plot(om_list, sum_square_residuals_wings, label=label)
    c = line.get_color()
    ax.axvline(om_opt_wings, linestyle="dotted", color=c, ymax=0.25)
    
    ax.set_ylim(0.0, 0.8)
    ax.set_xlabel(r"Cool fraction, $\omega$")
    ax.set_ylabel("Sum squared residuals")
    ax.legend(fontsize="small", title=rf"$\alpha = {alpha:.2f}$")
    figfile = "pn-ou5-2phase-a070-convolution-optimum.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


min(sum_square_residuals)

# #### $\alpha = 0.9$
#
# This one is ridiculous really, but I just want the residuals to start increasing

om_list = np.linspace(0.0, 1.0, 41)[::-1]
skip = 5
alpha = 0.9
with sns.color_palette("rocket", 1 + len(om_list) // skip):
    fig, ax = plt.subplots(figsize=(8, 5))
    sum_square_residuals = []
    sum_square_residuals_core = []
    sum_square_residuals_wings = []
    core_width = 20
    core = np.abs(vels + 33) < core_width
    ax.plot(vels,  vprofiles["oiii"]- vprofiles["ha"], label="Original", lw=2.5, color="b")
    for index, omega in enumerate(om_list):
        if omega is not None:
            p = TwoPhaseProfile(alpha=alpha, omega=omega)
            kernel = CustomKernel(
                discretize_model(pixel_profile, x_range, mode="oversample")
            )
            sprofile = convolve(vprofiles["oiii"], kernel)
        else:
            sprofile = vprofiles["oiii"]
        residuals = sprofile - vprofiles["ha"]
        if index % skip == 0:
            if index % (skip*2) == 0:
                label = f"$\omega = {omega:.2f}$"
                lw = 2.5
            else:
                label = None
                lw = 1.0
            ax.plot(vels, residuals, label=label, lw=lw)
        sum_square_residuals.append(np.sum(residuals**2))
        sum_square_residuals_core.append(np.sum(residuals[core]**2))
        sum_square_residuals_wings.append(np.sum(residuals[~core]**2))
    ax.axvline(-33, ls="dashed", color="k", lw=1)
    ax.axhline(0, ls="dashed", color="k", lw=1)
    ax.axvspan(-33 - core_width, -33 + core_width, color="b", alpha=0.1, zorder=-100)
    ax.legend(fontsize="small", title="Cool fraction")
    ax.set_ylim(None, 0.9)
    ax.set_xlabel("Heliocentric velocity, km/s")
    ax.set_ylabel(rf"$ \left\{{ I(\mathrm{{[O III]}}) \circ K({alpha:.2f}, \omega) \right\}} - I(\mathrm{{H\alpha}})$")
    figfile = "pn-ou5-2phase-a090-convolution-residuals.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


with sns.color_palette("colorblind", n_colors=3):
    fig, ax = plt.subplots(figsize=(8, 5))
    om_opt = np.interp(0.0, np.gradient(sum_square_residuals), om_list)
    label = f"All velocities, $\omega_\mathrm{{opt}} = {om_opt:.2f}$"
    line, = ax.plot(om_list, sum_square_residuals, label=label)
    c = line.get_color()
    ax.axvline(om_opt, linestyle="dotted", color=c, ymax=0.25)
    
    om_opt_core = np.interp(0.0, np.gradient(sum_square_residuals_core), om_list)
    label = f"Core only, $\omega_\mathrm{{opt}} = {om_opt_core:.2f}$"
    line, = ax.plot(om_list, sum_square_residuals_core, label=label)
    c = line.get_color()
    ax.axvline(om_opt_core, linestyle="dotted", color=c, ymax=0.25)

    om_opt_wings = np.interp(0.0, np.gradient(sum_square_residuals_wings), om_list)
    label = f"Wings only, $\omega_\mathrm{{opt}} = {om_opt_wings:.2f}$"
    line, = ax.plot(om_list, sum_square_residuals_wings, label=label)
    c = line.get_color()
    ax.axvline(om_opt_wings, linestyle="dotted", color=c, ymax=0.25)
    
    ax.set_ylim(0.0, 0.8)
    ax.set_xlabel(r"Cool fraction, $\omega$")
    ax.set_ylabel("Sum squared residuals")
    ax.legend(fontsize="small", title=rf"$\alpha = {alpha:.2f}$")
    figfile = "pn-ou5-2phase-a090-convolution-optimum.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


min(sum_square_residuals)

# ### Variation with $\alpha$
#
# Taking the results from the previous section

Table(
    {
        "alpha": [0.03, 0.1, 0.3, 0.5, 0.7, 0.9],
        "om_opt": [0.21, 0.36, 0.48, 0.79, 1.0, 1.0],
        "om_opt_wings": [0.58, 0.63, 0.84, 1.0, 1.0, 1.0],        
        "chi^2": [0.153, 0.153, 0.150, 0.147, 0.154, 0.195],
    }
)

# So there really is no significant variation in the goodness of fit, except when we put the T of the cool component above 7000 K. 
#
# Formally, the best fit is for $\alpha = 0.5$, $\omega = 0.8$

# ### Illustrations of the convolved 2-phase profiles

sns.set_color_codes()
with sns.color_palette("dark"):

    fig, ax = plt.subplots(figsize=(8, 7))
    ds = "steps-mid"
    olw, hlw = 4, 2
    ocolor = (0.2, 0.5, 1.0)
    offset_step = 0.5
    offset = 0
    ax.plot(vels, offset + vprofiles["oiii"], color="k", lw=olw, ds=ds, label="[O III]")
    ax.plot(vels, offset + vprofiles["ha"], lw=hlw, color=ocolor, ds=ds, label=r"H$\alpha$")
    text = "No convolution"
    ax.text(-120, offset + 0.1, text, fontsize="small")
    ax.axhline(offset, ls="dashed", color="k", lw=1)

    for alpha, omega, label in [
        [0.3, 0.84, "A1"],
        [0.1, 0.63, "A2"],
        [0.03, 0.58, "A3"],
        [0.03, 0.18, "B"],
    ]:
            
        offset += offset_step
        p = TwoPhaseProfile(alpha=alpha, omega=omega)
        kernel = CustomKernel(
            discretize_model(pixel_profile, x_range, mode="oversample")
        )
        sprofile = convolve(vprofiles["oiii"], kernel)
        ax.plot(vels, offset + sprofile, color="k", lw=olw)
        ax.plot(vels, offset + vprofiles["ha"], ds=ds, lw=hlw, color=ocolor)
        text = (rf"$T_\mathrm{{cool}} = {1e4*alpha:.0f}$ K" 
                + "\n" + rf"$\omega = {omega:.2f}$")
        ax.text(-120, offset + 0.1, text, fontsize="small")
        ax.text(20, offset + 0.1, label, fontsize="small")
        ax.axhline(offset, ls="dashed", color="k", lw=1)
    
    ax.axvspan(-33 - core_width, -33 + core_width, color="b", alpha=0.1, zorder=-100)
    ax.axvline(-33, ls="dashed", color="k", lw=1)
    ax.set_xlabel("Heliocentric velocity, km/s")
    ax.legend()
    
    figfile = "pn-ou5-2phase-convolution-fits.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


# So these show convolved fits that are extremely similar to the Gaussian ones. The top profile is optimised or core and shows clear excess in the wings
#
# The other profiles (optimised for wings) show the degeneracy between $\alpha$ and $\omega$ (only depends on approx $\omega - \alpha$). 
#
# The irreducible residuals are again due to the expansion profile slightly more split for [O III] than for H alpha.

# ### Tangent-point 70 micron: convolved 2-phase profiles
#
# Repeat all the fits, but for the narrow 70 micron profile
#

# #### Same parameters as for whole nebula

sns.set_color_codes()
with sns.color_palette("dark"):

    fig, ax = plt.subplots(figsize=(8, 6))
    ds = "default"
    #ds = "steps-mid"
    olw, hlw = 2, 4
    ocolor = (0.2, 0.5, 1.0)
    offset_step = 0.5
    offset = 0
    ax.plot(vels, offset + vprofiles_tp70["ha"], lw=hlw, color="k", ds=ds, label=r"H$\alpha$")
    ax.plot(vels, offset + vprofiles_tp70["oiii"], color=ocolor, lw=olw, ds=ds, label="[O III]")
    text = "No convolution"
    ax.text(-120, offset + 0.1, text, fontsize="small")
    ax.axhline(offset, ls="dashed", color="k", lw=1)

    for alpha, omega, label in [
        [0.3, 0.84, "A1"],
        [0.1, 0.63, "A2"],
        [0.03, 0.58, "A3"],
        [0.03, 0.18, "B"],
    ]:
            
        offset += offset_step
        p = TwoPhaseProfile(alpha=alpha, omega=omega)
        kernel = CustomKernel(
            discretize_model(pixel_profile, x_range, mode="oversample")
        )
        sprofile = convolve(vprofiles_tp70["oiii"], kernel)
        ax.plot(vels, offset + vprofiles_tp70["ha"], lw=hlw, color="k")
        ax.plot(vels, offset + sprofile, color=ocolor, lw=olw)
        text = (rf"$\alpha = {alpha:.2f}$" 
                + "\n" + rf"$\omega = {omega:.2f}$")
        ax.text(-120, offset + 0.1, text, fontsize="small")
        ax.text(20, offset + 0.1, label, fontsize="small")
        ax.axhline(offset, ls="dashed", color="k", lw=1)
    
    ax.axvspan(-33 - core_width, -33 + core_width, color="b", alpha=0.1, zorder=-100)
    ax.axvline(-33, ls="dashed", color="k", lw=1)
    ax.set_xlabel("Heliocentric velocity, km/s")

    ax.legend()
    
    figfile = "pn-ou5-2phase-tp70-convolution-fits.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


# So this maybe shows a better fit for the low alpha case, but we need to plot the sum-square residuals

# #### Optimize $\omega$ for $\alpha = 0.03$
#
# We modify this in a few ways, compared with the whole-nebula fits. 
#
# * The core width is smaller $\pm 15$ because the lines are narrower
# * We exclude everything outside a window of $\pm 35$ because it is just noise
# * We calculate the rms noise using the profile outside the window, so we can evaluate the $\chi^2$ properly.
# * We estimate the number of degrees of freedom: $\nu = n - m$, where $n$ is the number of independent observations and $m = 2$ is the number of model parameters. So we can check that $\chi^2 / \nu$ is of order unity



vsys_tp70 = vmean_o_tp70
core_width = 10
full_width = 35
is_full_window = np.abs(vels - vsys_tp70) < full_width
is_core = (np.abs(vels - vsys_tp70) < core_width) & is_full_window
is_wings = (np.abs(vels - vsys_tp70) >= core_width) & is_full_window
noise_rms_ha = np.std(vprofiles_tp70["ha"][~is_full_window])
noise_rms_oiii = np.std(vprofiles_tp70["oiii"][~is_full_window])
noise_rms_diff = np.sqrt(noise_rms_ha**2 + noise_rms_oiii**2)
noise_rms_ha, noise_rms_oiii, noise_rms_diff

npix_full = np.sum(is_full_window)
npix_core = np.sum(is_core)
npix_wings = np.sum(is_wings)
npix_full, npix_core, npix_wings

# To evaluate the number of independent observations, we need to use the original pixel size. And maybe also account for the instrumental broadening. 
#
# The original pixels of 70 micron slit are 2.50 km/s for Ha and 2.58 for [O III]. We will use 2.5
#
# The rms instrumental width is also 2.5 for the 70 micron slit, so if we add these in quadrature it would give an effective resolution of 3.5
#
# **But, I don't think we should include the instrumental width, since the noise will not be broadened ny it probably**

3e5 * 0.0430814698338509 / 5006.8

# dv_orig = np.sqrt(2.5**2 + 2.5**2)
dv_orig = 2.5
dv_orig

dv_vels = np.diff(vels)[0]
dv_vels

nu_full = int(npix_full * dv_vels / dv_orig) - 2
nu_core = int(npix_core * dv_vels / dv_orig) - 2
nu_wings = int(npix_wings * dv_vels / dv_orig) - 2
nu_full, nu_core, nu_wings

# We now want to save all the chi-square results

chisq = {}

om_list = np.linspace(0.0, 1.0, 41)[::-1]
skip = 5
alpha = 0.03
with sns.color_palette("rocket", 1 + len(om_list) // skip):
    fig, ax = plt.subplots(figsize=(8, 5))
    sum_square_residuals = []
    sum_square_residuals_core = []
    sum_square_residuals_wings = []
    ax.plot(
        vels,  
        (vprofiles_tp70["oiii"]- vprofiles_tp70["ha"]) / noise_rms_diff, 
        label="Original", lw=2.5, color="b"
    )
    for index, omega in enumerate(om_list):
        p = TwoPhaseProfile(alpha=alpha, omega=omega)
        kernel = CustomKernel(
            discretize_model(pixel_profile, x_range, mode="oversample")
        )
        sprofile = convolve(vprofiles_tp70["oiii"], kernel)
        residuals = (sprofile - vprofiles_tp70["ha"]) / noise_rms_diff
        if index % skip == 0:
            if index % (skip*2) == 0:
                label = f"$\omega = {omega:.2f}$"
                lw = 2.5
            else:
                label = None
                lw = 1.0
            ax.plot(vels, residuals, label=label, lw=lw)
        sum_square_residuals.append(np.sum(residuals[is_full_window]**2))
        sum_square_residuals_core.append(np.sum(residuals[is_core]**2))
        sum_square_residuals_wings.append(np.sum(residuals[is_wings]**2))
    ax.axvline(vsys_tp70, ls="dashed", color="k", lw=1)
    ax.axhline(0, ls="dashed", color="k", lw=1)
    ax.axvspan(vsys_tp70 - full_width, vsys_tp70 + full_width, color="b", alpha=0.1, zorder=-100)
    chisq[alpha] = sum_square_residuals
    
    ax.legend(fontsize="small", title=rf"$\alpha = {alpha:.2f}$")
    ax.set_ylim(None, 10)
    ax.set_xlabel("Heliocentric velocity, km/s")
    ax.set_ylabel(r"$ (\mathrm{[O III]}^\bigstar - \mathrm{H\alpha})\ / \ \sigma_\mathrm{noise}$")
    figfile = "pn-ou5-2phase-tp70-a003-convolution-residuals.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


# #### Repeat for alpha = 0.1

om_list = np.linspace(0.0, 1.0, 41)[::-1]
skip = 5
alpha = 0.1
with sns.color_palette("rocket", 1 + len(om_list) // skip):
    fig, ax = plt.subplots(figsize=(8, 5))
    sum_square_residuals = []
    sum_square_residuals_core = []
    sum_square_residuals_wings = []
    ax.plot(
        vels,  
        (vprofiles_tp70["oiii"]- vprofiles_tp70["ha"]) / noise_rms_diff, 
        label="Original", lw=2.5, color="b", ds="steps-mid",
    )
    for index, omega in enumerate(om_list):
        p = TwoPhaseProfile(alpha=alpha, omega=omega)
        kernel = CustomKernel(
            discretize_model(pixel_profile, x_range, mode="oversample")
        )
        sprofile = convolve(vprofiles_tp70["oiii"], kernel)
        residuals = (sprofile - vprofiles_tp70["ha"]) / noise_rms_diff
        if index % skip == 0:
            if index % (skip*2) == 0:
                label = f"$\omega = {omega:.2f}$"
                lw = 2.5
            else:
                label = None
                lw = 1.0
            ax.plot(vels, residuals, label=label, lw=lw)
        sum_square_residuals.append(np.sum(residuals[is_full_window]**2))
        sum_square_residuals_core.append(np.sum(residuals[is_core]**2))
        sum_square_residuals_wings.append(np.sum(residuals[is_wings]**2))
    ax.axvline(vsys_tp70, ls="dashed", color="k", lw=1)
    ax.axhline(0, ls="dashed", color="k", lw=1)
    ax.axvspan(vsys_tp70 - full_width, vsys_tp70 + full_width, color="b", alpha=0.1, zorder=-100)
    chisq[alpha] = sum_square_residuals
    
    ax.legend(fontsize="small", title=rf"$T_\mathrm{{cool}} = {10000*alpha:.0f}$ K")
    ax.set_ylim(None, 10)
    ax.set_xlabel("Heliocentric velocity, km/s")
    ax.set_ylabel(r"$ (\mathrm{[O III]}^\bigstar - \mathrm{H\alpha})\ / \ s_\mathrm{noise}$")
    figfile = "pn-ou5-2phase-tp70-a010-convolution-residuals.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


# #### Repeat for alpha = 0.3

om_list = np.linspace(0.0, 1.0, 41)[::-1]
skip = 5
alpha = 0.3
with sns.color_palette("rocket", 1 + len(om_list) // skip):
    fig, ax = plt.subplots(figsize=(8, 5))
    sum_square_residuals = []
    sum_square_residuals_core = []
    sum_square_residuals_wings = []
    ax.plot(
        vels,  
        (vprofiles_tp70["oiii"]- vprofiles_tp70["ha"]) / noise_rms_diff, 
        label="Original", lw=2.5, color="b"
    )
    for index, omega in enumerate(om_list):
        p = TwoPhaseProfile(alpha=alpha, omega=omega)
        kernel = CustomKernel(
            discretize_model(pixel_profile, x_range, mode="oversample")
        )
        sprofile = convolve(vprofiles_tp70["oiii"], kernel)
        residuals = (sprofile - vprofiles_tp70["ha"]) / noise_rms_diff
        if index % skip == 0:
            if index % (skip*2) == 0:
                label = f"$\omega = {omega:.2f}$"
                lw = 2.5
            else:
                label = None
                lw = 1.0
            ax.plot(vels, residuals, label=label, lw=lw)
        sum_square_residuals.append(np.sum(residuals[is_full_window]**2))
        sum_square_residuals_core.append(np.sum(residuals[is_core]**2))
        sum_square_residuals_wings.append(np.sum(residuals[is_wings]**2))
    ax.axvline(vsys_tp70, ls="dashed", color="k", lw=1)
    ax.axhline(0, ls="dashed", color="k", lw=1)
    ax.axvspan(vsys_tp70 - full_width, vsys_tp70 + full_width, color="b", alpha=0.1, zorder=-100)
    chisq[alpha] = sum_square_residuals
    
    ax.legend(fontsize="small", title=rf"$\alpha = {alpha:.2f}$")
    ax.set_ylim(None, 10)
    ax.set_xlabel("Heliocentric velocity, km/s")
    ax.set_ylabel(r"$ (\mathrm{[O III]}^\bigstar - \mathrm{H\alpha})\ / \ s_\mathrm{noise}$")
    figfile = "pn-ou5-2phase-tp70-a030-convolution-residuals.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


# #### Repeat for alpha = 0.5

om_list = np.linspace(0.0, 1.0, 41)[::-1]
skip = 5
alpha = 0.5
with sns.color_palette("rocket", 1 + len(om_list) // skip):
    fig, ax = plt.subplots(figsize=(8, 5))
    sum_square_residuals = []
    sum_square_residuals_core = []
    sum_square_residuals_wings = []
    ax.plot(
        vels,  
        (vprofiles_tp70["oiii"]- vprofiles_tp70["ha"]) / noise_rms_diff, 
        label="Original", lw=2.5, color="b"
    )
    for index, omega in enumerate(om_list):
        p = TwoPhaseProfile(alpha=alpha, omega=omega)
        kernel = CustomKernel(
            discretize_model(pixel_profile, x_range, mode="oversample")
        )
        sprofile = convolve(vprofiles_tp70["oiii"], kernel)
        residuals = (sprofile - vprofiles_tp70["ha"]) / noise_rms_diff
        if index % skip == 0:
            if index % (skip*2) == 0:
                label = f"$\omega = {omega:.2f}$"
                lw = 2.5
            else:
                label = None
                lw = 1.0
            ax.plot(vels, residuals, label=label, lw=lw)
        sum_square_residuals.append(np.sum(residuals[is_full_window]**2))
        sum_square_residuals_core.append(np.sum(residuals[is_core]**2))
        sum_square_residuals_wings.append(np.sum(residuals[is_wings]**2))
    ax.axvline(vsys_tp70, ls="dashed", color="k", lw=1)
    ax.axhline(0, ls="dashed", color="k", lw=1)
    ax.axvspan(vsys_tp70 - full_width, vsys_tp70 + full_width, color="b", alpha=0.1, zorder=-100)
    chisq[alpha] = sum_square_residuals
    
    ax.legend(fontsize="small", title=rf"$\alpha = {alpha:.2f}$")
    ax.set_ylim(None, 10)
    ax.set_xlabel("Heliocentric velocity, km/s")
    ax.set_ylabel(r"$ (\mathrm{[O III]}^\bigstar - \mathrm{H\alpha})\ / \ \sigma_\mathrm{noise}$")
    figfile = "pn-ou5-2phase-tp70-a050-convolution-residuals.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")




# #### Repeat for alpha = 0.01

om_list = np.linspace(0.0, 1.0, 41)[::-1]
skip = 5
alpha = 0.01
with sns.color_palette("rocket", 1 + len(om_list) // skip):
    fig, ax = plt.subplots(figsize=(8, 5))
    sum_square_residuals = []
    sum_square_residuals_core = []
    sum_square_residuals_wings = []
    ax.plot(
        vels,  
        (vprofiles_tp70["oiii"]- vprofiles_tp70["ha"]) / noise_rms_diff, 
        label="Original", lw=2.5, color="b"
    )
    for index, omega in enumerate(om_list):
        p = TwoPhaseProfile(alpha=alpha, omega=omega)
        kernel = CustomKernel(
            discretize_model(pixel_profile, x_range, mode="oversample")
        )
        sprofile = convolve(vprofiles_tp70["oiii"], kernel)
        residuals = (sprofile - vprofiles_tp70["ha"]) / noise_rms_diff
        if index % skip == 0:
            if index % (skip*2) == 0:
                label = f"$\omega = {omega:.2f}$"
                lw = 2.5
            else:
                label = None
                lw = 1.0
            ax.plot(vels, residuals, label=label, lw=lw)
        sum_square_residuals.append(np.sum(residuals[is_full_window]**2))
        sum_square_residuals_core.append(np.sum(residuals[is_core]**2))
        sum_square_residuals_wings.append(np.sum(residuals[is_wings]**2))
    ax.axvline(vsys_tp70, ls="dashed", color="k", lw=1)
    ax.axhline(0, ls="dashed", color="k", lw=1)
    ax.axvspan(vsys_tp70 - full_width, vsys_tp70 + full_width, color="b", alpha=0.1, zorder=-100)
    chisq[alpha] = sum_square_residuals
    
    ax.legend(fontsize="small", title=rf"$\alpha = {alpha:.2f}$")
    ax.set_ylim(None, 10)
    ax.set_xlabel("Heliocentric velocity, km/s")
    ax.set_ylabel(r"$ (\mathrm{[O III]}^\bigstar - \mathrm{H\alpha})\ / \ \sigma_\mathrm{noise}$")
    figfile = "pn-ou5-2phase-tp70-a001-convolution-residuals.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")




# #### Chi-square curves for all the alphas

alphas = sorted(chisq)
with sns.color_palette("mako_r", n_colors=len(alphas)):
        
    fig, ax = plt.subplots(figsize=(8, 5))
    for alpha in alphas:
        # _chisq = chisq[alpha] / (npix_full - 2)
        _chisq = np.array(chisq[alpha]) / nu_full
        
        om_opt = np.interp(0.0, np.gradient(_chisq), om_list)
        label = rf"$T_\mathrm{{cool}} = {1e4*alpha:.0f}$ K, $\omega_\mathrm{{best}} = {om_opt:.2f}$"
        line, = ax.plot(om_list, _chisq, label=label)
        c = line.get_color()
        # ax.axvline(om_opt, linestyle="dotted", color=c, ymax=0.25)
   
    ax.axhline(1, linestyle="dashed", color="r")
    ax.set_ylim(0.0, 5)
    ax.set_xlabel(r"Cool fraction, $\omega$")
    ax.set_ylabel(r"Goodness of fit: $\chi^2 / \nu$")
    ax.legend(fontsize="small", title="2-phase model")
    figfile = "pn-ou5-2phase-tp70-multi-convolution-optimum.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


#

# #### Profile plots for the best-fit models

sns.set_color_codes()
with sns.color_palette("dark"):

    fig, ax = plt.subplots(figsize=(8, 8))
    ds = "steps-mid"
    olw, hlw = 2, 4
    ocolor = (0.2, 0.5, 1.0)
    offset_step = 0.5
    offset = 0
    ax.plot(vels, offset + vprofiles_tp70["ha"], lw=hlw, color="k", ds=ds, label=r"H$\alpha$")
    ax.plot(vels, offset + vprofiles_tp70["oiii"], color=ocolor, lw=olw, ds=ds, label="[O III]")
    text = "No convolution"
    ax.text(-120, offset + 0.1, text, fontsize="small")
    ax.axhline(offset, ls="dashed", color="k", lw=1)

    ds = "default"
    dsha = "steps-mid"
    for alpha, omega, label in [
        [0.3, 1.0, r"$T_\mathrm{cool} = 3000$ K"],
        [0.1, 0.80, r"$T_\mathrm{cool} = 1000$ K"],
        [0.03, 0.71, r"$T_\mathrm{cool} = 300$ K"],
        [0.01, 0.69, r"$T_\mathrm{cool} = 100$ K"],
    ]:
            
        offset += offset_step
        p = TwoPhaseProfile(alpha=alpha, omega=omega)
        kernel = CustomKernel(
            discretize_model(pixel_profile, x_range, mode="oversample")
        )
        sprofile = convolve(vprofiles_tp70["oiii"], kernel)
        ax.plot(vels, offset + vprofiles_tp70["ha"], ds=dsha, lw=hlw, color="k")
        ax.plot(vels, offset + sprofile, color=ocolor, lw=olw)
        text = (label + "\n" + rf"$\omega = {omega:.2f}$")
        ax.text(-120, offset + 0.1, text, fontsize="small")
        # ax.text(0, offset + 0.15, label, fontsize="small")
        ax.axhline(offset, ls="dashed", color="k", lw=1)
    
    ax.axvspan(vmean_o_tp70 - full_width, vmean_o_tp70 + full_width, color="b", alpha=0.1, zorder=-100)
    ax.axvline(vmean_o_tp70, ls="dashed", color="k", lw=1)
    ax.set_xlabel("Heliocentric velocity, km/s")

    ax.legend()
    
    figfile = "pn-ou5-2phase-tp70-convolution-best-fits.pdf"
    fig.savefig(figfile, bbox_inches="tight")
    fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")








# + [markdown] editable=true slideshow={"slide_type": ""}
# # Some comments on the results
#
# Components
#
# ## Inner lobes
#
# Red component is brighter than blue.
#
# ## Outer lobes
#
# Bend towards blue on both sides (N and S)
#
# ## Equatorial high velocity wings
# More obvious in oiii. 
#
# ## Polar knots
#
#
#
# -

# That seems to have worked fine.  Now we will try it with a pre-defined common grid. Say, 1 km/s and 0.2 arcsec.

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

# # Things that did not work



# Make a `Fittable1DModel` for the profile, so we can then convert it into a convolution kernel. We can't just use the decorator since that makes a `FittableModel` instead, which does not work with the convolution `Model1DKernel`. We also need to worry about the pixel size, since the convolution is always done in pixel units

# +
from astropy.modeling import Fittable1DModel, Parameter

class TwoPhaseModel(Fittable1DModel):
    alpha = Parameter()
    omega = Parameter()
    Twarm = Parameter()

    @staticmethod
    def evaluate(x, alpha, omega, Twarm):
        return two_phase_profile(x, alpha, omega, Twarm)

    @staticmethod
    def fit_deriv(x, alpha, omega, Twarm):
        return None
# -

# This does not seem to work

# +
# model = TwoPhaseModel(alpha=0.1, omega=0.5, Twarm=1)

# +
# kernel = Model1DKernel(model, x_size=101, mode="oversample")
