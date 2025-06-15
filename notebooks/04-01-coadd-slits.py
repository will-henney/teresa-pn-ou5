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

# + jupyter={"source_hidden": true}
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

# And do the spatial profiles along the slit

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
    line, = ax.plot(pos, profile + offset, label=line_label, ds="steps-mid")
    ax.text(-30, offset + 0.15, labels[line_label], color=line.get_color())
    ax.axhline(offset, linestyle="dashed", c="k", lw=1,)
    offset += 1
ax.axvline(0.0, linestyle="dashed", c="k", lw=1,)
#ax.legend(ncol=2)
ax.set(
    xlabel="Displacement along slit, arcsec",
)
figfile = "ou5-coadd-spatial-profiles-1d.pdf"
fig.savefig(figfile)
fig.savefig(figfile.replace(".pdf", ".jpg"))
...;

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
vprofiles = {}
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
    vprofiles[line_label] = profile
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
fig.savefig(figfile)
fig.savefig(figfile.replace(".pdf", ".jpg"))
...;
# -

# ### Convolve [O III] with Gaussian to match Ha

from astropy.convolution import convolve, Gaussian1DKernel

# +
fig, ax = plt.subplots()
boost = np.sum(vprofiles["ha"]) / np.sum(vprofiles["oiii"])
ax.plot(vels, vprofiles["ha"], lw=4, color="k")
ax.plot(vels, boost * vprofiles["oiii"])

sprofile = convolve(vprofiles["oiii"], Gaussian1DKernel(stddev=3.8))
ax.plot(vels, boost * sprofile)

# -

ss_list = np.linspace(0.0, 8.0, 25)
skip = 3
sns.set_palette("rocket", 1 + len(ss_list) // skip)
fig, ax = plt.subplots()
boost = np.sum(vprofiles["ha"]) / np.sum(vprofiles["oiii"])
sum_square_residuals = []
sum_square_residuals_core = []
sum_square_residuals_wings = []
core_width = 18
core = np.abs(vels + 33) < core_width
for index, ss in enumerate(ss_list):
    if ss > 0:
        sprofile = convolve(vprofiles["oiii"], Gaussian1DKernel(stddev=ss))
    else:
        sprofile = vprofiles["oiii"]
    residuals = boost * sprofile - vprofiles["ha"]
    if index % skip == 0:
        if index % skip*2 == 0:
            label = f"sig = {ss:.0f}"
        else:
            label = None
        ax.plot(vels, residuals, label=label)
    sum_square_residuals.append(np.sum(residuals**2))
    sum_square_residuals_core.append(np.sum(residuals[core]**2))
    sum_square_residuals_wings.append(np.sum(residuals[~core]**2))
ax.axvline(-33, ls="dashed", color="k", lw=1)
ax.axhline(0, ls="dashed", color="k", lw=1)
ax.axvspan(-33 - core_width, -33 + core_width, color="k", alpha=0.1)
ax.legend(ncol=3, fontsize="x-small")
ax.set_ylim(None, 0.8)

sns.color_palette()

dv = np.diff(vels)[0]

with sns.color_palette("tab10", n_colors=3):
    fig, ax = plt.subplots()
    line, = ax.plot(dv * ss_list, sum_square_residuals, label="all velocities")
    ss_opt = np.interp(0.0, np.gradient(sum_square_residuals), ss_list)
    c = line.get_color()
    ax.axvline(ss_opt * dv, linestyle="dotted", color=c, ymin=0.15, ymax=0.3)
    
    line, = ax.plot(dv * ss_list, sum_square_residuals_core, label="core only")
    ss_opt_core = np.interp(0.0, np.gradient(sum_square_residuals_core), ss_list)
    c = line.get_color()
    ax.axvline(ss_opt_core * dv, linestyle="dotted", color=c, ymin=0.1, ymax=0.25)

    line, = ax.plot(dv * ss_list, sum_square_residuals_wings, label="wings only")
    ss_opt_wings = np.interp(0.0, np.gradient(sum_square_residuals_wings), ss_list)
    c = line.get_color()
    ax.axvline(ss_opt_wings * dv, linestyle="dotted", color=c, ymax=0.1)
    
    ax.set_ylim(0.0, None)
    ax.set_xlabel(r"Extra broadening, $\sigma$, km/s")
    ax.set_ylabel("Sum squared residuals")
    ax.legend(fontsize="x-small")

np.diff(sum_square_residuals)

np.interp(0.0, np.gradient(sum_square_residuals_wings), ss_list)

# +
# ax.axvline?
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


