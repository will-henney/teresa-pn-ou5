# -*- coding: utf-8 -*-
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

from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.convolution import Gaussian2DKernel, convolve_fft
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("talk")

dpath = Path.cwd().parent / "data"
pvpath2 = dpath / "pv-common"

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
    ("N outer", [9.0, 18.0]),
    ("N inner", [2.0, 6.0]),
    ("Core", [-1.5, 1.5]),
    ("S inner", [-6.0, -2.0]),    
    ("S outer", [-18.0, -9.0]),
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
    axtitle = f"  {pos_label}: [${s1:+.1f}$, ${s2:+.1f}$]"
    ax.set_title(axtitle, loc="left", y=0.7)
axes[3].legend(ncol=2)
axes[-1].set(
    xlabel="Heliocentric velocity, km/s",
)
figfile = "ou5-coadd-1dspec-all.pdf"
fig.savefig(figfile)
fig.savefig(figfile.replace(".pdf", ".jpg"))
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
        ax.plot(
            vels, 
            fitted_model(vels), 
            linestyle="dashed", 
            lw=2, 
            c=c,
            label=f"{line_label} fit",
        )
        if "knot" in pos_label:
            # special case of 1 component, therefore not compound model
            fitted_model = [fitted_model]
        for component in fitted_model:
            mark_component(component, c, ax)
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
    axtitle = f"  {pos_label}: [${s1:+.1f}$, ${s2:+.1f}$]"
    ax.set_title(axtitle, loc="left", y=0.7)
axes[3].legend(ncol=2)
axes[-1].set(
    xlabel="Heliocentric velocity, km/s",
)
figfile = "ou5-coadd-1dspec-ha-oiii.pdf"
fig.savefig(figfile)
fig.savefig(figfile.replace(".pdf", ".jpg"))
...;
# -

# Now zoom the y axis, to look at the wings, but just for the inner regions:

cpos = {
    k: v for k, v in positions
    if np.abs(v).min() < 4
}
cpos


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
    axtitle = f"  {pos_label}: [${s1:+.1f}$, ${s2:+.1f}$]"
    ax.set_title(axtitle, loc="left", y=0.7)
    ax.set(ylim=[-0.04, 0.25])
axes[1].legend(ncol=1)
axes[-1].set(
    xlabel="Heliocentric velocity, km/s",
)
figfile = "ou5-coadd-1dspec-wings.pdf"
fig.savefig(figfile)
fig.savefig(figfile.replace(".pdf", ".jpg"))
...;
# -

import pandas as pd

m = gfits[("oiii", "Core")]
dict(zip(m.param_names, m.parameters))

pd.set_option('display.precision', 2)
df = pd.DataFrame(
    {k: dict(zip(m.param_names, m.parameters)) for k, m in gfits.items()}
).T
df.style.format(na_rep='—')

# ## Compare with 2D spectra
#
# Repeat from previous notebook, but add positions

positions

# +
fig, ax = plt.subplots(figsize=(8, 10), subplot_kw=dict(projection=w))

vsys = -33
v1, v2 = vsys - 100, vsys + 100
s1, s2 = -35, 35

kernel = Gaussian2DKernel(x_stddev=2.0)
hdu = linehdus["oiii"]
w = WCS(hdu.header)
xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
y1, y2 = [int(_) for _ in ylims]
x1, x2 = [int(_) for _ in xlims]
im = hdu.data
ax.imshow(im, vmin=-0.1, vmax=1.0)
ims = convolve_fft(im, kernel)
ax.contour(
    ims, 
    levels=[0.01, 0.02, 0.04, 0.08, 0.16], 
    colors="w",
    linewidths=[0.5, 1.0, 1.5, 2.0, 2.5],
)
x0, y0 = w.world_to_pixel_values(vsys, 0.0)
ax.axhline(y0, color="orange", ls="dashed", lw=4, alpha=0.3)
ax.axvline(x0, color="orange", ls="dashed", lw=4, alpha=0.3)


# Add markers for the extracted regions
xp = 20.0
trw = ax.get_transform("world")
for plabel, ypp in positions:
    ax.plot([xp, xp], ypp, c="w", lw=4, transform=trw)
    for yp in ypp:
        ax.plot([xp, xp + 4], [yp, yp], c="w", lw=4, transform=trw)
    ax.text(xp + 8, np.mean(ypp), plabel, color="w", ha="left", va="center", transform=trw)

    
# Add markers for the Gaussian components
for plabel, ypp in positions:
    model = gfits[("oiii", plabel)]
    vmeans = [
        param 
        for name, param in zip(model.param_names, model.parameters)
        if name.startswith("mean")
    ]
    for vmean in vmeans:
        ax.scatter(
            vmean, 
            np.mean(ypp), 
            marker="o", 
            s=100, 
            c='orange',
            ec="k",
            alpha=1.0, 
            transform=trw,
            zorder=100,
        )
ax.set(xlim=xlims, ylim=ylims)
ax.set_title("[O III] 5007", pad=16)
figfile = "ou5-coadd-2dspec-oiii.pdf"
fig.savefig(figfile)
fig.savefig(figfile.replace(".pdf", ".jpg"))
...;
# -

df = df.assign(dv=df.mean_1 - df.mean_0)

df.style.format(na_rep='—')

# ## Try fitting Gaussians to all the rows
#
# Use bins of size 1 arcsec

fine_pix = 1.0
fine_positions = np.arange(-28, 29) * fine_pix
fine_positions

# +
sys = -33
v1, v2 = vsys - 100, vsys + 100
fitter = fitting.LevMarLSQFitter()
finefits = {}
finesingles = {}
fineprofiles = {}
hdu = linehdus["oiii"]
w = WCS(hdu.header)
ns, nv = hdu.data.shape

for finepos in fine_positions:
    s1, s2 = finepos - 0.5 * fine_pix, finepos + 0.5 * fine_pix 
    xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
    x1, x2 = [int(_) for _ in xlims]
    y1, y2 = [int(_) for _ in ylims]
    spec = hdu.data[y1:y2+1, x1:x2+1].mean(axis=0)
    vels, _ = w.pixel_to_world_values(np.arange(nv), [0]*nv)
    vels = vels[x1:x2+1]
    sm = spec.max()
    g1 = models.Gaussian1D(amplitude=sm, mean=-50, stddev=10.0)
    g1.stddev.bounds = (5.0, 10.0)
    g2 = models.Gaussian1D(amplitude=sm, mean=-20, stddev=10.0)
    g2.stddev.bounds = (5.0, 10.0)
    init_model = g1 + g2
    fitted_model = fitter(init_model, vels, spec)
    gs = models.Gaussian1D(amplitude=sm, mean=sys, stddev=10.0)
    fitted_single = fitter(gs, vels, spec)
    finefits[finepos] = fitted_model
    finesingles[finepos] = fitted_single
    fineprofiles[finepos] = {
        "v": vels,
        "spec": spec,
        "model": fitted_model(vels),
        "g1": fitted_model[0](vels),
        "g2": fitted_model[1](vels),
        "gs": fitted_single(vels),
    }
...;
# -
fdf = pd.DataFrame(
    {k: dict(zip(m.param_names, m.parameters)) for k, m in finefits.items()}
).T
fdf.style.format(na_rep='—')

fdfs = pd.DataFrame(
    {k: dict(zip(m.param_names, m.parameters)) for k, m in finesingles.items()}
).T
fdfs.style.format(na_rep='—')

fdf = fdf.join(fdfs)

# Calculate weighted mean of the velocities

fdf = fdf.assign(
    wmean = lambda x: (
        x.mean_0 * x.stddev_0 * x.amplitude_0
        + x.mean_1 * x.stddev_1 * x.amplitude_1
    ) / (x.stddev_0 * x.amplitude_0 + x.stddev_1 * x.amplitude_1)
)

fdf[["mean_0", "mean_1", "mean", "wmean"]].plot()

fdf[["amplitude_0", "amplitude_1", "amplitude"]].plot().set_yscale("log")


fdf[["stddev_0", "stddev_1", "stddev"]].plot().set_ylim(0.0, 25.0)


fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(
    fdf["mean_0"], 
    fdf.index,
    s=100 * fdf["amplitude_0"] * fdf["stddev_0"] / 7,    
)
ax.scatter(
    fdf["mean_1"], 
    fdf.index,
    s=100 * fdf["amplitude_1"] * fdf["stddev_1"] / 7,    
)
# ax.scatter(
#     fdf["mean"], 
#     fdf.index,
#     s=100 * fdf["amplitude"] * fdf["stddev"] / 14,    
# )
ax.plot(
    fdf["wmean"], 
    fdf.index,
    color="k",
#    s=100 * fdf["amplitude"] * fdf["stddev"] / 14,    
)
ax.axhline(0.0, color="k", ls="dashed", lw=2, alpha=0.3)
ax.axvline(vsys, color="k", ls="dashed", lw=2, alpha=0.3)
ax.set(
    xlabel="Heliocentric velocity, km/s",
    ylabel="Offset from star, arcsec",
)
ax.set_title("[O III] 5007 Gaussian components", pad=12)
figfile = "ou5-coadd-pv-oiii-gaussians.pdf"
fig.savefig(figfile)
fig.savefig(figfile.replace(".pdf", ".jpg"))
...;

# ## Look at the individua profile fits in more detail                                                                                                                        
# We want to see whether it makes sense to fit two gaussians in the outer lobes

fig, ax = plt.subplots(figsize=(8, 8))
for pos in fine_positions:
    data = fineprofiles[pos]
    v = data["v"]
    norm = np.max(data["spec"])
    g1 = np.where(data["g1"] > 0.02 * norm, data["g1"], np.nan)
    g2 = np.where(data["g2"] > 0.02 * norm, data["g2"], np.nan)
    ax.fill_between(v, pos + 4 * g1 / norm, pos, color="b", lw=0.5, alpha=0.3)
    ax.fill_between(v, pos + 4 * g2 / norm, pos, color="r", lw=0.5, alpha=0.3)
    ax.plot(v, pos + 4 * data["spec"] / norm, color="k", alpha=0.4, drawstyle="steps-mid")
ax.set(
    xlim=[-95, 25],
)


# Repeat but for northern outer lobe

fig, ax = plt.subplots(figsize=(8, 8))
for pos in fine_positions:
    if pos < 12.0:
        continue
    data = fineprofiles[pos]
    v = data["v"]
    norm = np.max(data["spec"])
    scale = 1.0
    g1 = np.where(data["g1"] > 0.02 * norm, data["g1"], np.nan)
    g2 = np.where(data["g2"] > 0.02 * norm, data["g2"], np.nan)
    ax.fill_between(v, pos + scale * g1 / norm, pos, color="b", lw=0.5, alpha=0.3)
    ax.fill_between(v, pos + scale * g2 / norm, pos, color="r", lw=0.5, alpha=0.3)
    if abs(pos) > 20.0:
        gs = np.where(data["gs"] > 0.01 * norm, data["gs"], np.nan)
        ax.plot(v, pos + scale * gs / norm, color="g", lw=1.3)
    ax.plot(v, pos + scale * data["spec"] / norm, color="k", alpha=0.4, drawstyle="steps-mid")


# And for southern lobe

fig, ax = plt.subplots(figsize=(8, 8))
for pos in fine_positions:
    if pos > -12.0:
        continue
    data = fineprofiles[pos]
    v = data["v"]
    norm = np.max(data["spec"])
    scale = 1.0
    g1 = np.where(data["g1"] > 0.02 * norm, data["g1"], np.nan)
    g2 = np.where(data["g2"] > 0.02 * norm, data["g2"], np.nan)
    ax.fill_between(v, pos + scale * g1 / norm, pos, color="b", lw=0.5, alpha=0.3)
    ax.fill_between(v, pos + scale * g2 / norm, pos, color="r", lw=0.5, alpha=0.3)
    if abs(pos) > 20.0:
        gs = np.where(data["gs"] > 0.01 * norm, data["gs"], np.nan)
        ax.plot(v, pos + scale * gs / norm, color="g", lw=1.3)
    ax.plot(v, pos + scale * data["spec"] / norm, color="k", alpha=0.4, drawstyle="steps-mid")

fig, ax = plt.subplots(figsize=(8, 8))
for pos in fine_positions:
    if abs(pos) > 12.0:
        continue
    data = fineprofiles[pos]
    v = data["v"]
    norm = np.max(data["spec"])
    scale = 1.0
    g1 = np.where(data["g1"] > 0.02 * norm, data["g1"], np.nan)
    g2 = np.where(data["g2"] > 0.02 * norm, data["g2"], np.nan)
    ax.fill_between(v, pos + scale * g1 / norm, pos, color="b", lw=0.5, alpha=0.3)
    ax.fill_between(v, pos + scale * g2 / norm, pos, color="r", lw=0.5, alpha=0.3)
    if abs(pos) > 20.0:
        gs = np.where(data["gs"] > 0.01 * norm, data["gs"], np.nan)
        ax.plot(v, pos + scale * gs / norm, color="g", lw=1.3)
    ax.plot(v, pos + scale * data["spec"] / norm, color="k", alpha=0.4, drawstyle="steps-mid")



