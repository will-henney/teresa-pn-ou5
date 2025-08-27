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
    ("Rings", [-1.5, 1.5]),
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
        # Shift Ha to reduce offset with oiii
        vfix = 1.1 if line_label == "ha" else 0.0
        w = WCS(hdu.header)
        ns, nv = hdu.data.shape
        xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
        x1, x2 = [int(_) for _ in xlims]
        y1, y2 = [int(_) for _ in ylims]
        spec = hdu.data[y1:y2, x1:x2].mean(axis=0)
        vels, _ = w.pixel_to_world_values(np.arange(nv), [0]*nv)
        vels = vels[x1:x2] + vfix
        dataline, = ax.plot(vels, spec, label=line_label, ds="steps-mid")
        c = dataline.get_color()
        
        # Fit two Gaussians
        mask1 = vels < vsys
        mask2 = ~mask1
        sm1 = spec[mask1].max()
        vm1 = vels[spec[mask1].argmax()]
        sm2 = spec[mask2].max()
        vm2 = vels[spec[mask2].argmax() + mask1.sum()]
        g1 = models.Gaussian1D(amplitude=sm1, mean=-50, stddev=8.0)
        g1.amplitude.bounds = (0, None)
        g2 = models.Gaussian1D(amplitude=sm2, mean=-20, stddev=8.0)
        g2.amplitude.bounds = (0, None)
        g2.stddev.bounds = (5.0, 15.0)
        if line_label == "ha":
            g1.stddev.bounds = (7.0, 15.0)
            g2.stddev.bounds = (7.0, 15.0)
        else:
            g1.stddev.bounds = (5.0, 12.0)
            g2.stddev.bounds = (5.0, 12.0)
            
        if "knot" in pos_label:
            init_model = g1 if "N" in pos_label else g2
            fac = -0.5
        elif "outer" in pos_label:
            init_model = g1 + g2
            fac = 0.02
            # Tie together the component widths
            init_model.stddev_1.tied = lambda model: model.stddev_0
        else:
            init_model = g1 + g2
            fac = 0.3
            init_model.stddev_1.tied = lambda model: model.stddev_0
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
            ds="steps-mid",

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

m = gfits[("oiii", "Rings")]
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

kernel = Gaussian2DKernel(x_stddev=1.0)
hdu = linehdus["oiii"]
w = WCS(hdu.header)
xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
y1, y2 = [int(_) for _ in ylims]
x1, x2 = [int(_) for _ in xlims]
im = hdu.data
ax.imshow(im, vmin=-0.1, vmax=1.0, aspect="auto")
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
ax.set_title("Co-added Slits c + d + e: [O III] 5007", pad=16)
ax.set_xlabel("Heliocentric velocity, km/s")
ax.set_ylabel("Offset N–S from star, arcsec")
figfile = "ou5-coadd-2dspec-oiii.pdf"
fig.savefig(figfile)
fig.savefig(figfile.replace(".pdf", ".jpg"))
...;
# -

df = df.assign(dv=df.mean_1 - df.mean_0)

df.style.format(na_rep='—')

# ### Compare the line widths to estimate the temperature

df.xs("oiii")[['stddev_0', 'stddev_1']]

df.xs("ha")[['stddev_0', 'stddev_1']]

# ## Try fitting Gaussians to all the rows
#
# Use bins of size 1 arcsec

fine_pix = 1.0
fine_positions = np.arange(-28, 29) * fine_pix
fine_positions

# +
vsys = -33
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
    # Tie together the component widths
    init_model.stddev_1.tied = lambda model: model.stddev_0
    fitted_model = fitter(init_model, vels, spec)
    gs = models.Gaussian1D(amplitude=sm, mean=vsys, stddev=10.0)
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
    ) / (x.stddev_0 * x.amplitude_0 + x.stddev_1 * x.amplitude_1),
    ratio_1_0 = lambda x: x.amplitude_1 / x.amplitude_0,
)

# + jupyter={"source_hidden": true}
fdf[["mean_0", "mean_1", "mean", "wmean"]].plot()
# -

fdf[["amplitude_0", "amplitude_1", "amplitude"]].plot().set_yscale("log")


fdf[["stddev_0", "stddev_1", "stddev"]].plot().set_ylim(0.0, 25.0)


fig, ax = plt.subplots(figsize=(8, 8))
bluepaths = ax.scatter(
    fdf["mean_0"], 
    fdf.index,
    s=100 * fdf["amplitude_0"] * fdf["stddev_0"] / 7,    
    color="b",
)
redpaths = ax.scatter(
    fdf["mean_1"], 
    fdf.index,
    s=100 * fdf["amplitude_1"] * fdf["stddev_1"] / 7,   
    color="r",
)
bluecolor = bluepaths.get_facecolor()
redcolor = redpaths.get_facecolor()
# ax.scatter(
#     fdf["mean"], 
#     fdf.index,
#     s=100 * fdf["amplitude"] * fdf["stddev"] / 14,    
# )
ax.plot(
    fdf["wmean"], 
    fdf.index,
    color="k",
    ds="steps-mid",
    lw=1,
)
# Indicate the turning points in the velocity profiles
blue_tp = [
    (-45.8, 7.5),
    (-49.5, 3.5),
    (-46.5, -0.5),
    (-47.8, -3.7),
    (-44.8, -7.0),
]
red_tp = [
    (-25.6, 8.4),
    (-21.9, 4.5),
    (-22.7, 1.1),
    (-20.4, -2.5),
    (-23.4, -6.2),
]
dv = 1.5
alpha = 0.5
for v0, y0 in blue_tp:
    ax.axhline(y0, color=bluecolor, alpha=alpha, ls="dotted")
    ax.plot(
        [v0 - dv, v0 + dv], 
        [y0, y0], 
        color=bluecolor,
        lw=3,
        alpha=alpha,
        zorder=100,
    )
for v0, y0 in red_tp:
    ax.axhline(y0, color=redcolor, alpha=alpha, ls="dotted")
    ax.plot(
        [v0 - dv, v0 + dv], 
        [y0, y0], 
        color=redcolor,
        lw=3,
        alpha=alpha,
        zorder=100,
    )
# ax.plot(
#     0.5 * (fdf["mean_0"] + fdf["mean_1"]), 
#     fdf.index,
#     color="0.5",
# )
ax.axhline(0.0, color="k", ls="dashed", lw=2, alpha=0.3)
ax.axvline(vsys, color="k", ls="dashed", lw=2, alpha=0.3)
# ax.axvline(-36, color="k", ls="dotted", lw=2, alpha=0.3)
ax.set(
    xlabel="Heliocentric velocity, km/s",
    ylabel="Offset from star, arcsec",
    xlim=[-83.0, -13.0],
)
ax.set_title("[O III] 5007 Gaussian components", pad=12)
figfile = "ou5-coadd-pv-oiii-gaussians.pdf"
fig.savefig(figfile)
fig.savefig(figfile.replace(".pdf", ".jpg"))
...;

redblue_shifts = [_r[1] - _b[1] for _r, _b in zip(red_tp, blue_tp)]
redblue_shifts

print(
f"""
Average redblue offset in turning points
is {np.mean(redblue_shifts):.1f} +/- {np.std(redblue_shifts):.1f} arcsec
""")

# From the profile along slit G we find that the cylindrical diameter of the nebula is $D = 8.7 +/- 0.1 arcsec. 
#
# The offset should be $dz = D \cos i$, which gives $i = 83 \pm 2$

# This compares very well with the Jones value of $82 \pm 1$

# ## Revisit the kinetic temperature measurement
#
# The fact that Corradi et al 2015 are suggesting low temperature for the H emission means that it is vital to have an independent estimate of the temperature. 
#
# I think that with the line width kinetic temperature we an get a good estimate

# ### Fine-scale fits for Ha
#
# We just copy what we did for oiii above.  We reuse all these variables since the oiii results got saved in a dataframe `fdf`
#
# We had to change the allowed bounds on the widths

# +
vsys = -33
v1, v2 = vsys - 100, vsys + 100
fitter = fitting.LevMarLSQFitter()
hfinefits = {}
hfinesingles = {}
hfineprofiles = {}
hdu = linehdus["ha"]
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
    g1.stddev.bounds = (7.0, 15.0)
    g2 = models.Gaussian1D(amplitude=sm, mean=-20, stddev=10.0)
    g2.stddev.bounds = (7.0, 15.0)
    init_model = g1 + g2
    # Tie together the component widths
    init_model.stddev_1.tied = lambda model: model.stddev_0
    fitted_model = fitter(init_model, vels, spec)
    gs = models.Gaussian1D(amplitude=sm, mean=vsys, stddev=10.0)
    fitted_single = fitter(gs, vels, spec)
    hfinefits[finepos] = fitted_model
    hfinesingles[finepos] = fitted_single
    hfineprofiles[finepos] = {
        "v": vels,
        "spec": spec,
        "model": fitted_model(vels),
        "g1": fitted_model[0](vels),
        "g2": fitted_model[1](vels),
        "gs": fitted_single(vels),
    }
...;
# -
fdf_h = pd.DataFrame(
    {k: dict(zip(m.param_names, m.parameters)) for k, m in hfinefits.items()}
).T
fdf_h.style.format(na_rep='—')

fdfs_h = pd.DataFrame(
    {k: dict(zip(m.param_names, m.parameters)) for k, m in hfinesingles.items()}
).T
fdfs_h.style.format(na_rep='—')

fdf_h = fdf_h.join(fdfs_h)
fdf_h = fdf_h.assign(
    wmean = lambda x: (
        x.mean_0 * x.stddev_0 * x.amplitude_0
        + x.mean_1 * x.stddev_1 * x.amplitude_1
    ) / (x.stddev_0 * x.amplitude_0 + x.stddev_1 * x.amplitude_1),
    ratio_1_0 = lambda x: x.amplitude_1 / x.amplitude_0,
)

# #### Compare velocities

ax = fdf_h[["mean_0", "mean_1"]].join(
    fdf[["mean_0", "mean_1"]],
    lsuffix="_h", rsuffix="_o",
).plot(colormap="Paired")
ax.set_xlim(-20, 20)
ax.legend(ncol=2, fontsize="x-small")

# #### Compare widths

# +
ax = fdf_h[["stddev_0", "stddev"]].join(
    fdf[["stddev_0", "stddev"]],
    lsuffix="_h", rsuffix="_o",
).plot(colormap="Paired")
ax.set_ylim(0.0, 25.0)
ax.set_xlim(-20, 20)

ax.legend(ncol=2, fontsize="x-small")
# -


# #### Compare intensities

ax = fdf_h[["amplitude_0", "amplitude_1"]].join(
    fdf[["amplitude_0", "amplitude_1"]],
    lsuffix="_h", rsuffix="_o",
).plot(colormap="Paired")
ax.legend(ncol=2, fontsize="x-small")
ax.set_yscale("log")


ax = fdf_h[["ratio_1_0"]].join(
    fdf[["ratio_1_0"]],
    lsuffix="_h", rsuffix="_o",
).plot(colormap="Paired")
ax.legend(ncol=2, fontsize="x-small")
ax.set_ylim(0.0, 4.0)
ax.set_xlim(-20, 20)
ax.axhline(1.0, color="k", ls="dotted")

# #### Compare everything

# +
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 12))

ax = axes[0]
fdf_h[["mean_0", "mean_1"]].join(
    fdf[["mean_0", "mean_1"]],
    lsuffix="_h", rsuffix="_o",
).plot(ax = ax, colormap="Paired")
ax.set_xlim(-20, 20)
ax.legend(ncol=2, fontsize="x-small")

ax = axes[1]
fdf_h[["stddev_0", "stddev"]].join(
    fdf[["stddev_0", "stddev"]],
    lsuffix="_h", rsuffix="_o",
).plot(ax=ax, colormap="Paired")
ax.set_ylim(0.0, 25.0)
ax.set_xlim(-20, 20)
ax.legend(ncol=2, fontsize="x-small")

ax = axes[2]
ax = fdf_h[["ratio_1_0"]].join(
    fdf[["ratio_1_0"]],
    lsuffix="_h", rsuffix="_o",
).plot(ax=ax, colormap="Paired")
ax.legend(ncol=2, fontsize="x-small")
ax.set_ylim(0.0, 4.0)
ax.set_xlim(-20, 20)
ax.axhline(1.0, color="k", ls="dotted")
# -

# ### Compare variances from fits
#
# If we assume that the non-thermal broadening is the same for Ha and [O III], then the variances should be related as 
# $$
# \sigma^2(\mathrm{H\alpha}) = \sigma^2(\mathrm{[O III]}) + 77.34 T_4 + 10.233
# $$
# where the constant term is from the fine-structure broadening (see Garcia-Díaz & Henney 2008). Although there may also be a difference in the instrumental width.

w_h = 3e5 * 0.26 / 6563
w_o = 3e5 * 0.19 / 5007
w_h, w_o

# So the instrumental width in km/s is almost identical for the two lines. Hence we will not try to correct for it. 

w_o ** 2 / (8 * np.log(2))

vardf = fdf_h[["stddev_0", "stddev_1", "stddev"]].join(
    fdf[["stddev_0", "stddev_1", "stddev", "amplitude_1"]],
    lsuffix="_h", rsuffix="_o",
)

fig, ax = plt.subplots(figsize=(8, 12))
scale = np.where(
    # vardf["amplitude_1"] > 0.3,
    vardf["amplitude_1"] > 0.05,
    vardf["amplitude_1"],
    np.nan,
)
scat = ax.scatter(
    vardf["stddev_0_o"]**2,
    vardf["stddev_0_h"]**2,
    s=200 * scale,
    c=vardf.index,
    cmap="Spectral",    
    vmin=-15, vmax=15,
    # cmap="plasma",    
    # vmin=-10, vmax=10,
    zorder=100,
    alpha=1.0,
    edgecolors="k",
    linewidths=0.5,
)
xx = np.array([0, 200])
ax.plot(xx, xx + 10.233, c="0.8")
x0 = 63
ax.text(x0, x0 - 2 + 10.233, r"$T = 0$ K", rotation=45, c="0.8")
ax.plot(xx, xx + 10.233 + 77.34 / 2, c="0.6")
x0 = 28
ax.text(x0, x0 - 2 + 10.233 + 77.34 / 2, r"$T = 5000$ K", rotation=45, c="0.6")
ax.plot(xx, xx + 10.233 + 77.34, c="0.4")
x0 = 28
ax.text(x0, x0 - 2 + 10.233 + 77.34, r"$T = 10\,000$ K", rotation=45, c="0.4")
fig.colorbar(
    scat, ax=ax, 
    orientation="horizontal",
    location="bottom",
    shrink=0.95,
    # fraction=0.05,
).set_label("Displacement along axis", fontsize="small")
# ax.set_xlim(0, 120)
# ax.set_ylim(0, 120)
ax.set_xlim(25, 85)
ax.set_ylim(70, 130)
# ax.set_xticks(ax.get_yticks())
ax.set_aspect("equal")
ax.set_xlabel(r"$\sigma^2$ ( [O III] ), km$^2$ / s$^2$")
ax.set_ylabel(r"$\sigma^2$ ( H$\alpha$ ), km$^2$ / s$^2$")
figfile = "pn-ou5-gaussfit-temperature.pdf"
fig.savefig(figfile, bbox_inches="tight")
fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")


dsigsq = vardf["stddev_0_h"]**2 - vardf["stddev_0_o"]**2
weights = scale
m = np.isfinite(scale)
mean_dsigsq = np.average(dsigsq[m], weights=weights[m])
std_dsigsq = np.sqrt(
    np.average((dsigsq[m] - mean_dsigsq)**2, weights=weights[m])
)
mean_dsigsq, std_dsigsq

mean_T4 = (mean_dsigsq - 10.233) / 77.34
std_T4 = std_dsigsq / 77.34
mean_T4, std_T4

# + [markdown] editable=true slideshow={"slide_type": ""}
# So $T = 5700 \pm 1300$ K
# -

# ### Non-parametric version
#
# Directly compare the moments in a window about the line. *Cancel this, since it is not going to give anything useful, owing to the fact that line splitting is larger in [O III] than in H alpha*

# + [markdown] editable=true slideshow={"slide_type": ""}
# ### Two-phase interpretation
#
# All of the papers on high-adf pne try to model it as a cool phase with T < 1000 K that contributes a small fraction to the H line emission. This is how the Balmer jump temperature is interpreted in García-Rojas et al (2022) for example. They find that a cold region that contributes 3% of the emission ($\omega = 0.03$) can result in a Balmer jump temperature that is half the warm region temperature (they use 8000 and 800 K). 
#
# If we look at how the variance should behave for two components (e.g., García-Díaz & Henney 2008) we have 
# $$
# \sigma^2 = (1 - \omega) \sigma_w^2 + \omega \sigma_c^2 + \omega (1 - \omega) (V_w - V_c)^2
# $$
# where $\omega, 1 - \omega$ are the fractions of the emission that comes from the cool, warm components. 
#
# So if the cool component has $T_c = \alpha T_w$, and the mean velocities are the same ($V_w - V_c$) then this is
# $$
# \sigma^2 = (1 + \alpha - \omega) \sigma_w^2 
# $$ 
# Since we find that $\sigma^2 \approx 0.6 \sigma_w^2$, this means $\omega \approx 0.4 + \alpha$. For instance, if $\alpha = 0.1$ ($T_c = 1000$ K), then $\omega = 0.5$ and half of the H$\alpha$ emission comes from the cool component. (Note that the implicit assumption is that [O III] comes from only the warm component).
#
# *In the other notebook, we will do the convolution using this model*
#

# + [markdown] editable=true slideshow={"slide_type": ""}
# ## Look at the difference in line splitting between Oiii and Ha
# -



splitdf = fdf_h[["mean_0", "mean_1"]].join(
    fdf[["mean_0", "mean_1", "amplitude_1"]],
    lsuffix="_h", rsuffix="_o",
)

# +
fig, ax = plt.subplots(figsize=(8, 12))
scale = np.where(
    splitdf["amplitude_1"] > 0.05,
    splitdf["amplitude_1"],
    np.nan,
)
scat = ax.scatter(
    splitdf["mean_1_o"] - splitdf["mean_0_o"],
    splitdf["mean_1_h"] - splitdf["mean_0_h"],
    s=200 * scale,
    c=splitdf.index,
    cmap="Spectral",    
    vmin=-15, vmax=15,
    zorder=100,
    alpha=1.0,
    edgecolors="k",
    linewidths=0.5,
)
xx = np.array([0, 50])
ax.plot(xx, xx, c="k", ls="dotted")
# ax.plot(xx, xx - 2.4, c="0.8")
ax.plot(xx, xx * 0.9, c="0.8")

# ax.text(5, 5 - 5 + 10.233, r"$T = 0$ K", rotation=45, c="0.8")
# ax.plot(xx, xx + 10.233 + 77.34 / 2, c="0.6")
# ax.text(5, 5 - 5 + 10.233 + 77.34 / 2, r"$T = 5000$ K", rotation=45, c="0.6")
# ax.plot(xx, xx + 10.233 + 77.34, c="0.4")
# ax.text(5, 5 - 5 + 10.233 + 77.34, r"$T = 10\,000$ K", rotation=45, c="0.4")
ax.set_xlim(20, 40)
ax.set_ylim(15, 35)
ax.set_xticks([20, 25, 30, 35, 40])
ax.set_yticks([15, 20, 25, 30, 35])
ax.set_aspect("equal")
ax.set_xlabel(r"$(V_\mathrm{red} - V_\mathrm{blue})$ ( [O III] ), km / s")
ax.set_ylabel(r"$(V_\mathrm{red} - V_\mathrm{blue})$ ( H$\alpha$ ), km / s")
ax.text(30, 30 + 0.7, "Slope = 1.0", rotation=45, fontsize="small")
ax.text(33, 29.7 + 0.6, "Slope = 0.9", rotation=np.rad2deg(np.arctan(0.9)), fontsize="small")
fig.colorbar(
    scat, ax=ax, 
    orientation="horizontal",
    location="bottom",
    shrink=0.95,
    # fraction=0.05,
).set_label("Displacement along axis", fontsize="small")
figfile = "pn-ou5-gauss-vsplit.pdf"
fig.savefig(figfile, bbox_inches="tight")
fig.savefig(figfile.replace(".pdf", ".jpg"), bbox_inches="tight")
# -


# ## Contours to show the residual wing emission

vlist = []
slist = []
resid_list = []
for pos in fine_positions:
    data = fineprofiles[pos]
    vlist.append(data["v"])
    slist.append(pos * np.ones_like(data["v"]))
    resid_list.append(data["spec"] - data["model"])

resid_arr = np.stack(resid_list)
v_arr = np.stack(vlist)
s_arr = np.stack(slist)

resid_arr.max()

# Repeat for Ha

hvlist = []
hslist = []
hresid_list = []
for pos in fine_positions:
    data = hfineprofiles[pos]
    hvlist.append(data["v"])
    hslist.append(pos * np.ones_like(data["v"]))
    hresid_list.append(data["spec"] - data["model"])

hresid_arr = np.stack(hresid_list)
hv_arr = np.stack(hvlist)
hs_arr = np.stack(hslist)

hresid_arr.max()

# Combined plot

# +
vsys = -33
v1, v2 = vsys - 100, vsys + 100
s1, s2 = -15, 15

hdu = linehdus["oiii"]
w = WCS(hdu.header)
fig, axes = plt.subplots(
    2, 1, 
    sharex=True,
    sharey=True,
    figsize=(8, 8), 
    subplot_kw=dict(projection=w),
)

xlims, ylims = w.world_to_pixel_values([v1, v2], [s1, s2])
y1, y2 = [int(_) for _ in ylims]
x1, x2 = [int(_) for _ in xlims]
im = hdu.data

ax = axes[0]
ax.imshow(im, vmin=-0.1, vmax=1.0, aspect="auto")
trw = ax.get_transform("world")
ax.contour(
    v_arr, s_arr, resid_arr,
    levels=[0.015, 0.03, 0.06], 
    colors="w",
    linewidths=[0.5, 1.0, 1.5, 2.0],
    transform=trw,
)
x0, y0 = w.world_to_pixel_values(vsys, 0.0)
ax.axhline(y0, color="orange", ls="dashed", lw=4, alpha=0.3)
ax.axvline(x0, color="orange", ls="dashed", lw=4, alpha=0.3)
ax.set(xlim=xlims, ylim=ylims)
ax.set_xlabel("")
ax.set_ylabel("Slit offset, arcsec")
ax.text(-125, 10, "[O III] residuals", transform=trw, color="w")

ax = axes[1]
hdu = linehdus["ha"]
w = WCS(hdu.header)
im = hdu.data
ax.imshow(im, vmin=-0.1, vmax=1.0, aspect="auto")
trw = ax.get_transform("world")
ax.contour(
    hv_arr, hs_arr, hresid_arr,
    levels=[0.015, 0.03, 0.06], 
    colors="w",
    linewidths=[0.5, 1.0, 1.5, 2.0],
    transform=trw,
)
x0, y0 = w.world_to_pixel_values(vsys, 0.0)
ax.axhline(y0, color="orange", ls="dashed", lw=4, alpha=0.3)
ax.axvline(x0, color="orange", ls="dashed", lw=4, alpha=0.3)

ax.set_xlabel("Heliocentric velocity, km/s")
ax.set_ylabel("Slit offset, arcsec")
ax.text(-125, 10, r"H$\alpha$ residuals", transform=trw, color="w")

figfile = "ou5-coadd-residuals-oiii-ha.pdf"
fig.savefig(figfile, bbox_inches="tight")
fig.savefig(figfile.replace(".pdf", ".jpg"))
...;
# -

# #### Displacement along slit between residual on red and blue wings

# Take a spatial profile of the residuals on the blue and red side. Use $\pm 15$ km/s around points that $\pm 35$ km/s with respect to the systemic velocity.

resid_blue = np.nansum(
    np.where(
        np.abs(v_arr - (vsys - 35)) < 15,
        resid_arr,
        np.nan,
    ),
    axis=1,
)
resid_red = np.nansum(
    np.where(
        np.abs(v_arr - (vsys + 35)) < 15,
        resid_arr,
        np.nan,
    ),
    axis=1,
)


# Fit gaussians to the red and blue spatial profiles.

maxpos = 12
is_near_center = np.abs(fine_positions) <= maxpos
g_b = fitter(
    models.Gaussian1D(amplitude=np.max(resid_blue), mean=0, stddev=3.0),
    fine_positions[is_near_center], 
    resid_blue[is_near_center],
)
g_r = fitter(
    models.Gaussian1D(amplitude=np.max(resid_red), mean=0, stddev=3.0),
    fine_positions[is_near_center], 
    resid_red[is_near_center],
)
g_b, g_r

fig, ax = plt.subplots()
ax.plot(fine_positions, resid_blue, color="b", ds="steps-mid")
ax.plot(fine_positions, g_b(fine_positions), lw=0.5, color="b")
ax.plot(fine_positions, resid_red, color="r", ds="steps-mid")
ax.plot(fine_positions, g_r(fine_positions), lw=0.5, color="r")
ax.set_xlim([-maxpos, maxpos])
ax.set_xlabel("Position")
ax.set_ylabel("[O III] Residual")


# Look at the difference in the centroids. 

g_r.mean - g_b.mean

# Repeat for H alpha

hresid_blue = np.nansum(
    np.where(
        np.abs(v_arr - (vsys - 35)) < 15,
        hresid_arr,
        np.nan,
    ),
    axis=1,
)
hresid_red = np.nansum(
    np.where(
        np.abs(v_arr - (vsys + 35)) < 15,
        hresid_arr,
        np.nan,
    ),
    axis=1,
)


hg_b = fitter(
    models.Gaussian1D(amplitude=np.max(hresid_blue), mean=0, stddev=3.0),
    fine_positions[is_near_center], 
    hresid_blue[is_near_center],
)
hg_r = fitter(
    models.Gaussian1D(amplitude=np.max(hresid_red), mean=0, stddev=3.0),
    fine_positions[is_near_center], 
    hresid_red[is_near_center],
)
hg_b, hg_r

fig, ax = plt.subplots()
ax.plot(fine_positions, hresid_blue, color="b", ds="steps-mid")
ax.plot(fine_positions, hg_b(fine_positions), lw=0.5, color="b")
ax.plot(fine_positions, hresid_red, color="r", ds="steps-mid")
ax.plot(fine_positions, hg_r(fine_positions), lw=0.5, color="r")
ax.set_xlim([-maxpos, maxpos])
ax.set_xlabel("Position")
ax.set_ylabel("H alpha Residual")


hg_r.mean - hg_b.mean

# So taking the average of H alpha and [O III] we have 3 +/- 1 arcsec

# Although, we should maybe give more weight to the brighter residuals. So we take the four displacements from zero and calculate mean weighted by the amplitudes. We do the same thing for the variance. Then we take twice that mean value as the mean displacement and sqrt of twice the variance for the error. This assumes that the error on the two sides is independent. 

gausses = g_r, g_b, hg_r, hg_b
weights = np.array([_.amplitude.value for _ in gausses])
shifts = np.array([np.abs(_.mean.value) for _ in gausses])
mean_shift = np.average(shifts, weights=weights)
var_shift = np.average((shifts - mean_shift)**2, weights=weights)
displacement = 2 * mean_shift
e_displacement = np.sqrt(2 * var_shift)
f"Displacement: {displacement:.1f} +/- {e_displacement:.1f} arcsec"

# Yes, that looks more realistic

# ## Look at the individual profile fits in more detail                                                                                                                        
# We want to see whether it makes sense to fit two gaussians in the outer lobes

fig, ax = plt.subplots(figsize=(8, 8))
for pos in fine_positions:
    data = fineprofiles[pos]
    v = data["v"]
    norm = np.max(data["spec"])
    g1 = np.where(data["g1"] > 0.02 * norm, data["g1"] , np.nan)
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

# ### Central region for [O III]

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

# Residuals

fig, ax = plt.subplots(figsize=(8, 8))
bmaxes, rmaxes, smaxes = [], [], []
for pos in fine_positions:
    if abs(pos) > 12.0:
        continue
    data = fineprofiles[pos]
    v = data["v"]
    bpix = v < -50
    rpix = v > -15
    # norm = np.max(data["spec"])
    norm = 1
    scale = 10.0
    resids = data["spec"] - data["model"]
    bmaxes.append(np.max(resids[bpix]))
    rmaxes.append(np.max(resids[rpix]))
    smaxes.append(pos)
    ax.axhline(pos, lw=0.5, color="k")
    ax.plot(v, pos + scale * resids / norm, color="k", alpha=0.4, drawstyle="steps-mid")

fig, ax = plt.subplots()
ax.plot(smaxes, bmaxes, color="b", ds="steps-mid")
ax.plot(smaxes, rmaxes, color="r", ds="steps-mid")
ax.set_xlabel("Offset, arcsec")
ax.set_ylabel("Maximum residual")


# ### Central region for Ha

fig, ax = plt.subplots(figsize=(8, 8))
for pos in fine_positions:
    if abs(pos) > 12.0:
        continue
    data = hfineprofiles[pos]
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
    data = hfineprofiles[pos]
    v = data["v"]
    # norm = np.max(data["spec"])
    norm = 1
    scale = 10.0
    resids = data["spec"] - data["model"]
    g1 = np.where(data["g1"] > 0.02 * norm, data["g1"], np.nan)
    g2 = np.where(data["g2"] > 0.02 * norm, data["g2"], np.nan)
    # ax.fill_between(v, pos + scale * g1 / norm, pos, color="b", lw=0.5, alpha=0.3)
    # ax.fill_between(v, pos + scale * g2 / norm, pos, color="r", lw=0.5, alpha=0.3)
    # if abs(pos) > 20.0:
    #     gs = np.where(data["gs"] > 0.01 * norm, data["gs"], np.nan)
    #     ax.plot(v, pos + scale * gs / norm, color="g", lw=1.3)
    ax.axhline(pos, lw=0.5, color="k")
    ax.plot(v, pos + scale * resids / norm, color="k", alpha=0.4, drawstyle="steps-mid")

# ### Inner lobes for both


fig, ax = plt.subplots(figsize=(8, 12))
for pos in fine_positions:
    if abs(pos) > 8.0:
        continue
    for shift, data, bc, rc in (-0.25, hfineprofiles[pos], "g", "y"), (0.25, fineprofiles[pos], "b", "r"):
        v = data["v"]
        norm = np.sum(data["spec"])
        scale = 15
        g1 = data["g1"]
        g2 = data["g2"]
        ax.fill_between(v, pos + scale * g1 / norm, pos, color=bc, lw=0.5, alpha=0.3)
        ax.fill_between(v, pos + scale * g2 / norm, pos, color=rc, lw=0.5, alpha=0.3)
        ax.plot(v, pos + scale * data["spec"] / norm, color="k", alpha=0.4, drawstyle="steps-mid")
ax.set_xlim(-90, 10)


