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

# # Analysis of Ou 5 central star using post-AGB tracks
#
# Originally based on what I did for the Turtle, using Miller Bertolami 2016. But I intent to also use MIST tracks, which have a finer grid of masses
#

import numpy as np
from astropy.table import Table
from pathlib import Path

datadir = Path("../cspn-tables")

# ## Miller Bertolami tracks

byte_by_byte_description = """
Byte-by-byte Description of file:
--------------------------------------------------------------------------------
   Bytes Format Units     Label       Explanations
--------------------------------------------------------------------------------
   1-  5  I5    ---       N           Track point number
   7- 15  F9.6  [Lsun]    logL        logarithm of the stellar luminosity
  17- 25  F9.6  [K]       logTeff     logarithm of the effective temperature
  27- 35  F9.6  [cm/s2]   logg        logarithm of the surface gravity
  40- 51  F12.4 yr        t           Age since the point at LogTeff=3.85
  53- 61  F9.6  ---       Menv        Fractional mass of the envelope
  63- 71  F9.6  Msun      Mstar       Total mass of the star
  73- 82  F10.6 [Msun/yr] log(-dM/dt)  Logarithm of the Mass Loss Rate,
                                       log(-dMstar/dt)
--------------------------------------------------------------------------------
"""


def read_tracks(datafile):
    """Read each Miller–Bertolami track into a separate astropy.table
    
    Input argument `datafile` is a CDS file containing all tracks 
    for a given metallicity, e.g., "0100_t03.dat"
    
    Returns list of tables. Each table has a metadata "comments" field 
    that contains additional info (mass and surface composition). 
    """
    with open(datafile) as f:
        # Each track is separated by two blank lines
        tracks = f.read().split("\n\n\n")[:-1]
        tables = []
        for track in tracks:
            lines = track.split("\n")
            metadata = lines[:6]
            data = lines[7:]
            datastring = "\n".join(
                [byte_by_byte_description] + data
            )
            table = Table.read(datastring, format="ascii.cds")
            table.meta["comments"] = metadata
            tables.append(table)
    return tables


tabs = read_tracks(datadir / "miller-bertolami-2006" / "0100_t03.dat")
[_.meta for _ in tabs]

10**3.589592

tabs[2].show_in_notebook()

from matplotlib import pyplot as plt
import seaborn as sns
# %matplotlib inline
sns.set_context("talk")
sns.set_color_codes()


# Plot of effective temperature versus gravity, which we compare with the Ou 5 observed values.

def extract_masses(data):
    _, Mi, Mf, _ = data.meta["comments"][2].split()
    return round(float(Mi), 2), round(float(Mf), 3)


# ### Kiel diagram
#
# Make it into a Kiel diagram like in Jones+2022

log_g, dlog_g = 6.38, 0.08
# say 10% uncertainty in T
T, dT = 115, 10

fig, ax = plt.subplots(figsize=(6, 6))
ax.axvspan(T - dT, T + dT, color="k", alpha=0.1)
ax.axvline(T, color="k", lw=0.5)
ax.axhspan(log_g - dlog_g, log_g + dlog_g, color="k", alpha=0.1)
ax.axhline(log_g, color="k", lw=0.5)
for data in tabs:
    try:
        Mi, Mf = extract_masses(data)
        label = f"({Mi}, {Mf})"
    except:
        continue
    ax.plot(
        0.001 * 10**data["logTeff"], data["logg"],
        label=label,
    )
ax.legend()
ax.set(
    ylabel=r"$\log_{10}\, g$",
    xlabel=r"$T_{\mathrm{eff}}$, kK",
    ylim=[7.8, 2.5],
    xlim=[180, 30],
)
sns.despine()
None

# From this we would conclude that the $1\,M_\odot$ model is the best fit, and it is also consistent with the gravity from the final mass of 0.5 or so

# ### Time evolution

# Next, look at the timescales.

with sns.color_palette("viridis", n_colors=len(tabs)):
    fig, [axL, axM, ax] = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    ax.axhspan(1000 * (T - dT), 1000 * (T + dT), color="b", alpha=0.2)
    ax.axhline(1000 * T, color="k", lw=0.5)
    ax.axhline(20000, color="r", ls="--", lw=0.8)
    
    axL.axhspan(310, 600, color="b", alpha=0.2)
    for axx in axL, axM, ax:
        axx.axvline(0.0, color="k", lw=0.5)
        axx.axvspan(-100, 100, color="m", alpha=0.05)
        axx.axvspan(7e3, 11e3, color="b", alpha=0.2)
    for data in tabs:
        try:
            Mi, Mf = extract_masses(data)
            label = f"({Mi}, {Mf})"
        except:
            continue
        data["Teff"] = 10**data["logTeff"]
        data["L"] = 10**data["logL"]
        data["Mdot"] = 10**data["log(-dM/dt)"]
        ax.plot(
            "t", "Teff",
            data=data, label=label,
        )
        axM.plot(
            "t", "Mstar",
            data=data, label=label,
        )
        axL.plot(
            "t", "L",
            data=data,
        )
    ax.legend()
    ax.set(
        xlabel="time, years",
        ylabel=r"$T_{\mathrm{eff}}$, K",
        xlim=[-100000, 100000],
        ylim=[3000, 3e5],
    )
    ax.set_xscale("symlog", linthresh=100)
    ax.set_yscale("log")
    axL.set(
        ylim=[None, None],
        ylabel=r"$L / L_\odot$",
    )
    axL.set_yscale("log")
    axL.set_ylim(10, 2e4)
    axM.set(
        ylim=[0.5, 0.75],
        ylabel=r"$M_{*}$, $M_\odot$",
        yscale="linear",
    )
    sns.despine()


# So, this shows the problem with the (1.0, 0.532) model. It takes far too long to heat up to < 1e5 K.
#
# On the other hand, the (1.25, 0.566) model is more or less the right T at 8000 years, but is several times too luminous (2000 Lsun)

def make_table_of_times(tabs, Teff):
    logTeff = np.log10(Teff)
    tTkey = f"t({Teff})"
    rslts = {
        "Mi": [],
        "Mf": [],
        tTkey: [],
        "t_cross": [],
        "t_tr": [],
    }
    for data in tabs:
        Mi, Mf = extract_masses(data)
        rslts["Mi"].append(Mi)
        rslts["Mf"].append(Mf)
        # First time to reach given Teff
        mask = data["logTeff"] >= logTeff
        tT = data[mask]["t"].min()
        rslts[tTkey].append(tT)
        # Time to cross to maximum Teff
        icross = data["logTeff"].argmax()
        rslts["t_cross"].append(data[icross]["t"])
        # Transition time before t = 0
        rslts["t_tr"].append(-data["t"].min())
    return Table(rslts)


times = make_table_of_times(tabs, 75000)

times

# ### HR diagram with kinematic ages

# +
fig, ax = plt.subplots(figsize=(8, 8))
#ax.axvspan(4.7, 5.0, 0.6, 0.9, color="k", alpha=0.1)

ax.axvspan(np.log10(1000 * (T - dT)), np.log10(1000 * (T + dT)), color="b", alpha=0.2)
ax.axvline(np.log10(1000 * T), color="k", lw=0.5)
ax.axhspan(np.log10(310), np.log10(600), color="b", alpha=0.2)

lw = 0.5
tkin = 8200.0
# logTion = 4.3
logTion = np.log10(15_000)
for data in tabs:
    try:
        Mi, Mf = extract_masses(data)
        label = f"({Mi}, {Mf})"
    except:
        continue
    ax.plot(
        "logTeff", "logL",
        data=data, label=label,
        zorder=-100, c="k", lw=lw,
    )
    t0 = np.interp(logTion, data["logTeff"], data["t"])
    logT = np.interp(tkin + t0, data["t"], data["logTeff"])
    logL = np.interp(tkin + t0, data["t"], data["logL"])
    ax.plot(logT, logL, "*", c="k")
    m = (data["t"] > t0 + tkin/1.5) & (data["t"] < t0 + tkin*1.5)
    ax.plot(
        "logTeff", "logL",
        data=data[m], label="_nolabel_",
        zorder=-100, c="y", lw=7, alpha=0.4,
    )
    lw += 0.2
    
ax.legend()
ax.set(
    ylabel="$\log_{10}\, L/L_\odot$",
    xlabel="$\log_{10}\, T_{\mathrm{eff}}$",
    xlim=[5.5, 3.8],
    ylim=[2.0, None],
)
sns.despine()
fig.savefig("hr-planetaries.pdf")
None
# -

# So, when we look at it like this, then none of the models seem to work

# ## Now try the MIST tracks

mist_files = sorted((datadir / "MIST").glob("*.track.eep"))
mist_files

mist_tables = [
    Table.read(p, format="ascii.commented_header", 
               guess=False, fast_reader=False, header_start=-1)
    for p in mist_files
]

# Significant rows correspond to `EEP_number - 1` (Equivalent evolutionary points)

iAGB = 808 - 1
iPost = 1409 - 1
iWD = 1710 - 1

# ### Masses of the models
#
# Initial masses

m_init = [np.round(tab["star_mass"][0], 4) for tab in mist_tables]
m_init

# Final masses

m_agb = [np.round(tab["star_mass"][iAGB], 4) for tab in mist_tables]
m_post = [np.round(tab["star_mass"][iPost], 4) for tab in mist_tables]
m_wd = [np.round(tab["star_mass"][iWD], 4) for tab in mist_tables]

mass_tab = Table({
    "Initial": m_init,
    "AGB": m_agb,
    "Post-AGB": m_post,
    "Final": m_wd,
})

mass_tab["Label"] = [f"{_mi:.2f}, {_mf:.4f}" for _mi, _, _, _mf in mass_tab]

mass_tab

mist_tab = mist_tables[-1]

mist_tab.meta

mist_tab.colnames



mist_tab.show_in_notebook()

# ### Plot the MIST time evolution, compared with MB

from astropy.constants import R_sun
import astropy.units as u

R_sun.cgs.value

(1 * u.km).cgs.value

# Range of allowed kinematic timescales

tkin_min, tkin, tkin_max = 7.5e3, 9e3, 10.5e3

with sns.color_palette("rocket_r", n_colors=len(mist_tables)):

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 10))
    axM, axV, axL, ax = axes
    ax.axhspan(1000 * (T - dT), 1000 * (T + dT), color="b", alpha=0.2)
    ax.axhline(1000 * T, color="k", lw=0.5)
    ax.axhline(15_000, color="r", ls="--", lw=0.8)
    
    axL.axhspan(310, 600, color="b", alpha=0.2)

    axM.axhline(0.5, color="k", lw=0.5)
    axM.axhspan(0.5 - 0.06, 0.5 + 0.06, color="b", alpha=0.2)
    
    for axx in axes:
        axx.axvline(0.0, color="k", lw=0.5)
        axx.axvspan(-100, 100, color="m", alpha=0.05)
        axx.axvspan(tkin_min, tkin_max, color="b", alpha=0.2)
    
    # Miller Bertolami as pale lines
    alphaMB = 0.2
    for data in tabs[:3]:
        try:
            Mi, Mf = extract_masses(data)
            label = f"({Mi}, {Mf})"
        except:
            continue
        data["Teff"] = 10**data["logTeff"]
        data["L"] = 10**data["logL"]
        data["Mdot"] = 10**data["log(-dM/dt)"]
        # Time zerop point is where T_eff = 20,000 K
        t00 = np.interp(logTion, data["logTeff"], data["t"])
        age = data["t"] - t00
        ax.plot(
            age, data["Teff"],
            # label=label,
            label=None,
            alpha=alphaMB,
            color="g",
        )
        axM.plot(
            age, data["Mstar"],
            # label=label,
            label=None,
            alpha=alphaMB,
            color="g",
         )
        axL.plot(
            age, data["L"],
            label=None,
            alpha=alphaMB,
            color="g",
         )
    
    # MIST tracks
    mist_plot_kws = dict(lw=0.7)
    for label, data in zip(mass_tab["Label"], mist_tables):
        # Where MIST says that post-AGB starts
        t0 = data["star_age"][iPost]
        # But for MB comparison we want consistent time zero point 
        # Choose where T_eff = 20,000 K
        t00 = np.interp(logTion, data["log_Teff"], data["star_age"])
        age = data["star_age"] - t00
        ax.plot(
            age,
            10**data["log_Teff"],
            label=label,
            **mist_plot_kws,
        )
        # axMd.plot(
        #     age,
        #     -data["star_mdot"],
        #     **mist_plot_kws,
        # )
        axM.plot(
            age,
            data["star_mass"],
            **mist_plot_kws,
        )
        axL.plot(
            age,
            10**data["log_L"],
            **mist_plot_kws,
        )
        g = 10**data["log_g"] * u.cm / u.s **2
        R = 10**data["log_R"] * R_sun
        v_esc = np.sqrt(2 * g * R).to_value(u.km / u.s)
        axV.plot(
            age,
            # data["v_wind_Km_per_s"],
            v_esc,
            **mist_plot_kws,
        )
            
    ax.legend(ncol=2, fontsize="xx-small")
    ax.set(
        xlabel="time, years",
        ylabel=r"$T_{\mathrm{eff}}$, K",
        xlim=[-30_000, 30_000],
        ylim=[0, 2e5],
    )
    # ax.set_xscale("symlog", linthresh=3000)
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    
    axL.set_ylabel(r"$L / L_\odot$")
    axL.set_yscale("log")
    axL.set_ylim(10, 2e4)
    axM.set(
        ylim=[0.4, 0.9],
        ylabel=r"$M$, $M_\odot$",
        yscale="linear",
    )
    axV.set(
        ylim=[10, 3000],
        ylabel=r"$V_\mathrm{esc}$, km/s",
        yscale="log",
    )
    sns.despine()

# ### MIST Kiel diagram

with sns.color_palette("rocket_r", n_colors=len(mist_tables)):

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axvspan(T - dT, T + dT, color="k", alpha=0.1)
    ax.axvline(T, color="k", lw=0.5)
    ax.axhspan(log_g - dlog_g, log_g + dlog_g, color="k", alpha=0.1)
    ax.axhline(log_g, color="k", lw=0.5)
    for data in tabs:
        try:
            Mi, Mf = extract_masses(data)
            label = f"({Mi}, {Mf})"
        except:
            continue
        ax.plot(
            0.001 * 10**data["logTeff"], data["logg"],
            label=None, alpha=alphaMB, color="k",
        )
    for data, label in zip(mist_tables, mass_tab["Label"]):
        ax.plot(
            0.001 * 10**data["log_Teff"],
            data["log_g"],
            label=label,
            **mist_plot_kws,
        )
    
    ax.legend(ncol=2, fontsize="xx-small")
    ax.set(
        ylabel=r"$\log_{10}\, g$",
        xlabel=r"$T_{\mathrm{eff}}$, kK",
        ylim=[7.8, 2.5],
        xlim=[180, 30],
    )
    sns.despine()
...;

# ### MIST HR diagram

# Use tighter bounds on the kinematic age: $8.2 \pm 0.4$ as in the Table

tkin_min, tkin, tkin_max = 7.8e3, 8.2e3, 8.6e3


def limits(a, b, ax):
    if a>b : a, b = b, a
    ymin, ymax = ax.get_ylim()
    dy = ymax-ymin
    return ((y-ymin)/dy for y in (a, b))



with sns.color_palette("icefire", n_colors=len(mist_tables)):
    fig, ax = plt.subplots(figsize=(8, 8))
    #ax.axvspan(4.7, 5.0, 0.6, 0.9, color="k", alpha=0.1)
    
    # ax.axvline(np.log10(1000 * T), color="k", lw=0.5)
    # ax.axhspan(np.log10(310), np.log10(600), color="b", alpha=0.1)
    
    lw = 0.5
    logTion = np.log10(10_000)
    
    # for data in tabs:
    #     try:
    #         Mi, Mf = extract_masses(data)
    #         label = f"({Mi}, {Mf})"
    #     except:
    #         continue
    #     ax.plot(
    #         "logTeff", "logL",
    #         data=data, label="_nolabel_",
    #         zorder=-100, c="0.95", lw=lw,
    #     )
    #     t0 = np.interp(logTion, data["logTeff"], data["t"])
    #     logT = np.interp(tkin + t0, data["t"], data["logTeff"])
    #     logL = np.interp(tkin + t0, data["t"], data["logL"])
    #     ax.plot(logT, logL, "*", c="0.9")
    #     m = (data["t"] > t0 + tkin_min) & (data["t"] < t0 + tkin_max)
    #     ax.plot(
    #         "logTeff", "logL",
    #         data=data[m], label="_nolabel_",
    #         zorder=-100, c="y", lw=7, alpha=0.2,
    #     )
    #     lw += 0.2
    
    lw = 0.5
    for data, label in zip(mist_tables, mass_tab["Label"]):
        # Plot the evolutionary track
        ax.plot(
            "log_Teff", "log_L",
            data=data, label="_nolabel_",
            zorder=-100, c="k", lw=lw,
        )
        # Calculate zero point for ages: T_eff > 15_000 K
        t00 = np.interp(logTion, data["log_Teff"], data["star_age"])
        age = data["star_age"] - t00
        logT = np.interp(tkin + t00, data["star_age"], data["log_Teff"])
        logL = np.interp(tkin + t00, data["star_age"], data["log_L"])
        # Put symbol at kinematic age
        line, = ax.plot(logT, logL, "*", markersize=15, mew=0.5, mec="k", label=label)
        m = (data["star_age"] > t00 + tkin_min) & (data["star_age"] < t00 + tkin_max)
        # Indicate uncertainty range in kinematic age
        ax.plot(
            "log_Teff", "log_L",
            data=data[m], label="_nolabel_",
            zorder=-100, c=line.get_color(), lw=3, alpha=0.7,
        )
        # lw += 0.1
        ...

    # ou5_text = "\n".join([
    #     "Ou 5",
    #     r"$T = 100$ to 115 kK",
    #     r"$L = 300$ to 660 L$_\odot$",
    # ])
    ou5_text = "Ou 5"
    ax.text(3 + np.log10(T), np.log10(430), ou5_text, fontweight="black", ha="center", va="center")
    title = (
        r"MIST Post-AGB tracks: $M_\mathrm{initial}$, $M_\mathrm{final}$"
        "\n"
        r"$t_\mathrm{evol}$ = 8200 $\pm$ 400 yr"
    )
    ax.legend(
        ncol=2, 
        loc="right",
        fontsize="x-small",
        title=title, 
        title_fontsize="x-small",
    )
    ax.set(
        ylabel=r"$\log_{10}\, L\,/\,L_\odot$",
        xlabel=r"$\log_{10}\, T_{\mathrm{eff}}$",
        xlim=[5.2, 4.7],
        ylim=[2.1, 3.7],
    )
    ax.axvspan(
        np.log10(1000 * (T - dT)), np.log10(1000 * (T + dT)),
        *limits(np.log10(310), np.log10(600), ax),
        color="xkcd:dull green", alpha=0.6, zorder=-1000,
    )

    sns.despine()
    fig.savefig("hr-mist-ou5.pdf", bbox_inches="tight")
...;







Table.read.help("ascii.commented_header")


