""""
Two-phase model for the Hα emission line profile.

This module models the emission line profile of hydrogen Hα
produced by a two-phase ionized gas with fine-structure splitting
and thermal broadening. The profile is computed as a weighted sum
of two temperature components, each represented by a Gaussian-broadened
fine-structure model derived from the emissivity tables of Clegg (1999).

Classes
-------
PhaseProfile
    Computes the fine-structure + thermal profile at a single
    temperature.

TwoPhaseProfile
    Combines warm and cool PhaseProfiles into a composite line
    profile.

Functions
---------
discrete_gaussian(x, amplitude, mean, stddev, bin_width)
    Evaluate a Gaussian profile integrated over discrete bins.

Author: William Henney, 2025
"""
import numpy as np
import scipy.stats
from astropy.table import Table
from astropy import constants
import astropy.units as u
from astropy.io import ascii

_cdf = scipy.stats.norm.cdf

def discrete_gaussian(x, amplitude, mean, stddev, bin_width):
    "Gaussian profile integrated over finite bins"
    return amplitude * (
        _cdf(x + 0.5 * bin_width, loc=mean, scale=stddev)
        - _cdf(x - 0.5 * bin_width, loc=mean, scale=stddev)
    )

# Reference Ha wavelength used by Clegg 1999
WAV_REF = 6562.8812

# RMS thermal broadening at T = 1e4 K for A = 1
SIG0 = np.sqrt( 
    constants.k_B * 10_000 * u.K / constants.m_p
).to(u.km / u.s).value

# Data tables for the fine structure components from Clegg 1999
# So far, this is only Case B and only n=1e2, 1e4 pcc
H_CASE_B_DATA = {
    "n2": """
Index,d lam,d v,300,1000,3000,10000,30000
1,-.130,-5.93,.018,.023,.031,.044,.061
2,.028,1.27,.036,.046,.061,.088,.121
3,-.110,-5.03,.057,.069,.085,.106,.125
4,-.157,-7.16,.113,.138,.170,.213,.250
5,-.172,-7.85,.259,.241,.218,.183,.148
6,-.014,-0.65,.052,.048,.044,.037,.030
7,-.030,-1.36,.465,.435,.392,.330,.266
    """,
    "n4": """
Index,d lam,d v,300,1000,3000,10000,30000
1,-.130,-5.93,.021,.025,.032,.044,.061
2,.028,1.27,.042,.049,.063,.088,.121
3,-.110,-5.03,.064,.073,.087,.107,.125
4,-.157,-7.16,.129,.147,.175,.215,.250
5,-.172,-7.85,.248,.235,.214,.182,.147
6,-.014,-0.65,.050,.047,.043,.036,.029
7,-.030,-1.36,.447,.423,.386,.328,.265
    """,
}

class PhaseProfile:
    """
    Fine structure plus thermal line profile for a single temperature.

    Computes the thermal plus fine-structure line profile of Hα at a
    given temperature. The fine structure components are taken from
    Clegg (1999). Thermal broadening is computed for hydrogen and
    adjusted relative to a heavier ion of atomic weight `A_other`.

    Currently supports Hα only.

    Parameters
    ----------
    temperature : float
        Temperature of the gas (in Kelvin). Used to compute thermal
        broadening.
    density_tag : str, optional
        Key into the `H_CASE_B_DATA` dictionary, specifying which case
        B emissivity table to use. Default is 'n2'.
    A_other : float, optional
        Atomic weight of the other ion for comparing the line
        broadening. Default is 16 (i.e., oxygen).
    dv : float, optional
        Velocity resolution (in km/s) of the output profile grid.
        Default is 0.1 km/s.
    vmax : float, optional
        Half-width of the velocity grid (in km/s). The profile is
        computed on `[-vmax, vmax]` with steps of `dv`. Default is
        50 km/s.
    """

    def __init__(
        self, 
        temperature,
        density_tag="n2",
        A_other=16,
        dv=0.1,
        vmax=50,
    ):
        self.temperature = temperature
        self.A_other = A_other
        self._initialize_components(
            ascii.read(
                H_CASE_B_DATA[density_tag],
                format='csv',
           )
        )
        # Mean velocity over components
        self.vmean = np.average(self.vcomps, weights=self.icomps)
        # Centroid lab wavelength (in air)
        self.wav0 = WAV_REF * (1 + self.vmean / 3e5)
        # RMS excess thermal sigma of each component
        self.sigma = SIG0 * np.sqrt(
            (self.temperature / 1e4) * (1 - 1 / self.A_other)
        )
        # Velocity grid for evaluating profile
        self.dv = dv
        self.nvgrid = 1 + int(2 * vmax / dv)
        self.vgrid = np.linspace(-vmax, vmax, self.nvgrid)
        self._initialize_igrid()

    def _initialize_components(self, table):
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
        
    def _initialize_igrid(self):
        self.igrid = np.zeros_like(self.vgrid)
        for _icomp, _vcomp in zip(self.icomps, self.vcomps):
            self.igrid += discrete_gaussian(
                self.vgrid,
                _icomp,
                _vcomp,
                self.sigma,
                self.dv,
            ) / self.dv

            
class TwoPhaseProfile:
    """
    Two-phase emission line profile with warm and cool gas components.

    Computes a composite emission line profile assuming two gas
    phases at different temperatures. Each phase is modeled using a
    thermal plus fine-structure profile. The combined profile is a
    weighted sum of the warm and cool components.

    Parameters
    ----------
    alpha : float
        Temperature ratio between the cool and warm phases:
        `T_cool = alpha * T_warm`. Must be between 0 and 1.
    omega : float
        Relative emission of the cool phase compared to the
        warm phase. Should be between 0 (no cool gas) and 1 (only
        cool gas).
    Twarm : float, optional
        Temperature of the warm phase in Kelvin. Default is 1e4 K.
    dv : float, optional
        Velocity resolution of the internal profile grid in km/s.
        Default is 0.1 km/s.
    
    Callable Behavior
    -----------------
    An instance of this class can be called as a function:

        profile = TwoPhaseProfile(alpha, omega)
        intensity = profile(v)

    where `v` is a scalar or array of velocity shifts (in km/s),
    relative to line center. Returns the interpolated profile
    intensity at each velocity.
    """
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
        """
        Interpolated line profile evaluated at velocity shift `v`.

        Parameters
        ----------
        v : array_like
            Velocity (in km/s) at which to evaluate the profile,
            relative to the mean velocity of the line.

        Returns
        -------
        intensity : ndarray
            Interpolated profile intensity at each velocity value.
        """
        return np.interp(v + self.vmean, self.vgrid, self.igrid)


