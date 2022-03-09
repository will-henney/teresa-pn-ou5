"""
Reduction of echelle longslit data from the Manchester Echelle
Spectrograph (MES, or Mezcal), installed on the 2.1m telescope at
Observatorio Astronómico Nacional, San Pedro Mártir (OAN-SPM
Observatory, Mexico)

William Henney, 2019, 2020, 2021
"""

from .regrid import regrid_images
from .slit_utils import (
    slit_profile_circle,
    slit_profile,
    find_slit_coords,
    subtract_sky,
    extract_full_profile_from_pv,
    extract_slit_profile_from_imslit,
    extract_line_and_regularize,
    make_slit_wcs,
    fit_cheb,
    make_three_plots,
    pv_extract,
    remove_cosmic_rays,
    trim_edges,
)
from .spectral_map import (
    make_vmap,
    convert_pv_offset_vels,
)

__version__ = "0.1"
