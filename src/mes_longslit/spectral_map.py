import glob
import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
import astropy.units as u
sys.path.append('/Users/will/Dropbox/OrionWest')
from helio_utils import helio_topo_from_header, vels2waves


try:
    line_id = sys.argv[1]
except IndexError:
    print('Usage: {} LINE_ID [VRANGE [OUTPUT_FOLDER]]'.format(sys.argv[0]))

try:
    vrange = sys.argv[2]
except IndexError:
    vrange = None

try:
    outdir = sys.argv[3]
except IndexError:
    outdir = "maps"


def waves2pixels(waves, w):
    n = len(waves)
    pixels, _, _ = w.all_world2pix(waves, [RA0]*n, [Dec0]*n, 0)
    return pixels.astype(int)

# First set up WCS for the output image
#
NX, NY = 512, 512
pixel_scale = 0.3               # arcsec
dRA, dDec = -pixel_scale/3600., pixel_scale/3600.
# Center on central star of NGC 6210
RA0, Dec0 = 251.122998321, 23.7998586853
w = WCS(naxis=2)
w.wcs.cdelt = [dRA, dDec]
w.wcs.crpix = [0.5*(1 + NX), 0.5*(1 + NY)]
w.wcs.crval = [RA0, Dec0]
w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
w.wcs.cunit = ['deg', 'deg']

# Arrays to hold the output image
outimage = np.zeros((NY, NX))
outweights = np.zeros((NY, NX))

# Use a slightly wider slit than is strictly accurate 
slit_width = 2.0                
slit_pix_width = slit_width/pixel_scale

speclist = glob.glob(f'data/pvextract/*-{line_id}.fits')

# Window widths for line and BG
dwline = 7.0*u.Angstrom

for fn in speclist:
    print('Processing', fn)
    spechdu, = fits.open(fn)
    wspec = WCS(spechdu.header, key='A')

    # Trim to good portion of the slit
    goodslice = slice(None, None)

    # Find per-slit weight
    slit_weight = spechdu.header['WEIGHT']

    # Find sign of delta wavelength
    dwav = wspec.wcs.get_cdelt()[0]*wspec.wcs.get_pc()[0, 0]
    sgn = int(dwav/abs(dwav))         # Need to take slices backwards if this is negative

    # Eliminate degenerate 3rd dimension from data array and trim off bad bits
    spec2d = spechdu.data[0]

    # Rest wavelength from FITS header is in meters
    wavrest = wspec.wcs.restwav*u.m

    # Find wavelength limits for line extraction window
    if vrange is None:
        # Original case: use a window of wavelength full width dwline
        waves =  wavrest + np.array([-0.5, 0.5])*dwline
    else:
        # Extract velocity limits from the vrange command line argument
        # vrange should be of a form like '-100+100' or '+020+030'
        v1, v2 = float(vrange[:4]), float(vrange[-4:])
        print('Velocity window:', v1, 'to', v2)
        waves = vels2waves([v1, v2], wavrest,  spechdu.header, usewcs="A")
    print('Wavelength window: {:.2f} to {:.2f}'.format(*waves.to(u.Angstrom)))

    # Find pixel indices for line extraction window
    i1, i2 = waves2pixels(waves, wspec)
    print('Pixel window:', i1, 'to', i2, 'in direction', sgn)

    # Extract profile for this wavelength or velocity window
    profile = spec2d[:, i1:i2:sgn].sum(axis=-1)

    # Find celestial coordinates for each pixel along the slit
    NS = len(profile)
    slit_coords = pixel_to_skycoord(range(NS), [0]*NS, wspec, 0)

    # Trim off bad parts of slit
    profile = profile[goodslice]
    slit_coords = slit_coords[goodslice]

    # Deal with NaNs in profile:
    # - Make an array of per-pixel weights
    wp = np.ones_like(profile)*slit_weight
    # - Set profile and weight to zeros wherever there are NaNs
    badmask = ~np.isfinite(profile)
    profile[badmask] = 0.0
    wp[badmask] = 0.0

    # Convert to pixel coordinates in output image
    xp, yp = skycoord_to_pixel(slit_coords, w, 0)

    for x, y, bright, wt in zip(xp, yp, profile, wp):
        # Find output pixels corresponding to corners of slit pixel
        # (approximate as square)
        i1 = int(0.5 + x - slit_pix_width/2)
        i2 = int(0.5 + x + slit_pix_width/2)
        j1 = int(0.5 + y - slit_pix_width/2)
        j2 = int(0.5 + y + slit_pix_width/2)
        # Make sure we don't go outside the output grid
        i1, i2 = max(0, i1), max(0, i2)
        i1, i2 = min(NX, i1), min(NX, i2)
        j1, j2 = max(0, j1), max(0, j2)
        j1, j2 = min(NY, j1), min(NY, j2)
        # Fill in the square
        outimage[j1:j2, i1:i2] += bright*wt
        outweights[j1:j2, i1:i2] += wt

# Save everything as different images in a single fits file:
# 1. The sum of the raw slits 
# 2. The weights
# 3. The slits normalized by the weights
if vrange is None:
    label = line_id + '-allvels'
else:
    label = line_id + vrange

fits.HDUList([
    fits.PrimaryHDU(),
    fits.ImageHDU(header=w.to_header(), data=outimage, name='slits'),
    fits.ImageHDU(header=w.to_header(), data=outweights, name='weight'),
    fits.ImageHDU(header=w.to_header(), data=outimage/outweights, name='scaled'),
    ]).writeto(f'data/{outdir}/turtle-slits-{label}.fits', overwrite=True)
