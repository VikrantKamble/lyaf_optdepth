import numpy as np
import fitsio
import os


def create_comp(flux, ivar, zqso,
                rest_wave, zqso_bins, outfile='temp'):

    '''
    shape of flux = (zq, rest_wave)
    '''
    nzqso_bins = len(zqso_bins) - 1

    comp_flux = np.zeros((nzqso_bins, len(rest_wave)))
    comp_ivar = np.zeros((nzqso_bins, len(rest_wave)))

    comp_red = np.zeros(nzqso_bins)

    for i in range(len(zqso_bins) - 1):
        z_indices = np.where((zqso > zqso_bins[i]) &
                             (zqso <= zqso_bins[i+1]))[0]

        loc_flux = flux[z_indices]
        loc_ivar = ivar[z_indices]

        for j in range(len(rest_wave)):
            mask = np.where(loc_ivar[:, j] > 0)[0]

            if len(mask) > 10:  # else reject points using comp_ivar
                data = loc_flux[:, j][mask]

                # unbiased estimator of the true mean but not the most
                # precise, but shoul be okay
                comp_flux[i, j] = data.mean()
                comp_ivar[i, j] = len(data) / data.std() ** 2

        comp_red[i] = zqso[z_indices].mean()

    outfile = os.environ['OPT_COMPS'] + outfile + '.fits'
    ff = fitsio.FITS(outfile, 'rw', clobber=True)

    ff.write(comp_flux)
    ff.write(comp_ivar)
    ff.write([comp_red], names=['REDSHIFT'])
    ff.write([rest_wave], names=['WAVELENGTH'])

    ff.close()
