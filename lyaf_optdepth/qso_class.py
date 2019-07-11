#!/bin/python

""" Script that defines a QSO base class """

import os
import numpy as np
import matplotlib.pyplot as plt
import fitsio

# local imports
from lyaf_optdepth import param_fit, config


class QSO:
    """ Class to hold the quasar data used in the project """

    def __init__(self, data_file=None, attr_file=None, load_data=False):
        # G = config_map('GLOBAL')

        # if G == 'FAIL':
        #     sys.exit("Errors! Couldn't locate section in .ini file")

        if data_file is None:
            self.catalog_file = config.catalog
        else:
            self.catalog_file = data_file

        # Which redshift estimate to use?
        self.drq_z = config.drq_z

        # tb is the table that has all features of any given spectra
        if attr_file is None:
            attr_file = config.attr

        # store the metadata as a table
        self.tb = fitsio.read(attr_file, ext=-1)

        self.zq, self.sn = self.tb[self.drq_z], self.tb['SN']
        self.scale = self.tb['SCALE']

        self.wl = config.wl

        # Indicator flags
        self.flux = None
        self.ivar = None
        self.vcorrected = False
        self.scaled = False

        if load_data:
            self.load_data(self.catalog_file)

    def manual_cut(self, ind):
        """
        Manually remove some 'malicious' spectra
        """
        self.ivar[ind] = 0

    def load_data(self, cfile=None, scaling=True, scale_type='median',
                  col_name='F0_V2'):
        """
        Paramters
        ---------
        scaling   : flag whether to normalize the spectra
        scale_type: type of scaling to use 'median' or 'alpha_fit'
        col_name  : if scale_type == 'alpha_fit', the column name
                    of attr_file to use
        """
        if cfile is None:
            cfile = self.catalog_file

        print('Catalog file : ' + cfile)

        self.flux = fitsio.read(cfile, ext=0)
        self.ivar = fitsio.read(cfile, ext=1)

        print('Number of objects in the catalog: {}'.format(len(self.flux)))

        if scaling:
            if scale_type == 'median':
                scale = self.scale
            elif scale_type == 'alpha_fit':
                scale = self.tb[col_name]
            else:
                raise ValueError("Scaling type not understood. Must be either"
                                 "'median' or 'alpha_fit'")

            print("Scaling spectra using %s" % scale_type)
            self.flux /= scale[:, None]
            self.ivar *= scale[:, None] ** 2

            # Remove objects where scale is negative or nan that leads to
            # inverted spectra and also where signal-to-noise is nan
            ind = (scale < 0) | (np.isnan(scale)) | (np.isnan(self.sn))

            self.ivar[ind] = 0
            self.scaled = True

        # Remove bad spectra manually
        bad_spec = [4691]
        self.manual_cut(bad_spec)

    def get_alpha(self, index):
        """calculate spectral index for a given index"""

        if index == 'all':
            res = param_fit.process_all(self.flux, self.ivar, self.zq,
                                        param='alpha')
        else:
            res = param_fit.fit_alpha(self.flux[index], self.ivar[index],
                                      self.zq[index], plotit=True)
        return res

    def get_ew(self, index, ax=None):
        """calculate C IV equivalent width for a given index"""

        if index == 'all':
            res = param_fit.process_all(self.flux, self.ivar, self.zq,
                                        param='EW')
        else:
            if ax is None:
                fig, ax = plt.subplots(1)
            res = param_fit.fit_EW(self.flux[index], self.ivar[index],
                                   self.zq[index], plotit=True, ax=ax)
        return res

    def get_scale(self):
        """Scaling constant to bring spectra on equal footing"""

        pass
