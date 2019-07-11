import os
import numpy as np
from scipy.interpolate import interp1d

# local imports
from lyaf_optdepth.bin_analyze import analyze
from lyaf_optdepth import config, create_comp, corrections

import warnings
warnings.filterwarnings('ignore')


class binObj:
    """
    Class to hold information about a particular bin over which
    optical depth is calculated
    """

    input_calib = os.environ['OPT_DATA'] + config.input_calib
    input_var_correct = os.environ['OPT_DATA'] + config.input_var_correct

    def __init__(self, name, qso, par_names, par_ranges, snt, preprocess=True):

        print('Building binclass object', name)

        if not all(x in qso.tb.dtype.names for x in par_names):
            raise ValueError('***** Feature Combination Not Available ***** \
                Options available are: {}'.format(qso.tb.dtype.names))

        self.name = name
        self.continuum = None
        self.wl = qso.wl
        self.par_names = par_names
        self.par_ranges = par_ranges

        if len(snt) == 1:
            self.snt_min, self.snt_max = snt, 100
        else:
            self.snt_min, self.snt_max = snt[0], snt[1]

        # selection criterion
        self._ixs = ((qso.sn > self.snt_min) & (qso.sn <= self.snt_max) &
                     (qso.tb[self.par_names[0]] > self.par_ranges[0][0]) &
                     (qso.tb[self.par_names[0]] <= self.par_ranges[0][1]) &
                     (qso.tb[self.par_names[1]] > self.par_ranges[1][0]) &
                     (qso.tb[self.par_names[1]] <= self.par_ranges[1][1]))

        self.tb = qso.tb[self._ixs]

        self._flux = qso.flux[self._ixs]
        self._ivar = qso.ivar[self._ixs]
        self._zq = qso.zq[self._ixs]
        self._par1 = qso.tb[self.par_names[0]][self._ixs]
        self._par2 = qso.tb[self.par_names[1]][self._ixs]

        if self.par_names[0] == "ALPHA_V0":
            self._alpha = self._par1
        elif self.par_names[1] == "ALPHA_V0":
            self._alpha = self._par2
        else:
            self._alpha = qso.tb["ALPHA_V0"][self._ixs]

        # observed wavelengths
        self._wAbs = self.wl * (1 + self._zq[:, np.newaxis])
        self._zAbs = self._wAbs / config.lya - 1

        # histogram rebinning weights
        self._hWeights = None

        self.vcorrected = False
        self.calib_corrected = False

        # Used for fake data analysis
        self.f_noise = None
        self.total_var = None
        self.f_true = None

        if preprocess:
            self.mask_pixels()
            self.mask_calcium()
            self.vcorrect()
            self.calib_correct()
        else:
            print("Be sure to run 'vcorrect' for variance corrections and"
                  "'calib_correct' for flux calibration corrections")

        print("The number of objects in this bin are %d" % len(self._zq))

    def __len__(self):
        return len(self._zq)

    def bin_info(self, plot_zabs_dist=False, plot_zabs_kwargs=None):
        """ Information about the bin """
        # Add extra functionalities here!

        print("{} : {}".format(self.parNames[0], self.parRanges[0]))
        print("{} : {} \n".format(self.parNames[1], self.parRanges[1]))

        print("Total objects in this bin: {} \n".format(len(self)))

    def mask_pixels(self, obs_ranges=None):
        # Use this to mask pixels in the observer wavelength frame
        if obs_ranges is None:
            obs_ranges = np.array([3700, 7000])

        print("Masking pixels outside the range", obs_ranges)
        mask = (self._wAbs < obs_ranges[0]) | (self._wAbs > obs_ranges[1])

        self._ivar[mask] = 0

    def mask_calcium(self):
        """ Masking Ca Ha & K lines """

        mask1 = (self._wAbs > 3925) & (self._wAbs < 3942)
        self._ivar[mask1] = 0

        mask2 = (self._wAbs > 3959) & (self._wAbs < 3976)
        self._ivar[mask2] = 0

    def vcorrect(self, vcorrect_file=input_var_correct):
        """ Apply variance corrections to the ivar vector """

        coeff = np.loadtxt(vcorrect_file)

        eta = np.piecewise(self._wAbs, [self._wAbs < 5850, self._wAbs >= 5850],
                           [np.poly1d(coeff[0:2]), np.poly1d(coeff[2:])])

        self._ivar *= eta
        self.vcorrected = True

        print('Variance corrections applied')

    def calib_correct(self, calib_file=input_calib):
        """ Use this to correct for flux calibrations in BOSS/eBOSS """

        calib_file = np.loadtxt(calib_file)

        interp_func = interp1d(calib_file[:, 0], calib_file[:, 1],
                               bounds_error=False)

        # Corrections over the observer wavelengths
        interp_mat = interp_func(self._wAbs)

        self._flux /= interp_mat
        self._ivar *= interp_mat ** 2

        print('Flux calibration corrections applied')

    def distort_correct(self, cen_alpha=None):
        if cen_alpha is None:
            cen_alpha = np.median(self.alpha)

        distort_mat = (self.wl[:, None] / 1450.) ** (cen_alpha - self.alpha)

        self._flux *= distort_mat
        self._ivar /= distort_mat ** 2

    def make_composite(self, zbins=None):
        if zbins is None:
            zbins = np.arange(2.1, 4.5, 0.05)

        create_comp.create_comp(self._flux, self._ivar, self._zq, self.wl,
                                zbins, self.folder)

    def get_calibration(self):
        ixs = (self._zq > 1.6) & (self._zq < 4)

        rest_range = [[1280, 1290], [1320, 1330], [1345, 1360], [1440, 1480]]

        # normalization range used
        obs_min, obs_max = 4600, 4640

        corrections.calibrate(self.wl, self._flux[ixs], self._ivar[ixs],
                              self._zq[ixs], rest_range, obs_min, obs_max,
                              self.folder, True)

    def set_continuum(self, tau_mean, forest=None):
        if forest is None:
            forest = [1070, 1160]

        forest_indices = (self.wl > forest[0]) & (self.wl < forest[1])

        abs_mat = np.exp(-np.exp(tau_mean[0]) *
                         (1 + self._zAbs[:, forest_indices]) ** tau_mean[1])
        continuum = self._flux[:, forest_indices] / abs_mat
        self.continuum = continuum.mean(0)

    def transmission_points(self):
        pass

    def raw_data_analysis(self):
        pass

    def skewer_analyze(self, index, a_kwargs=None, s_kwargs=None):
        """ Optical depth analysis on a given skewer index """

        return analyze(self, skewer_index=index, skewer_kwargs=s_kwargs,
                       calib_kwargs=a_kwargs)

    def build_fake(self, truths=None, pl_only=True, lss=False, plotit=False):
        """ Build fake lyman alpha forest spectra.
        Only xi(r = 0) correlations used
        """

        # self.fake_spec = helper.fake_spec(truths, self._ivar, self._alpha,
        #                                   self.wl, self._zq, pl_only=pl_only,
        #                                   lss=lss)

        # if plotit:
        #     rInd = np.random.randint(len(self), size=10)

        #     plt.figure()
        #     plt.plot(self.wl, self.fake_spec[rInd].T, lw=0.3)
        #     plt.show()
        pass

    def test_fake(self, index, a_kwargs={}, s_kwargs={}):  # <--
        return analyze(self, test_fake=True, distort=False,
                       skewer_index=index, skewer_kwargs=s_kwargs, **a_kwargs)
