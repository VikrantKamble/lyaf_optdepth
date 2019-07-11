import numpy as np
import matplotlib.pyplot as plt
import fitsio
import time
import emcee
import os
from functools import partial
from multiprocessing import Pool

from lyaf_optdepth.utils import xfm
from getdist import MCSamples


class Optimizer:
    def __init__(self, shift=None, tilt=None, n_mcmc=5000,
                 burn_frac=40, n_walkers=50, par_limits=None, n_bins=100):

        """ Basic fitting class for optical depth parameter estimation

        Args:
            shift, tilt (Optional): Parameters that govern the linear
                transformation between t0-gamma and x0-x1 basis
            n_mcmc (int, Optional): The number of MCMC steps
            burn_frac (float, Optional): Fraction of the MCMC steps to
                discard as burn-in
            n_walkers (int, Optional): The number of walkers for the
                affine-invariant ensemble sampler
            par_limits (Optional): A sequence of 2-tuples for the uniform
                prior
            n_bins (int, Optional): Number of grid cells to divide the par_limits
                for evaluating the likelihood

        """
        if par_limits is None:
            par_limits = [(0, 3), (-10, 10), (-0.5, 0.5)]

        if shift is None:
            shift = np.array([-5.27, 3.21])

        if tilt is None:
            tilt = np.array([[-0.8563,  0.5165],
                             [0.5165,  0.8563]])

        self.shift = shift
        self.tilt = tilt
        self.ndim = 3
        self.n_mcmc = n_mcmc
        self.burn_frac = burn_frac
        self.n_walkers = n_walkers
        self.par_limits = par_limits
        self.n_bins = n_bins

        self.flux = None
        self.ivar = None
        self.zlyaf = None

    @property
    def get_x0_x1(self):
        x0_vec = np.linspace(*self.par_limits[1], self.n_bins)
        x1_vec = np.linspace(*self.par_limits[2], self.n_bins)
        return x0_vec, x1_vec

    def set_data(self, flux, ivar, zlyaf):
        self.flux = flux
        self.ivar = ivar
        self.zlyaf = zlyaf

    def set_data_from_zqso(self, flux, ivar, zqso, rest_wave):
        zlyaf = rest_wave * (1 + zqso[:, None]) / 1215.67 - 1
        self.set_data(flux, ivar, zlyaf)

    def set_data_from_fits(self, composite, forest=None):
        if forest is None:
            forest = [1070, 1160]

        fits = fitsio.FITS(composite)
        flux = fits[0].read()
        ivar = fits[1].read()
        zqso = fits[2]['REDSHIFT'][:]
        rest_wave = fits[3]['WAVELENGTH'][:]

        lam_indices = np.where((rest_wave > forest[0]) &
                               (rest_wave < forest[1]))[0]

        return self.set_data_from_zqso(flux[:, lam_indices],
                                       ivar[:, lam_indices],
                                       zqso, rest_wave[lam_indices])

    @staticmethod
    def _get_model_mod(theta, x, shift, tilt):
        f0, x0, x1 = theta
        lnt0, gamma = xfm([x0, x1], shift, tilt)[0]
        model = f0 * np.exp(-np.exp(lnt0) * (1 + x) ** gamma)
        return model

    @staticmethod
    def _lnprior_mod(theta):
        f0, x0, x1 = theta
        if 0 <= f0 < 3 and -10 <= x0 < 10 and -0.5 <= x1 < 0.5:
            return 0.0
        return -np.inf

    @staticmethod
    def _lnlike(theta, xx, yy, ee, shift, tilt):
        model = Optimizer._get_model_mod(theta, xx, shift, tilt)
        residual = yy - model
        if ee.ndim == 1:
            return -0.5 * np.sum(residual ** 2 / ee ** 2)
        return -0.5 * residual.dot(np.linalg.inv(ee)).dot(residual)

    @staticmethod
    def _lnprob(theta, xx, yy, ee, shift, tilt):
        lp = Optimizer._lnprior_mod(theta)

        if not np.isfinite(lp):
            return -np.inf
        return lp + Optimizer._lnlike(theta, xx, yy, ee, shift, tilt)

    def _lngrid_from_trace(self, trace, make_plot):
        extents = {'x0': self.par_limits[1], 'x1': self.par_limits[2]}

        samps = MCSamples(samples=trace, names=['x0', 'x1'], ranges=extents)
        density = samps.get2DDensity('x0', 'x1')

        # set up the grid on which to evaluate the likelihood
        x_bins, y_bins = self.get_x0_x1

        xx, yy = np.meshgrid(x_bins, y_bins)
        pos = np.vstack([xx.ravel(), yy.ravel()]).T

        # Evalaute density on a grid
        prob = np.array([density.Prob(*ele) for ele in pos])
        prob[prob < 0] = 1e-50
        ln_prob = np.log(prob)
        ln_prob -= ln_prob.max()
        ln_prob = ln_prob.reshape(xx.shape)

        if make_plot:
            plt.pcolormesh(x_bins, y_bins, ln_prob)
            plt.colorbar()
            plt.clim(0, -5)
        return ln_prob

    @staticmethod
    def _get_chisquare(xdata, ydata, yerr, model_func, model_params,
                       f_args, make_plot=False):
        """ Returns the chisquare goodness-of-fit metric
        """
        chisq = -2 * Optimizer._lnlike(model_params, xdata, ydata, yerr, *f_args)
        print(chisq)

        if make_plot:
            plt.figure()
            if yerr.ndim == 1:
                plt.errorbar(xdata, ydata, yerr, fmt='o', c='k', capsize=3)
            else:
                plt.errorbar(xdata, ydata, np.sqrt(np.diag(yerr)), fmt='o',
                             c='k', capsize=3)
            plt.plot(xdata, model_func(model_params, xdata, *f_args), '-k',
                     label=r"$\chi^2 = %.2f, dof = %d$" % (chisq, len(xdata)))
            plt.legend()
            plt.show()

    @staticmethod
    def _get_percentiles(trace):
        """ Return the percentile estimates from samples
        """
        res = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                   zip(*np.percentile(trace, [16, 50, 84], axis=0))))

        return np.array(res)

    def fit_model(self, xx, yy, ee, init_guess=None, n_bins=100,
                  make_plot=False, id=-1):
        """ Generic function to fit data with the model defined here

        Args:
            xx, yy [ndarray] : the x, y axis data
            ee [ndarray] : if 1d --> sigma, if 2d --> covariance matrix
            init_guess [sequence] : starting guess
            n_bins [int] : number of bins to divide the ranges into for
                           computing likelihoods on a grid
            make_plot [bool] : draw likelihood contours
            id [int, str] : identifier for the files produced

        """

        if init_guess is None:
            init_guess = [1.2, 0, 0]

        pos = [init_guess + 1e-4*np.random.randn(3) for i in range(self.n_walkers)]

        sampler = emcee.EnsembleSampler(self.n_walkers, self.ndim,
                                        Optimizer._lnprob,
                                        args=(xx, yy, ee, self.shift, self.tilt))
        sampler.run_mcmc(pos, self.n_mcmc)

        start = int(self.burn_frac * self.n_mcmc / 100.)
        samples = sampler.chain[:, start:, :].reshape((-1, self.ndim))

        # we don't care about the f0 parameter
        x_bins, y_bins = self.get_x0_x1
        ln_prob = self._lngrid_from_trace(samples[:, 1:], make_plot)

        np.savetxt('lnlike_%s.dat' % str(id), ln_prob)

        # the actual pdf of the parameters f0, x0 and x1 are definitely
        # non-gaussian

        # --> use the percentile estimates to capture the disttribution
        mod_est = Optimizer._get_percentiles(samples)

        self._get_chisquare(xx, yy, ee, self._get_model_mod, mod_est[:, 0],
                            f_args=(self.shift, self.tilt), make_plot=make_plot)
        np.savetxt('xx_percentile_est_%s.dat' % str(id), mod_est)

        # --> CHeck if we are using this in the analysis - else REMOVE
        tg_points = xfm(samples[:, 1:], self.shift, self.tilt)

        orig_est = Optimizer._get_percentiles(tg_points)
        np.savetxt('tg_percentile_est_%s.dat' % str(id), orig_est)

    def _fit_skewer_index(self, index, **kwargs):
        # cleaning the data
        # YOU DON'T WANT TO SEND JUNK TO THE FITTING ROUTINE - RIGHT!!
        mask = self.ivar[:, index] > 0
        xx = self.zlyaf[:, index][mask]
        yy = self.flux[:, index][mask]
        ee = 1. / np.sqrt(self.ivar[:, index][mask])

        self.fit_model(xx, yy, ee, id=index, **kwargs)

    def fit_batch(self, save_path, indices, parallel=False, **kwargs):
        '''
        Utility function to fit the model for a given set of skewers

        --> Don't make plots when using parallel=True
        '''
        curr_dir = os.getcwd()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        os.chdir(save_path)

        start = time.time()
        if not parallel:
            for index in indices:
                self._fit_skewer_index(index, **kwargs)
        else:
            opt_func = partial(self._fit_skewer_index, **kwargs)

            pool = Pool()
            pool.map(opt_func, indices)
            pool.close()
            pool.join()

        print("Time elapsed: {} seconds".format(time.time() - start))
        os.chdir(curr_dir)
