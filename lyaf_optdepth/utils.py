import numpy as np
import matplotlib.pyplot as plt

from astroML.plotting.mcmc import convert_to_stdev as cts
from scipy.optimize import curve_fit


def xfm(pos, shift, tilt, direction='down'):
    """
    Perform conversion from one system to another

    dir : direction to do the transform
        up : orig to mod
        down(default) : mod to orig
    """
    pos = np.atleast_2d(pos)
    if direction == 'up':
        return np.dot((pos - shift), tilt.T)
    elif direction == 'down':
        return np.dot(np.linalg.inv(tilt), pos.T).T + shift


def marg_estimates(xx, yy, logL, levels=None, par_labels=["x_0", "x_1"],
                   ax=None, plot_marg=True, label='temp', **kwargs):
    """
    Marginalized statistics that follows from a jont likelihood.
    Simple mean and standard deviation estimates.

    Parameters:
        x0 : vector in x-direction of the grid
        x1 : vector in y-direction of the grid
        joint_pdf : posterior log probability on the 2D grid

    Returns:
        [loc_x0, sig_x0, loc_x1, sig_x1, sig_x0_x1]
    """
    if levels is None:
        levels = [0.683, 0.955]

    pdf = np.exp(logL)

    # normalize the pdf too --> though not necessary for
    # getting mean and the standard deviation
    x0_pdf = np.sum(pdf, axis=1)
    x0_pdf /= x0_pdf.sum() * (xx[1] - xx[0])

    x1_pdf = np.sum(pdf, axis=0)
    x1_pdf /= x1_pdf.sum() * (yy[1] - yy[0])

    mu_x0 = (xx * x0_pdf).sum() / x0_pdf.sum()
    mu_x1 = (yy * x1_pdf).sum() / x1_pdf.sum()

    sig_x0 = np.sqrt((xx ** 2 * x0_pdf).sum() / x0_pdf.sum() - mu_x0 ** 2)
    sig_x1 = np.sqrt((yy ** 2 * x1_pdf).sum() / x1_pdf.sum() - mu_x1 ** 2)

    sig_x0_x1 = ((xx - mu_x0) * (yy[:, None] - mu_x1) * pdf.T).sum() / pdf.sum()

    print("param1 = %.4f pm %.4f" % (mu_x0, sig_x0))
    print("param2 = %.4f pm %.4f\n" % (mu_x1, sig_x1))

    if ax is None:
        ax = plt.axes()
    CS = ax.contour(xx, yy,  cts(logL.T),
                    levels=levels, label=label, **kwargs)
    CS.collections[0].set_label(label)

    ax.set_xlim(mu_x0 - 4 * sig_x0, mu_x0 + 4 * sig_x0)
    ax.set_ylim(mu_x1 - 4 * sig_x1, mu_x1 + 4 * sig_x1)

    if plot_marg:
        xx_extent = 8 * sig_x0
        yy_extent = 8 * sig_x1

        pdf_xx_ext = x0_pdf.max() - x0_pdf.min()
        pdf_yy_ext = x1_pdf.max() - x1_pdf.min()

        ax.plot(xx, 0.2 * (x0_pdf - x0_pdf.min()) * yy_extent / pdf_xx_ext
                + ax.get_ylim()[0])
        ax.axvline(mu_x0 - sig_x0)
        ax.axvline(mu_x0 + sig_x0)
        ax.plot(0.2 * (x1_pdf - x1_pdf.min()) * xx_extent / pdf_yy_ext +
                ax.get_xlim()[0], yy)
        ax.axhline(mu_x1 - sig_x1)
        ax.axhline(mu_x1 + sig_x1)

        plt.title(r"$%s = %.3f \pm %.3f, %s = %.3f \pm %.3f$" %
                  (par_labels[0], mu_x0, sig_x0, par_labels[1], mu_x1, sig_x1))
    plt.legend()
    plt.tight_layout()
    plt.show()

    return mu_x0, sig_x0, mu_x1, sig_x1, sig_x0_x1


def combine_likelihoods(folder_name, indices, xx, yy, ax=None,
                        individual=False, **kwargs):
    """ Computes the combined likelihood surface for a given set of
    restframe wavelength indices

    Parameters:
        folder_name: folder containing the individual likelihoods
        indices: which restframe wavelengths to use
        xx: the x vector of the grid
        yy: the y vector of the grid
        ax: axes object to draw the figure on
        individual: whether to draw contours for each wavelength

    Returns:
        ax: handle on the axes object for future manipulation
        joint_estimates: Gaussian estimates of the likelihood surface
    """
    if ax is None:
        ax = plt.axes()

    joint_lnlike = np.zeros((len(xx), len(yy)))

    for index in indices:
        ll = np.loadtxt(folder_name + 'lnlike_%s.dat' % str(index))
        if individual:
            marg_estimates(xx, yy, ll.T, ax=ax,
                           plot_marg=False, levels=[0.683], label=str(index),
                           colors='k')
        joint_lnlike += ll
    joint_lnlike -= joint_lnlike.max()

    np.savetxt('joint_pdf.dat', joint_lnlike)

    # Remember: Here we are modeling the combined likelihood in x0-x1 as
    # a 2D Gaussian - these are the actual values used for the statistical
    # estiamtes before applying the stretch corrections due to LSS
    joint_estimates = marg_estimates(xx, yy, joint_lnlike.T, ax=ax,
                                     label='joint', **kwargs)

    return ax, joint_estimates


def get_stretch_factor(folder_name, indices):
    """ Computes the stretch factor using the (16-50-84) percentile estimates
    of x0 - x1 for each restframe wavelength assuming orthogonality

    Parameters:
        folder_name: folder containing the individual likelihoods and their
                     percentile estimates
        indices:  which restframe wavelengths to use

    Returns:
        stretch_x0, stretch_x1: the stretch factors along x0 and x1
    """
    x0_cen = np.zeros(len(indices))
    x0_err = np.zeros(len(indices))
    x1_cen = np.zeros(len(indices))
    x1_err = np.zeros(len(indices))

    for i, index in enumerate(indices):
        _, est_x0, est_x1 = np.loadtxt('xx_percentile_est_%d.dat' % index)

        x0_cen[i] = est_x0[0]
        x0_err[i] = (est_x0[1] + est_x0[2]) / 2.

        x1_cen[i] = est_x1[0]
        x1_err[i] = (est_x1[1] + est_x1[2]) / 2.

    res0 = get_corrfunc(x0_cen, x0_err, model=True, est=True,
                        sfx="x0_corr")
    res1 = get_corrfunc(x1_cen, x1_err, model=True, est=True,
                        sfx="x1_corr")
    stretch_x0 = res0[1] / res0[3]
    stretch_x1 = res1[1] / res1[3]

    return stretch_x0,  stretch_x1


def plot_percentile_estimates(folder_name, indices, basis='mod'):
    """ Plots the (16-50-84) percentile estimates
    of x0 - x1 for each restframe wavelength assuming orthogonality

    Parameters:
        folder_name: folder containing the individual likelihoods and their
                     percentile estimates
        indices:  which restframe wavelengths to use

    Returns:
        axs: handle to the axes object
    """
    if basis == 'mod':
        n_params = 3
        prefix = folder_name + 'xx_percentile_est_'
    else:
        n_params = 2
        prefix = folder_name + 'tg_percentile_est_'

    e_cube = np.empty((len(indices), n_params, 3))
    for i, index in enumerate(indices):
        e_cube[i] = np.loadtxt(prefix + '%d.dat' % index)

    fig, axs = plt.subplots(nrows=n_params, sharex=True)

    for i in range(n_params):
        axs[i].errorbar(indices, e_cube[:, i, 0], yerr=[e_cube[:, i, 2], e_cube[:, i, 1]],
                        fmt='.-', color='k', lw=0.6)
    plt.tight_layout()
    plt.show()

    return axs


def get_corrfunc(x, x_err=None, y=None, y_err=None, n_frac=2, viz=True,
                 model=False, est=False, sfx="corr", scale_factor=None):
    """
    Auto correlation of a signal and mean estimation

    Parameters
        x : samples of the first variable
        x_err : error vector for the first variable
        n_frac : number of pixels over which to estimate correlation wrt
                 the size of the samples
        viz : plot the correlation function
        model : model the correlation function
        est : Get estimates on the best-fit values using the covariance
              matrices estmated

    Returns:
        loc_simple, sig_simple : the mean and uncertainty using simple
                                 weighted estimation
        loc, sig : the mean and uncertainty on the location parameter
                   after incorporating correlations
    """
    if y is None:
        y = x.copy()
        y_err = x_err.copy()

    # Direct weighted average along each dimension
    loc_simple = np.sum(x / x_err ** 2) / np.sum(1 / x_err ** 2)
    sig_simple = 1. / np.sqrt(np.sum(1 / x_err ** 2))

    npp = len(x)
    x_data, y_data = x - x.mean(), y - y.mean()

    coef = [np.sum(x_data[:npp - j] * y_data[j:]) / \
            np.sqrt(np.sum(x_data[:npp - j] ** 2) * np.sum(y_data[j:] ** 2)) for j in
            range(npp // n_frac)]
    np.savetxt(sfx + ".dat", coef)

    if viz:
        fig, ax = plt.subplots(1)
        ax.plot(coef, '-k')
        ax.axhline(0, ls='--')
        ax.set_xlabel(r"$|i-j|$")
        ax.set_ylabel(r"$\xi(|i-j|)$")

    if model:
        def model_exp(x, cl):
            return np.exp(- x / cl)  # A simple exponential model

        if scale_factor is None:
            popt_exp, __ = curve_fit(model_exp, np.arange(npp // n_frac)[:5],
                                     coef[:5])
        else:
            popt_exp = [scale_factor]

        if viz:
            rr = np.linspace(0, 50, 500)
            val = model_exp(rr, *popt_exp)
            ax.plot(rr, val, '-r')
            ax.text(10, 0.8, "$r_{x_0}=%.1f$" % popt_exp[0])
            ax.set_xlim(0, 50)

        if est:
            if x_err is None:
                raise TypeError("Requires errorbars on the samples")

            # Obtain band-diagonal correlation matrix
            from scipy.linalg import toeplitz

            rr = np.arange(npp)
            val = model_exp(rr, *popt_exp)
            Xi = toeplitz(val)

            cov = np.diag(x_err).dot(Xi.dot(np.diag(y_err)))

            if np.any(np.linalg.eigh(cov)[0] < 0):
                raise TypeError("The covariance matrix \
                                 is not positive definite")

            # Minimization using iminuit
            from iminuit import Minuit
            ico = np.linalg.inv(cov)

            def chi2(mu):
                dyi = x - mu
                return dyi.T.dot(ico.dot(dyi))

            mm = Minuit(chi2,
                        mu=0.2, error_mu=x.std(), fix_mu=False,
                        errordef=1., print_level=-1)
            mm.migrad()
            loc, sig = mm.values["mu"], mm.errors["mu"]

            if viz:
                fig, ax = plt.subplots(figsize=(9, 3))
                ax.errorbar(rr, x, x_err, fmt='.-', lw=0.6, color='k')
                ax.fill_between(rr, loc + sig,
                                loc - sig, color='r', alpha=0.2)
                plt.show()
            return loc_simple, sig_simple, loc, sig


def get_intrinsic_covariance(locs, covs):
    """ Computes the intrinsic covariance matrix in 2D

    Parameters:
        locs: the central values as a vector
        covs: the corresponding covariances

    Returns:
        loc: the best-fit location
        cov: the covariance of the above best-fit
        sys_cov: the estimated intrinsic covariance
        one_sig, two_sig: points along the one-sigma and two-sigma
                          confidence intervals
    """
    from numpy.linalg import inv, det
    from iminuit import Minuit
    from scipy.stats import chi2 as chisq

    def neg_ln_like(x0, x1, lnsig1, rho, lnsig2):
        sig_x0, sig_x1 = np.exp(lnsig1), np.exp(lnsig2)
        sig_x0_x1 = rho * sig_x0 * sig_x1

        cov_int = np.array([[sig_x0 ** 2, sig_x0_x1],
                            [sig_x0_x1, sig_x1 ** 2]])

        mod_cov = covs + cov_int
        temp = [np.dot([x0, x1] - locs[i], np.dot(inv(mod_cov[i]), [x0, x1] - locs[i])) +\
                np.log(det(mod_cov[i])) for i in range(len(locs))]
        foo = np.sum(temp, 0)
        return foo

    # optimize using Minuit
    mm = Minuit(neg_ln_like,
                x0=0, error_x0=0.1,
                x1=0, error_x1=0.01,
                lnsig1=-2, error_lnsig1=0.1, limit_lnsig1=(-6, 2),
                lnsig2=-2, error_lnsig2=0.2, limit_lnsig2=(-6, 2),
                rho=0, error_rho=0.1, limit_rho=(-1, 1),
                errordef=1, print_level=-1)

    __ = mm.migrad()
    __ = mm.minos()

    # Relevant data for plotting contours from minuit
    mm.set_errordef(chisq.ppf(0.683, 2))
    _, _, one_sig = mm.mncontour('x0', 'x1', numpoints=200)
    mm.set_errordef(chisq.ppf(0.955, 2))
    _, _, two_sig = mm.mncontour('x0', 'x1', numpoints=200)

    mm.set_errordef(1.)
    loc = np.array([mm.values['x0'], mm.values['x1']])
    corr = mm.np_matrix(correlation=True)[:2, :2]

    # Error bars - from MINOS
    mm_errors_x0 = (mm.merrors[('x0', 1.0)] - mm.merrors[('x0', -1.0)]) / 2.
    mm_errors_x1 = (mm.merrors[('x1', 1.0)] - mm.merrors[('x1', -1.0)]) / 2.

    error_mat = np.diag([mm_errors_x0, mm_errors_x1])
    cov = error_mat.T.dot(corr.dot(error_mat))

    # intrinsic covariance matrix
    mm_v = mm.np_values()

    sys_x0_x1 = mm_v[3] * np.exp(mm_v[2]) * np.exp(mm_v[4])
    sys_cov = np.array([[np.exp(mm_v[2]) ** 2, sys_x0_x1],
                        [sys_x0_x1, np.exp(mm_v[4]) ** 2]])

    return loc, cov, sys_cov, one_sig, two_sig

# =============================================================================
# THE ROUTINES BELOW SHOULD ONLY BE RUN FROM THE CORRECT FOLDER


def get_statistical_estimates(template, indices, xx, yy, n_bins=7):
    # These are obtained from the full 2D likelihood surfaces
    loc_mod = np.zeros((n_bins, 2))
    cov_mod_no_stretch = np.zeros((n_bins, 2, 2))
    cov_mod_with_stretch = np.zeros((n_bins, 2, 2))

    print("======= COMPUTING STATISTICAL ESTIMATES =======")
    for i in range(n_bins):
        # statistical likelihoods without LSS correlations
        folder_name = template.format(i+1)

        _, estimates_no_stretch = combine_likelihoods(folder_name,
                                                      indices, xx, yy)
        mu_x0, sig_x0, mu_x1, sig_x1, sig_x0_x1 = estimates_no_stretch

        loc_vec = np.array([mu_x0, mu_x1])
        cov_mat = np.array([[sig_x0 ** 2, sig_x0_x1],
                            [sig_x0_x1, sig_x1 ** 2]])

        # stretch factor
        res = get_stretch_factor(folder_name, indices)

        st_x0 = res[0][1] / res[0][3]
        st_x1 = res[1][1] / res[1][3]

        # expand the confidence intervals
        st_mat = np.diag([st_x0, st_x1])
        cov_new = st_mat.T.dot(cov_mat.dot(st_mat))

        # Assign to variables to store later
        loc_mod[i] = loc_vec
        cov_mod_no_stretch[i] = cov_mat
        cov_mod_with_stretch[i] = cov_new

    # Write all relevant information in the modified basis to files
    print("======= WRITING TO FILES =======")
    np.savetxt("central_values_bins.dat", loc_mod)
    np.savetxt("cov_mat_bins_no_stretch.dat",
               cov_mod_no_stretch.reshape(n_bins, 4))
    np.savetxt("cov_mat_bins_with_stretch.dat",
               cov_mod_with_stretch.reshape(n_bins, 4))


def get_systematic_estimates(template, bins_to_use=None):
    if bins_to_use is None:
        bins_to_use = np.arange(7)

    n_bins = len(bins_to_use)

    # load in the statistical estimates
    locations = template + "central_values_bins.dat"
    covariances = template + "cov_mat_bins_with_stretch.dat"

    # get the systematic matrix modeled as intrinsic covariance
    res = get_intrinsic_covariance(locations, covariances)
    loc, cov, sys_cov, one_sig, two_sig = res

    print("======= WRITING TO FILES =======")
    np.savetxt("best_fit_loc_{}.dat".format(n_bins),  loc)
    np.savetxt("best_fit_cov_mat_{}.dat".format(n_bins), cov)
    np.savetxt("best_fit_sys_mat_{}.dat".format(n_bins), sys_cov)
    np.savetxt("contour_one_sigma_{}.dat".format(n_bins), one_sig)
    np.savetxt("contour_two_sigma_{}.dat".format(n_bins), two_sig)
