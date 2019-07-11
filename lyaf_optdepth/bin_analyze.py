import os
import shutil
import numpy as np

from timeit import default_timer as timer
from multiprocessing import Pool
from scipy.stats import binned_statistic
from functools import partial

# local imports
from lyaf_optdepth import corrections, create_comp

import importlib
importlib.reload(create_comp)
from Scripts import comp_simple


def find_zbins(z, z_start=2.1, delta_z=None, min_num=50):
    """
    Create bins in quasar redshifts with a given minimum
    number of objects in each bin.

    Parameters:
    ----------
    z : the input quasar redshifts
    z_start : starting redshift for binning
    delta_z: a list of bin widths
    min_num : target minimum number of objects per bin

    Returns:
    -------
    curr_zbins : sequence of the required bins
    """

    if delta_z is None:
        deltaz = [0.12, 0.18, 0.24]

    curr_zbins, curr_z = [z_start], z_start

    is_end = False
    while True:
        for j, delta in enumerate(deltaz):
            new_z = curr_z + delta
            pos = (z > curr_z) & (z <= new_z)

            if sum(pos) > min_num:
                curr_z = new_z
                curr_zbins.append(curr_z)
                break  # break for loop
            elif j == len(deltaz) - 1:
                is_end = True

        if is_end:
            break  # break while loop

    return curr_zbins


def hist_weights(p1, p2, z, zbins, n_chop=4, truncated=True):
    """ Function to get the appropriate weights, such that
    quasars i n different bins all have the same probability
    distribution in different redshift bins

    * This assumes that the redshifts z have been properly
    truncated to be in the zbins range.
    """
    if not truncated:
        ixs = (z >= zbins[0]) & (z < zbins[-1])
        z = z[ixs]
        p1, p2 = p1[ixs], p2[ixs]

    n_zbins = len(zbins) - 1

    # Left closed, right open partitioning
    z0_bins = zbins
    z0_bins[-1] += 0.001
    z_ind = np.digitize(z, z0_bins)

    chop1 = np.linspace(min(p1), max(p1), n_chop)
    chop2 = np.linspace(min(p2), max(p2), n_chop)

    # CREATING A 3D DATACUBE OF WEIGHTS
    cube = np.zeros((n_zbins, n_chop - 1, n_chop - 1))

    for i in range(n_zbins):
        ind = (z >= zbins[i]) & (z < zbins[i + 1])
        cube[i] = np.histogram2d(p1[ind], p2[ind], bins=(chop1, chop2))[0]

    # Trim bins with no objects
    # Outer - parameter; Inner - redshift
    for i in range(n_chop - 1):
        for j in range(n_chop - 1):
            # Sets all bins to 0 if any one bin has no objects in it
            if 0 in cube[:, i, j]:
                cube[:, i, j] = 0

    cube_sum = np.sum(cube, axis=0)

    # A. NORMALIZED WEIGHTS ACROSS ALL REDSHIFTS
    p0_bins, p1_bins = chop1, chop2

    # <-- Required since histogram2d and digitize have different
    # binning schemes
    p0_bins[-1] += 0.001
    p1_bins[-1] += 0.001

    foo = np.digitize(p1, p0_bins)
    blah = np.digitize(p2, p1_bins)

    weight_mat = cube_sum / cube
    weight_mat[np.isnan(weight_mat)] = 0

    # To obtain consistent weights across all redshifts
    weight_mat /= np.linalg.norm(weight_mat, axis=(1, 2))[:, None, None]

    # Final histogram weights to be applied
    h_weights = weight_mat[z_ind - 1, foo - 1, blah - 1]

    return h_weights


def analyze(binObj, task='skewer', frange=None, distort=True, CenAlpha=None,
            histbin=False, statistic='mean', suffix='temp', overwrite=False,
            skewer_index=None, zq_cut=[0, 5], parallel=False, tt_bins=None,
            verbose=True, nboot=100, calib_kwargs=None, skewer_kwargs=None):
    """
    Function to perform important operations on the binObj

    Parameters:
        binObj: An instance of the bin_class
        task: one of ["data_points", "calibrate", "composite", "skewer"]
        frange: the Lyman Alpha forest ranges used for the analysis
        distort: warp the spectra to a common spectral index
        CenAlpha: the common spectral index to warp to
        histbin: perform histogram rebinninb
        statistic: statistic to use when creating composites [task="composite"]
        suffix: name of the file to write to
        overwrite: overwrite the skewer in the LogLikes folder if duplicates
        skewer_index: index of the skewers in the forest range (frange) to use
        zq_cut: allows to perform a cut in the quasar redshift
        parallel: whether to run the skewers in parallel
        tt_bins: Bins in lyman alpha redshift to use for task="data_points"
        calib_kwargs: additional keyword arguments for task="calibrate"
        skewer_kwargs: additional keyword arguments for task="skewer"
    """

    if frange is None:
        frange = [1070, 1160]

    lyInd = np.where((binObj.wl > frange[0]) & (binObj.wl < frange[1]))[0]

    if skewer_index is None:
        skewer_index = range(len(lyInd))
    else:
        skewer_index = np.atleast_1d(skewer_index)

    outfile = task + '_' + suffix

    if task == 'skewer' or task == 'data_points':
        if verbose:
            print('Total skewers available: {}, skewers analyzed in this '
                  'run: {}'.format(len(lyInd), len(skewer_index)))

        myspec = binObj._flux[:, lyInd[skewer_index]]
        myivar = binObj._ivar[:, lyInd[skewer_index]]
        zMat = binObj._zAbs[:, lyInd[skewer_index]]
        mywave = binObj.wl[lyInd[skewer_index]]
    else:
        myspec, myivar, zMat = binObj._flux, binObj._ivar, binObj._zAbs
        mywave = binObj.wl

    myz, myalpha = binObj._zq, binObj._alpha

    # selecting according to quasar redshifts
    zq_mask = (myz > zq_cut[0]) & (myz < zq_cut[1])

    myspec = myspec[zq_mask]
    myivar = myivar[zq_mask]
    zMat = zMat[zq_mask]
    myz, myalpha = myz[zq_mask], myalpha[zq_mask]

    # B. DATA PREPROCESSING ---------------------------------------------------
    if histbin:
        # Histogram binning in parameter space
        myp1, myp2 = binObj._par1, binObj._par2

        myzbins = find_zbins(myz)
        hInd = np.where((myz >= myzbins[0]) & (myz < myzbins[-1]))

        # Modify the selection to choose only objects that fall in the
        # zbins range
        myz, myalpha = myz[hInd], myalpha[hInd]
        myp1, myp2 = myp1[hInd], myp2[hInd]

        myspec, myivar = myspec[hInd], myivar[hInd]
        zMat = zMat[hInd]

        if binObj._hWeights is None:
            h_weights = hist_weights(myp1, myp2, myz, myzbins)
            binObj._hWeights = h_weights

            myivar = myivar * h_weights[:, None]
        else:
            myivar = myivar * binObj._hWeights[:, None]

    if distort:
        # Distort spectra in alpha space
        outfile += '_distort'

        if CenAlpha is None:
            CenAlpha = np.median(myalpha)
        distortMat = np.array([(mywave / 1450.) ** ele for
                              ele in (CenAlpha - myalpha)])

        myspec *= distortMat
        myivar /= distortMat ** 2

        if verbose:
            print('All spectra distorted to alpha:', CenAlpha)

    # C. CALIBRATION VS ESTIMATION --------------------------------------------

    if task == "data_points":
        print("Make sure that the reconstructed continuum has been run using "
              "the same frange as that being used right now!")
        # Data points for the transmission, using a continuum as the base
        if binObj.continuum is None:
            raise AttributeError("Set the reconstructed continuum for the"
                                 "bin first!!!")

        ivar_mask = (myivar > 0).flatten()
        zLyAs = zMat.flatten()
        zLyAs = zLyAs[ivar_mask]

        # bin centers for the redshift-transmission plot
        if tt_bins is None:
            tt_bins = np.linspace(zLyAs.min(), zLyAs.max(), 40)
        tt_cens = (tt_bins[1:] + tt_bins[:-1]) / 2.

        # errors from t0-gamma fluctuations
        # We are not going to use this in the paper !!!
        tt_binned = np.zeros((len(binObj.continuum), len(tt_cens)))
        for i in range(len(binObj.continuum)):
            tt = (myspec / binObj.continuum[i]).flatten()
            tt = tt[ivar_mask]

            tt_binned[i] = binned_statistic(zLyAs, tt, statistic=np.mean,
                                            bins=tt_bins).statistic

        continuum = binObj.continuum.mean(0)

        # estimates of the transmission central values - errors obtained
        # using bootstrap as below
        tt_cen = (myspec / continuum).flatten()
        tt_cen = tt_cen[ivar_mask]
        tt_data = binned_statistic(zLyAs, tt_cen, statistic=np.mean,
                                   bins=tt_bins).statistic
        # tt_std = binned_statistic(zLyAs, tt_cen, statistic=np.std,
        #                           bins=tt_bins).statistic
        # tt_counts = binned_statistic(zLyAs, None, statistic='count',
        #                              bins=tt_bins).statistic

        # errors from bootstrapping
        print("Computing bootstrap samples of transmission")
        tt_boot = np.zeros((nboot, len(tt_cens)))
        for i in range(nboot):
            np.random.seed()
            ixs = np.random.randint(0, len(myivar), len(myivar))

            sp_boot, iv_boot = myspec[ixs], myivar[ixs]
            zz_boot = zMat[ixs]

            ivar_mask = (iv_boot > 0).flatten()
            zLyAs = zz_boot.flatten()
            zLyAs = zLyAs[ivar_mask]

            tt = (sp_boot / continuum).flatten()
            tt = tt[ivar_mask]
            tt_boot[i] = binned_statistic(zLyAs, tt, statistic=np.mean,
                                          bins=tt_bins).statistic

        # Save this to a file for future use -
        # Use this for the analysis of figure 6 <--
        data_full = np.array([tt_cens, tt_data, *tt_boot])
        np.savetxt("data_points_" + binObj.name + ".dat", data_full)

        return tt_cens, tt_data, tt_binned, tt_boot  # , tt_std / np.sqrt(tt_counts)

    if task == 'calibrate':
        ixs = (myz > 1.6) & (myz < 4)
        print('Number of spectra used for calibration are: %d' % ixs.sum())

        rest_range = [[1280, 1290], [1320, 1330], [1345, 1360], [1440, 1480]]

        # normalization range used
        obs_min, obs_max = 4600, 4640

        corrections.calibrate(binObj.wl, myspec[ixs], myivar[ixs], myz[ixs],
                              rest_range, obs_min, obs_max, binObj.name, True)

    # D. COMPOSITE CREATION IF SPECIFIED --------------------------------------

    if task == 'composite':
        # Create composites using the spectra
        # zbins = find_zbins(myz)

        zbins = np.arange(2.1, 4.5, 0.05)
        # comp_simple.compcompute(myspec, myivar, myz, mywave,
        #                         zbins, statistic, outfile)
        create_comp.create_comp(myspec, myivar, myz,
                                mywave, zbins, outfile)

    # E. LIKELIHOOD SKEWER ----------------------------------------------------
    if task == 'skewer':
        currDir = os.getcwd()
        destDir = '../LogLikes' + '/Bin_' + outfile +\
                  str(frange[0]) + '_' + str(frange[1])  # <--

        if not os.path.exists(destDir):
            os.makedirs(destDir)
        else:
            if overwrite:
                shutil.rmtree(destDir)
                os.makedirs(destDir)
        os.chdir(destDir)

        start = timer()

        # Do not plot graphs while in parallel
        res = None
        if parallel:
            pass
        #     print('Running in parallel now!')

        #     myfunc_partial = partial(mcmc_skewer.mcmcSkewer, **skewer_kwargs)

        #     pool = Pool()
        #     res = pool.map(myfunc_partial,
        #                    zip(np.array([zMat, myspec, myivar]).T, skewer_index))
        #     pool.close()
        #     pool.join()
        # else:
        #     for j, ele in enumerate(skewer_index):
        #         res = mcmc_skewer.mcmcSkewer(
        #             [np.array([zMat[:, j], myspec[:, j], myivar[:, j]]).T, ele],
        #             **skewer_kwargs)

        stop = timer()
        print('Time elapsed:', stop - start)

        os.chdir(currDir)

        return mywave, res


def reconstruct(binObj, tau_mean, tau_cov, niter=100, frange=None,
                verbose=True, b_kwargs={}):
    """
    Create functional pdf of reconstructed continuum using a
    pdf on optical depth parameters

    Parameters:
    ----------
    bin      : the binObj on which to apply the function
    tau_mean : location of optical depth params
    tau_cov  : their covariance matrix
    niter    : number of continuum evaluations following the pdf

    Returns:
    ---------
    A 2D array of size niter * n_forest_pixels
    """

    if frange is None:
        frange = [1070, 1160]

    # Wavelength to return for plotting
    ixs = (binObj.wl > frange[0]) & (binObj.wl < frange[1])

    # Sample points from the Gaussian pdf
    if tau_cov is None:
        ln_tau0, gamma = (np.atleast_2d(tau_mean)).T
        niter = 1
    else:
        ln_tau0, gamma = np.random.multivariate_normal(tau_mean,
                                                       tau_cov, size=niter).T
    tau0 = np.exp(ln_tau0)

    # Create 2d array to store the result
    data = np.zeros((niter, ixs.sum()))

    # Loop through and aggregate the results
    for i in range(niter):
        s_kwargs = {'logdef': 4, 'truths':  [tau0[i], gamma[i]]}

        __, data[i] = analyze(binObj, parallel=True, frange=frange,
                              verbose=verbose,
                              skewer_index='all', skewer_kwargs=s_kwargs,
                              **b_kwargs)

    # Save the reconstructed continuum as an attribute
    binObj.continuum = data
    return binObj.wl[ixs], data
