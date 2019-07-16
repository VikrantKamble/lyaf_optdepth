"""
Main module that streamlines the steps taken in the analysis
till likelihood estimation
"""
import numpy as np

from lyaf_optdepth import qso_class, bin_class
from lyaf_optdepth import bin_analyze, fitting


def create_comp():
    qso = qso_class.QSO()
    qso.load_data()

    bin1 = bin_class.binObj('bin1', qso, ['ALPHA_V0', 'EW_V0'],
                            [[-2.8, -2.13], [20, 40]], snt=[5, 120])
    bin2 = bin_class.binObj('bin2', qso, ['ALPHA_V0', 'EW_V0'],
                            [[-2.8, - 2.13], [40, 60]], snt=[5, 120])
    bin3 = bin_class.binObj('bin3', qso, ['ALPHA_V0', 'EW_V0'],
                            [[-2.13, -1.46], [20, 33.3]], snt=[5, 120])
    bin4 = bin_class.binObj('bin4', qso, ['ALPHA_V0', 'EW_V0'],
                            [[-2.13, -1.46], [33.3, 46.6]], snt=[5, 120])
    bin5 = bin_class.binObj('bin5', qso, ['ALPHA_V0', 'EW_V0'],
                            [[-2.13, -1.46], [46.6, 60]], snt=[5, 120])
    bin6 = bin_class.binObj('bin6', qso, ['ALPHA_V0', 'EW_V0'],
                            [[-1.46, -0.8], [20, 40]], snt=[5, 120])
    bin7 = bin_class.binObj('bin7', qso, ['ALPHA_V0', 'EW_V0'],
                            [[-1.46, -0.8], [40, 60]], snt=[5, 120])
    bin_list = [bin1, bin2, bin3, bin4, bin5, bin6, bin7]

    # MAKE THE COMPOSITES
    for i, bin in enumerate(bin_list):
        suffix = "bin{}_simple".format(i+1)
        _ = bin_analyze.analyze(bin, task='composite',
                                frange=[1070, 1160], suffix=suffix)
    print("CREATING COMPOSITES FOR ALL THE BINS FINISHED")


def create_likelihoods():
    n_skewers = 351
    for i in range(6, 7):
        fit = fitting.Optimizer()

        comp_file = "./bin/composites/" +\
                    "composite_bin{}_simple_distort.fits".format(i+1)
        fit.set_data_from_fits(comp_file)

        save_path = "./bin/likelihoods/bin{}".format(i+1)
        indices = np.arange(n_skewers)

        _ = fit.fit_batch(save_path, indices, parallel=True)


if __name__ == "__main__":
    # create_comp()
    create_likelihoods()
