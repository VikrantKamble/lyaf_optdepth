import numpy as np

drqcat = "/uufs/astro.utah.edu/common/home/u0882817/" + \
          "Work/lyaf_optdepth/data/DR12Q.fits"
drq_dataext = 1

COEFF0 = 2.73
COEFF1 = 0.0001
NPIX = 8140

paramstowrite = "PLATE, MJD, FIBERID, Z_PCA, Z_VI, ALPHA_NU," +\
                " REWE_CIV, FWHM_CIV, PSFMAG, ERR_PSFMAG"

cat_file = "/uufs/astro.utah.edu/common/uuastro/astro_data/vikrant/cat/cat_sep16"

skyline_file = "/uufs/astro.utah.edu/common/home/u0882817/" +\
               "Work/lyaf_optdepth/data/dr9-sky-mask.txt"

spec_dir = "/uufs/chpc.utah.edu/common/home/sdss00/ebosswork/eboss/" +\
           "spectro/redux/v5_10_0/spectra/lite"

dla_file = "/uufs/astro.utah.edu/common/home/u0882817/Work/" +\
           "lyaf_optdepth/data/DLA_DR12tmp_v1.dat"

drq_z = "Z_PCA"
zmin = 1.6
run = 1

author = "Vikrant Kamble"

catalog = "cat_jan30_v_5_10_0_Z_PCA_1.6_0.fits"
attr = "attributes.fits"
wl = 10 ** (2.73 + np.arange(8140) * 10 ** -4)
drq_z = "Z_VI"

input_calib = "calib_helion.txt"
input_var_correct = "var_correct.txt"

lya = 1215.67

shift = np.array([-5.27, 3.21])
tilt = np.array([[-0.8563,  0.5165], [0.5165,  0.8563]])
