import read_hwo_yaml as test
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u


def generate_wavelength_grid(lammin, lammax, R):
    # working in log space to maintain a constant resolving power across the grid

    lammin_log = np.log(lammin)
    lammax_log = np.log(lammax)
    dlam_log = np.log(1 + 1 / R)  # step size in log space

    lam_log = np.arange(lammin_log, lammax_log, dlam_log)
    lam = np.exp(lam_log)
    return lam


R = 1000
lammin = 0.5
lammax = 2.0
internal_lam = generate_wavelength_grid(lammin, lammax, R) * u.um
# print(internal_lam)
# print(internal_lam[:-1]/np.diff(internal_lam))

ci = test.CI(internal_lam)
print(ci)
ci.calculate_throughput()
ci.plot()

telescope = test.TELESCOPE(internal_lam)
telescope.load_EAC1()
telescope.plot()

detector = test.DETECTOR(internal_lam)
detector.load_imager()
detector.plot()

detector = test.DETECTOR(internal_lam)
detector.load_IFS()
detector.plot()

import ipdb

ipdb.set_trace()
