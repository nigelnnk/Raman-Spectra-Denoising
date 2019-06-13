import numpy as np
from numba import *
import matplotlib.pyplot as plt
from scipy import signal as sig
import generate_spectrum as gs
import smoothing_functions as sf


def get_corrected_spectrum(spectrum, noise_window, baseline_window):
    """
    Processes the given spectrum as detailed in [2] A simple background elimination method
    for Raman spectra by Baek et. al. (2009) to remove baseline errors from spectra.
    :param spectrum: Spectrum to be processed
    :param noise_window: Number of datapoints, usually less than smallest peak width
    :param baseline_window: Number of datapoints, usually larger than widest peak width
    :return: differentiated and baseline-corrected differentiated spectra
    """
    ans = sf.convo_filter_n(spectrum, noise_window, 3)
    ds = sf.convo_filter_n(np.diff(ans), baseline_window, 1)
    cs = ds - sf.convo_filter_n(ds, baseline_window, 10)
    return ds, cs


def normalise(y):
    return y/np.max(y)


def test_case_2():
    """
    This is a test case for [2] A simple background elimination method for Raman spectra.

    It showcases how it is a useful tool when the spectrum baseline changes drastically.
    However, for most Raman spectra, this is not needed as the baseline changes slowly or
    has already been corrected. Nevertheless, this is useful for illustration.
    """
    a = np.linspace(0, 5, 1000)
    b = ((a - 2.5) ** 3) + gs.lorentzian(a, 3, 0.2, 5) + np.random.normal(size=a.size) / 2
    ds, cs = get_corrected_spectrum(b, 5, 53)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].plot(a, b)
    ax[0, 0].set_title("Signal")
    ax[0, 1].plot(a, sf.convo_filter_n(b, 5, 20))
    ax[0, 1].set_title("SG smoothed")
    ax[1, 0].plot(a[:-1], ds, c='b', label="differentiated spectrum")
    ax[1, 0].plot(a[:-1], np.diff(((a - 2.5) ** 3)), c='r', label="differentiated baseline")
    ax[1, 0].set_title("Differentiated")
    ax[1, 0].legend()
    ax[1, 1].plot(a[:-1], cs)
    ax[1, 1].set_title("Corrected")
    plt.show()


if __name__ == "__main__":
    test_case_2()
