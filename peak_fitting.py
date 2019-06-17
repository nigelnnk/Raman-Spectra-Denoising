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
    ds = sf.convo_filter_n(-np.diff(ans), baseline_window, 1)  # difference must be negated for correct calculations
    cs = ds - sf.convo_filter_n(ds, baseline_window, 10)
    return ds, cs


def detect_peaks(spectrum, wavenumbers):
    # TODO documentation for this function
    # TODO how to extract peaks using zero-max-zero-min-zero pattern
    sign_change = np.asarray(np.sign(spectrum[:-1]) != np.sign(spectrum[1:])).nonzero()

    zeros_y = spectrum[sign_change]
    zeros_x = wavenumbers[sign_change]
    diff = np.diff(spectrum)
    maxima = np.asarray(np.sign(diff[:-1]) > np.sign(diff[1:])).nonzero()
    minima = np.asarray(np.sign(diff[:-1]) < np.sign(diff[1:])).nonzero()

    h_y = spectrum[maxima]
    significant_h = np.extract(h_y/np.amax(h_y) > 0.1, maxima)
    h_y = spectrum[significant_h]
    h_x = wavenumbers[significant_h]

    l_y = spectrum[minima]
    significant_l = np.extract(l_y / np.amin(l_y) > 0.1, minima)
    l_y = spectrum[significant_l]
    l_x = wavenumbers[significant_l]
    return zeros_x, zeros_y, h_x, h_y, l_x, l_y


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
