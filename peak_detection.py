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


def detect_peaks(spectrum, diff_spectrum, wavenumbers):
    """
    Peak detection as mentioned briefly in [2], but without any algorithm provided. This
    is original code, not referenced from any paper.

    By finding maxima and minima in the differentiated spectrum, one can identify where
    the peaks are, as peaks have a pattern of zero-maxima-zero-minima-zero. Hence, these
    are used to determine if a peak is present by analysing if each maxima has a succeeding
    minima. Additionally, the maxima and minima must be of a certain height, determined by
    the cutoff value of 10% of the maximum height in the graph.

    :param spectrum: Noisy spectrum as given by raw data
    :param diff_spectrum: Differentiated spectrum, which can be produced by get_corrected_spectrum()
    :param wavenumbers: Numpy array of wavenumbers that correspond to the spectrum
    :return: Two dictionaries are returned,for different display purposes:
             Results_diff returns the zeros, troughs and peaks of the differentiated spectrum
             Results_original returns the peaks and positions for the original spectrum
    """
    cutoff_diff = 0.05
    cutoff_original = 0.1

    # Change in sign for array signifies a zero crossing
    sign_change = np.asarray(np.sign(diff_spectrum[:-1]) != np.sign(diff_spectrum[1:])).nonzero()

    zeros_y = diff_spectrum[sign_change]
    zeros_x = wavenumbers[sign_change]
    diff = np.diff(diff_spectrum)
    # Changes in sign correspond to either peaks or troughs
    maxima = np.asarray(np.sign(diff[:-1]) > np.sign(diff[1:])).nonzero()
    minima = np.asarray(np.sign(diff[:-1]) < np.sign(diff[1:])).nonzero()

    # Calculate (x,y) values of peaks
    h_y = diff_spectrum[maxima]
    significant_h = np.extract(h_y / np.amax(h_y) > cutoff_diff, maxima)
    h_y = diff_spectrum[significant_h]
    h_x = wavenumbers[significant_h]
    highs_x = np.copy(h_x)

    # Calculate (x,y) values of troughs
    l_y = diff_spectrum[minima]
    significant_l = np.extract(l_y / np.amin(l_y) > cutoff_diff, minima)
    l_y = diff_spectrum[significant_l]
    l_x = wavenumbers[significant_l]
    lows_x = np.copy(l_x)

    # Summarises results into a dictionary
    results_diff = {"zeros_x": zeros_x,
                    "zeros_y": zeros_y,
                    "highs_x": highs_x,
                    "highs_y": h_y,
                    "lows_x": lows_x,
                    "lows_y": l_y, }

    peaks = []
    peak_widths = []
    peak_heights = []
    smooth = sf.convo_filter_n(spectrum, 5, 20)
    max_height = np.amax(smooth)
    # Iterating through all peaks (peaks must come first before troughs)
    while h_x.size > 0:
        while l_x.size > 0 and l_x[0] < h_x[0]:
            l_x = np.delete(l_x, 0)

        if l_x.size == 0:
            break

        # Search for position of the peaks and troughs and see if they are in succession
        h_position = np.searchsorted(zeros_x, h_x[0])
        l_position = np.searchsorted(zeros_x, l_x[0])

        if h_position + 1 == l_position:
            height = smooth[np.searchsorted(wavenumbers, zeros_x[h_position])]
            # print("{}\t{}".format(zeros_x[h_position], height))
            if height / max_height >= cutoff_original and l_position + 1 < zeros_x.size:
                peaks.append(zeros_x[h_position])
                peak_heights.append(height)
                peak_widths.append([zeros_x[h_position - 1], zeros_x[l_position + 1]])

        h_x = np.delete(h_x, 0)

    # Summarises results into a dictionary
    peaks = np.array(peaks)
    peak_widths = np.array(peak_widths)
    peak_heights = np.array(peak_heights)
    results_original = {"peaks_x": peaks,
                        "peaks_y": peak_heights,
                        "peak_widths": peak_widths}
    return results_diff, results_original


def normalise(y):
    return y / np.max(y)


def test_case_2():
    """
    This is a test case for [2] A simple background elimination method for Raman spectra.

    It showcases how it is a useful tool when the spectrum baseline changes drastically.
    However, for most Raman spectra, the baseline changes slowly. Nevertheless, this is
    still implemented because of its benefits in clearing up the differentiated spectra.
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
