import numpy as np
from numba import *
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy import sparse
from scipy.sparse.linalg import splu
import generate_spectrum as gs
import smoothing_functions as sf
import saveLoadCSV as sl


def corrected_diff_spectrum(spectrum, noise_window=5, baseline_window=23):
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


def poly_baseline_corrected(spectrum, wavenumbers, power=7):
    """
    This is baseline correction as explained in Baseline correction by improved iterative
    polynomial fitting with automatic threshold by Gan et. al. (2006).

    It attempts to fit the baseline via an "averaging" of points, and excluding data above
    this average. This is repeated until a stopping criterion is reached.

    :param spectrum: Numpy array containing spectra data
    :param wavenumbers: Numpy array of wavenumbers in spectra
    :param power: Power of polynomial that is used to fit baseline. Default is 7.
    :return:
    """

    # Iterating through baseline = X (X.T X)^-1 X.T y
    x = np.ones_like(wavenumbers)
    for i in range(1, power + 1):
        x = np.vstack((x, np.power(wavenumbers, i)))
    x = np.transpose(x)
    xtxi = np.linalg.inv(np.matmul(x.T, x))
    multiplier = np.matmul(np.matmul(x, xtxi), x.T)
    b_prev = spectrum
    b = np.matmul(multiplier, spectrum)
    new_spectrum = np.where(spectrum > b, b, spectrum)

    # Stopping criterion is not defined well in paper, so I guessed what it's supposed to be
    while np.linalg.norm(b-b_prev)/np.linalg.norm(b_prev) > 0.01:
        b_prev = b.copy()
        b = np.matmul(multiplier, new_spectrum)
        new_spectrum = np.where(new_spectrum > b, b, new_spectrum)
    return new_spectrum, b


def als_baseline(spectrum, lmbd, p=0.05):
    N = spectrum.size
    diagonals = np.array([1, -2, 1])
    D = sparse.diags(diagonals, np.arange(3), (N-2, N), format='csc')
    diff_matrix = D.T.dot(D)
    weights = np.ones_like(spectrum)
    for i in range(10):
        W = sparse.diags(weights, 0, (N, N))
        new_baseline = splu(W + lmbd * diff_matrix).solve(W.dot(spectrum))
        weights = np.where(spectrum > new_baseline, p, 1 - p)
    return new_baseline, weights


def detect_peaks(raw_spectrum, diff_spectrum, noise_mean=-1, noise_stdd=-1):
    """
    Peak detection as mentioned briefly in [2], but without any algorithm provided. This
    is original code, not referenced from any paper.

    By finding maxima and minima in the differentiated spectrum, one can identify where
    the peaks are, as peaks have a pattern of zero-maxima-zero-minima-zero. Hence, these
    are used to determine if a peak is present by analysing if each maxima has a succeeding
    minima. Additionally, the maxima and minima must be of a certain height, determined by
    the standard deviation of noise level in the spectrum.

    :param raw_spectrum: Noisy spectrum as given by raw data
    :param diff_spectrum: Differentiated spectrum, which can be produced by corrected_diff_spectrum()
    :return: Two dictionaries are returned,for different display purposes:
             Results_diff returns indexes of zeros, troughs and peaks of the differentiated spectrum
             Results_original returns the indexes of peaks and positions for the original spectrum
    """
    cutoff_diff = 0.05

    # Change in sign for array signifies a zero crossing
    sign_change = np.asarray(np.sign(diff_spectrum[:-1]) != np.sign(diff_spectrum[1:])).nonzero()[0]
    diff = np.diff(diff_spectrum)
    # Changes in sign correspond to either peaks or troughs
    maxima = np.asarray(np.sign(diff[:-1]) > np.sign(diff[1:])).nonzero()[0]
    minima = np.asarray(np.sign(diff[:-1]) < np.sign(diff[1:])).nonzero()[0]

    # Calculate (x,y) values of peaks
    h_y = diff_spectrum[maxima]
    significant_h = np.extract(h_y / np.amax(h_y) > cutoff_diff, maxima)

    # Calculate (x,y) values of troughs
    l_y = diff_spectrum[minima]
    significant_l = np.extract(l_y / np.amin(l_y) > cutoff_diff, minima)

    # Summarises results into a dictionary
    results_diff = {"zeros": sign_change,
                    "highs": significant_h,
                    "lows": significant_l}

    peaks = []
    peak_widths = []
    smooth = sf.convo_filter_n(raw_spectrum, 5, 10)
    # Iterating through all peaks (peaks must come first before troughs)
    for high in maxima:
        if np.searchsorted(minima, high) == minima.size:
            break
        low = minima[np.searchsorted(minima, high)]

        # Search for position of the peaks and troughs and see if they are in succession
        h_position = np.searchsorted(sign_change, high)
        l_position = np.searchsorted(sign_change, low)

        if h_position + 1 == l_position:
            if l_position + 1 < sign_change.size:
                peaks.append(sign_change[h_position]+1)
                peak_widths.append([sign_change[h_position - 1], sign_change[l_position + 1]])

    # Checks for prominence of peaks and ensures that they are above the mean
    peaks = np.array(peaks)
    peak_widths = np.array(peak_widths)

    prominence = sig.peak_prominences(smooth, peaks)[0]
    shift = np.nonzero(prominence == 0)[0]
    for s in shift:
        for i in [1, -1, 2, -2, 3, -3]:
            index = peaks[s]+i
            if index < 0 or index >= smooth.size:
                continue
            p = sig.peak_prominences(smooth, np.array([index]))[0]
            if p > 0:
                peaks[s] += i
                break
    prominence = sig.peak_prominences(smooth, peaks)[0]

    # Peaks are compared to the noise level of the spectrum
    if noise_stdd == -1 and noise_mean == -1:
        noise_only = sf.get_noise(raw_spectrum)
        noise_mean = np.mean(noise_only)
        noise_stdd = np.std(noise_only)
    z = (prominence - noise_mean)/noise_stdd

    peaks = peaks[np.nonzero(z > 2)]
    prominence = sig.peak_prominences(smooth, peaks)[0]

    results_original = {"peaks": peaks,
                        "prom": prominence,
                        "widths": peak_widths}
    return results_diff, results_original


def normalise(y):
    return y / np.max(y)
