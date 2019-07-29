import numpy as np
from numba import *
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy import sparse
from scipy.sparse.linalg import splu
import smoothing_functions as sf


def corrected_diff_spectrum(spectrum, noise_window=5, baseline_window=23):
    """
    Processes the given spectrum as detailed in [2] A simple background elimination method
    for Raman spectra by Baek et. al. (2009) to remove baseline errors from spectra.
    :param spectrum: Spectrum to be processed
    :param noise_window: Number of data points, usually less than smallest peak width
    :param baseline_window: Number of data points, usually larger than widest peak width
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
    :return: baseline of spectrum
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
    return b


def als_baseline(spectrum, lmbd, p=0.05):
    """
    This baseline correction method is taken from Eiler and Boelen (2005) Baseline Correction
    with Asymmetric Least Squares Smoothing.

    It minimises the least squares fitting of the baseline and spectrum while also adjusting
    the weights of each point in the spectrum to tune out the peaks.
    :param spectrum: Spectrum to be processed to generate baseline
    :param lmbd: As defined in algorithm. Raised to the power of 10.
    :param p: Weight given to points that are determined to be peaks
    :return: baseline of spectrum and weights of each individual points
    """
    if lmbd < 15:
        lmbd = 10**lmbd
    N = spectrum.size
    diagonals = np.array([1, -2, 1])
    D = sparse.diags(diagonals, np.arange(3), (N-2, N), format='csc')
    diff_matrix = D.T.dot(D)
    weights = np.ones_like(spectrum)
    for i in range(10):
        W = sparse.diags(weights, 0, (N, N), format='csc')
        new_baseline = splu(W + lmbd * diff_matrix).solve(W.dot(spectrum))
        weights = np.where(spectrum > new_baseline, p, 1 - p)
    return new_baseline, weights


def auto_als_baseline(spectrum, p=0.05):
    """
    This is the automated form of als_baseline, to determine which is the best lambda value.
    This is original work. Calculated by finding spectrum that has most number of points
    counted as baseline. Usually lambda is determined by human intervention.

    An algorithm for gradient descent could possibly give better lambda values more quickly
    but I am unsure how to implement it.
    :param spectrum: Spectrum to be processed to generate baseline
    :param spectrum: Spectrum to be processed to generate baseline
    :param p: Weight given to points that are determined to be peaks
    :return: als_baseline with best lambda value
    """
    rms = []
    for L in range(6, 17):
        b, w = als_baseline(spectrum, L/2, p)
        rms.append(np.count_nonzero(w == p) * np.sum(np.square(np.diff(b)))**0.5)
    m = min(rms)
    L = rms.index(m)/2 + 3

    # print(f"Lambda = {L}")
    # print(rms)

    # TODO automatic calculation of best p value?
    # Perhaps use the (spectrum - baseline) distribution?? Ensures that the correct p is
    # chosen.
    return als_baseline(spectrum, L, p)


def detect_peaks(raw_spectrum, diff_spectrum, noise_mean=-1, noise_stdd=-1, t="raman"):
    """
    Peak detection as mentioned briefly in [2], but without any algorithm provided. This
    is original code, not referenced from any paper.

    By finding maxima and minima in the differentiated spectrum, one can identify where
    the peaks are, as peaks have a pattern of zero-maxima-zero-minima-zero. Hence, these
    are used to determine if a peak is present by analysing if each maxima has a succeeding
    minima. Additionally, the prominence of the peaks must be sufficiently tall, determined
    by the standard deviation of noise level in the spectrum.

    :param raw_spectrum: Noisy spectrum as given by raw data
    :param diff_spectrum: Differentiated spectrum, which can be produced by corrected_diff_spectrum()
    :param noise_mean: Noise mean used to filter peaks. Can be auto calculated.
    :param noise_stdd: Noise std dev used to filter peaks. Can be auto calculated.
    :param t: type of spectrum being processed.
            Raman spectra will filter peaks based on the noise levels of different segments
                of the spectrum. Usually, higher wavenumbers have greater noise levels.
            Gamma spectra will filter peaks based on the noise level of the entire spectra,
                as the noise level is consistent throughout the spectra.
    :return: Two dictionaries are returned,for different display purposes:
             Results_diff returns indexes of zeros, troughs and peaks of the differentiated spectrum
             Results_original returns the indexes of peaks and positions for the original spectrum
    """
    cutoff_diff = 0.05

    # ALS baseline correction is not chosen because the lambda adjustment might result in
    # negative signals, which is not ideal for peak detection.
    # diff_spectrum = np.diff(raw_spectrum - auto_als_baseline(raw_spectrum)[0])

    # Change in sign for array signifies a zero crossing
    sign_change = np.asarray(np.sign(diff_spectrum[:-1]) != np.sign(diff_spectrum[1:])).nonzero()[0]
    diff = np.diff(diff_spectrum)
    # Changes in sign correspond to either peaks or troughs
    maxima = np.asarray(np.sign(diff[:-1]) > np.sign(diff[1:])).nonzero()[0]
    minima = np.asarray(np.sign(diff[:-1]) < np.sign(diff[1:])).nonzero()[0]

    # Calculate (x,y) values of peaks
    h_y = diff_spectrum[maxima]
    significant_h = maxima[h_y / np.amax(h_y) > cutoff_diff]

    # Calculate (x,y) values of troughs
    l_y = diff_spectrum[minima]
    significant_l = minima[l_y / np.amin(l_y) > cutoff_diff]

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

    peaks = np.array(peaks)
    peak_widths = np.array(peak_widths)

    # Find best peak that has prominence (i.e. its neighbours on left and right are lower than itself)
    original_peaks = peaks.copy()
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
        if t == "raman":
            bin_len, noise_mean, noise_stdd = sf.get_moving_noise(raw_spectrum)
            ct = np.cumsum(bin_len)
            ind = np.searchsorted(ct, peaks)
            z = (prominence - noise_mean[ind])/noise_stdd[ind]
        elif t == "gamma":
            noise_only = sf.get_noise(raw_spectrum)
            noise_mean = np.mean(noise_only)
            noise_stdd = np.std(noise_only)
            z = (prominence - noise_mean) / noise_stdd
    else:
        z = (prominence - noise_mean)/noise_stdd

    # Filter peaks such that peaks must be >99.7% of background noise
    peaks = peaks[np.nonzero(z > 3)]
    prominence = sig.peak_prominences(smooth, peaks)[0]

    # optional filter based on prominence and actual signal level
    # prom_filter = np.nonzero(prominence / raw_spectrum[peaks] > 0.5)
    # peaks = peaks[prom_filter]
    # prominence = prominence[prom_filter]
    # peak_widths = peak_widths[prom_filter]

    # Calculate peak widths based on the differentiated spectra
    final_width = []
    p, q = 0, 0
    while p < original_peaks.size and q < peaks.size:
        if np.abs(original_peaks[p] - peaks[q]) <= 3:
            final_width.append((peak_widths[p][1] - peak_widths[p][0])/3)
            p += 1
            q += 1
        elif original_peaks[p] < peaks[q]:
            p += 1
        elif original_peaks[p] > peaks[q]:
            q += 1

    results_original = {"peaks": peaks,
                        "prom": prominence,
                        "widths": final_width}
    return results_diff, results_original


def normalise(y):
    return y / np.max(y)
