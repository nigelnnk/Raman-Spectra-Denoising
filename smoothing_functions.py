import numpy as np
from numba import *


@njit
def SGfilter(y):
    """
    Standard Savitzky-Golay filter based on coefficients. Window size is fixed at 5.
    :param y: Array of spectrum values to be smoothed
    :return: Array of smoothed spectrum
    """
    ans = np.zeros(y.size)
    ans[0] = (-3*y[2] + 12*y[1] + 17*y[0] + 12*y[1] - 3*y[2])/35
    ans[1] = (-3*y[1] + 12*y[0] + 17*y[1] + 12*y[2] - 3*y[3])/35
    ans[y.size-2] = (-3*y[y.size-4] + 12*y[y.size-3] + 17*y[y.size-2] + 12*y[y.size-1] - 3*y[y.size-2])/35
    ans[y.size-1] = (-3*y[y.size-3] + 12*y[y.size-2] + 17*y[y.size-1] + 12*y[y.size-2] - 3*y[y.size-3])/35
    for i in range(2, y.size - 2):
        ans[i] = (-3*y[i-2] + 12*y[i-1] + 17*y[i] + 12*y[i+1] - 3*y[i+2])/35
    return ans


@njit
def SGfiltern(y, n):
    """
    Applies the coefficients-based Savitzky-Golay filter n times
    :param y: Array of spectrum values to be smoothed
    :param n: Number of times smoothing function called
    :return: Array of smoothed spectrum after n smooths
    """
    ans = y.copy()
    for i in range(n):
        ans = SGfilter(ans)
    return ans


# @njit
def convo_filter_n(spectrum, window_size, n):
    """
    Applies the convolution-based low-pass filter n times based on a window size.
    Filter is based on the Hanning window. Boundary effects are still present, but
    effects are reduced by extending te edges of the Raman spectrum.
    :param spectrum: Array of spectrum values to be smoothed
    :param window_size: Number of datapoints to include in window for smoothing
    :param n: Number of times smoothing function is called
    :return: Array of smoothed spectrum after n smooths
    """
    ans = spectrum.copy()
    w = np.hanning(window_size)
    w /= np.sum(w)
    for i in range(n):
        temp = np.pad(ans, int((window_size-1)/2), "edge")  # reduces boundary effects
        ans = np.convolve(temp, w, mode="valid")
    return ans


@njit
def iter_convo_filter(spectrum, window_size):
    """
    Smooths spectrum as detailed in [1] Fully Automated High-Performance Signal-to-Noise
    Ratio Enhancement Based on an Iterative Three-Point Zero-Order Savitzky–Golay Filter
    by Schulze et. al. (2008).

    This algorithm iteratively applies the convolution filter until it reaches a chi
    squared value, which is the number of wavenumbers in the spectrum. The filter has been
    changed to the convolution filter as the SG filter takes too long (not optimised).
    :param spectrum: Spectrum to be smoothed
    :param window_size: Smoothing window size
    :return: Smoothed spectrum
    """
    ans = spectrum.copy()
    n = spectrum.size
    w = np.hanning(window_size)
    w /= np.sum(w)
    i = 0
    while chi_sq(spectrum, ans) < n:
        temp = np.pad(ans, int((window_size - 1) / 2), "edge")  # reduces boundary effects
        ans = np.convolve(temp, w, mode="valid")
        i += 1
    print(chi_sq(spectrum, ans))
    print(i)
    return ans


@njit
def chi_sq(original, modified):
    """
    This is a modified chi squared parameter, as the sum is divided by the noise level.
    This first calculates what the noise level is, then proceeds with the algorithm.
    :param original: Spectrum to compare against, "noisy" spectra
    :param modified: Spectrum being compared, "processed" spectra
    :return: Chi Squared value
    """
    n = get_noise(original)
    stddev = np.std(n) ** 2
    return np.sum(np.square(original-modified) / stddev)


@njit
def rms_error(original, modified):
    """
    A general error function widely used in signal processing
    :param original: Spectrum to compare against, "correct" spectra
    :param modified: Spectrum being compared, "processed" spectra
    :return: RMS error value
    """
    return np.sqrt(np.mean(np.square(original-modified)))


@njit
def get_noise(spectrum):
    """
    To determine noise in the spectrum, peaks must be removed. This can be done by taking
    the difference between successive peaks, then removing any values that lie more than
    99.7% (i.e. 3 sigma) away from the normal distribution. As peaks are usually sharp,
    and have high signal to noise ratio, they lie far from the normal distribution and
    will be removed by calculating their z-score.

    Note that the standard deviation of the returned noise is generally higher than the
    true standard deviation of the noise because the base of peaks are still present in
    the spectrum and cannot be removed easily.
    :param spectrum: Spectrum with its noise to be assessed
    :return: Spectra containing only noise. Will have a shorter length then original
    """
    spectrum = np.diff(spectrum)
    mean = np.mean(spectrum)
    stddev = np.std(spectrum)
    z_score = np.abs(spectrum-mean)/stddev
    new = np.extract(z_score < 3, spectrum)
    return new


@njit
def hann(y):
    return np.hanning(y)

def main():
    print(hann(5))

if __name__ == "__main__":
    main()