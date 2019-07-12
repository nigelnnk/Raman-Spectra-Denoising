import numpy as np
import matplotlib.pyplot as plt
from numba import *
from scipy import signal as sig
import saveLoadCSV as sl
import smoothing_functions as sf
import peak_detection as pd
import generate_spectrum as gs


def test_case_smoothing():
    """
    For debugging purposes, to print out certain functions and see how good they are
    :return: A plot of the signal, noisy spectrum, and some smoothing functions
    """
    wavenumbers = sl.read_spectrum("data/4.csv")[0]
    wavenumbers = np.flip(wavenumbers)
    x = wavenumbers
    signal = gs.populate(x, gs.LSIGNAL)
    np.random.seed(3141592653)
    rand = np.random.randn(x.size) * np.amax(signal) / 20
    noise = rand + signal

    fig, ax = plt.subplots(nrows=2, ncols=3)
    ax[0, 0].plot(x, signal)
    ax[0, 0].set_title("Signal")
    ax[0, 1].plot(x, noise)
    ax[0, 1].set_title("Noisy Spectrum")
    convolved = sf.iter_convo_filter(noise, 5)
    ax[1, 0].plot(x, convolved)
    ax[1, 0].set_title("Iterative Convolution Smoothing")

    ds, cs = pd.corrected_diff_spectrum(noise, 5, 53)
    ax[0, 2].plot(x[:-1], ds, color='C1')
    ax[0, 2].set_title("Differentiated")
    ax[1, 2].plot(x[:-1], cs, color='C1')
    ax[1, 2].set_title("Corrected")

    # new_x, new_y = pd.detect_peaks(cs, x[:-1])
    # print(new_x)
    # print(new_y)
    # ax[1, 1].plot(new_x, new_y, color='b', marker="x", markersize=6)

    ax[1, 1].plot(x, sf.convo_filter_n(noise, 5, 20))
    ax[1, 1].set_title("Convolution filter (20)")

    plt.show()


def test_case_2():
    """
    Shows how the differentiated spectrum looks like and how it is used to detect peaks
    """
    seed = 314159
    np.random.seed(seed)
    wavenumbers, signal = sl.read_spectrum("data/4.csv")
    wavenumbers = np.flip(wavenumbers)
    x = wavenumbers
    signal, correct_peaks = gs.generate_random(x, seed)
    print(correct_peaks)
    rand = np.random.randn(x.size) * np.amax(signal) / 20
    noise = signal + rand

    # x, noise, signal = sl.load_raman()

    # x, noise, signal = sl.load_nuclear()

    ds, cs = pd.corrected_diff_spectrum(noise, 5, 23)
    result_diff, result_original = pd.detect_peaks(noise, cs)

    fig, ax = plt.subplots(nrows=2, ncols=2)
    # ax.plot(x, sf.convo_filter_n(noise, 5, 20)/4, color='k', alpha=0.3)
    ax[0, 0].plot(x, signal, color='C1')
    ax[0, 0].set_title("Signal Spectrum")

    ax[1, 0].plot(x[:-1], cs, color='C1')
    ax[1, 0].scatter(x[result_diff["zeros"]], cs[result_diff["zeros"]], color='b', marker="o", label="Zeros", zorder=5)
    ax[1, 0].scatter(x[result_diff["highs"]], cs[result_diff["highs"]], color='r', marker="x", label="Highs", zorder=5)
    ax[1, 0].scatter(x[result_diff["lows"]], cs[result_diff["lows"]], color='g', marker="x", label="Lows", zorder=5)
    ax[1, 0].set_title("Differentiated Spectrum (Corrected)")
    ax[1, 0].legend()

    g = ax[0, 1].get_gridspec()
    ax[0, 1].remove()
    ax[1, 1].remove()
    big = fig.add_subplot(g[0:, -1])

    sm = sf.convo_filter_n(noise, 5, 20)
    big.plot(x, noise, color='k', alpha=0.5, label="Noisy spectrum")
    big.plot(x, sm, color='C1', linewidth=2, label="Smoothed spectrum")
    big.scatter(x[result_original["peaks"]], sm[result_original["peaks"]], color='m', marker="s", label="Peaks", zorder=10)

    # peaks = sig.find_peaks_cwt(sm, np.arange(3, 40), min_snr=1.5)
    # peaks, _ = sig.find_peaks(sm, distance=5, prominence=(0.5, None))
    # big.scatter(x[peaks], sm[peaks], color='m', marker="s", label="Peaks", zorder=10)

    big.set_title("Original (Noisy) Spectrum")
    big.legend()

    plt.show()


def generate_spectra():
    """
    Generates a 3 by 3 grid showcasing what synthetic spectra look like
    """
    x = np.load("data/spectrum_clean.npz")["wavenumbers"]
    x = np.flip(x)

    fig, ax = plt.subplots(nrows=3, ncols=3)
    for i in range(3):
        for j in range(3):
            ax[i, j].plot(x, gs.generate_random(x))
    plt.show()


def test_case_1():
    """
    This is a test case for [2] A simple background elimination method for Raman spectra.

    It showcases how it is a useful tool when the spectrum baseline changes drastically.
    However, for most Raman spectra, the baseline changes slowly. Nevertheless, this is
    still implemented because of its benefits in clearing up the differentiated spectra.
    """
    a = np.linspace(0, 5, 1000)
    b = ((a - 2.5) ** 3) + gs.lorentzian(a, 3, 0.2, 5) + np.random.normal(size=a.size) / 2
    ds, cs = pd.corrected_diff_spectrum(b, 5, 53)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].plot(a, b, c='b')
    ax[0, 0].set_title("Signal")
    ax[0, 1].plot(a, sf.convo_filter_n(b, 5, 20), c='b')
    ax[0, 1].set_title("SG smoothed")
    x = a[:-1]
    ax[1, 0].plot(x, ds, c='b', label="differentiated spectrum")
    ax[1, 0].plot(x, np.diff(((a - 2.5) ** 3)), c='r', label="differentiated baseline")
    ax[1, 0].set_title("Differentiated")
    ax[1, 0].legend()
    ax[1, 1].plot(x, cs, c='b', label="corrected spectrum")
    ax[1, 1].plot(x, np.zeros_like(x), c='r', label="baseline")
    ax[1, 1].set_title("Corrected")
    ax[1, 1].legend()
    plt.show()


def test_case_3():
    """
    This is a test case for Baseline correction by improved iterative polynomial fitting with
    automatic threshold by Gan et. al. (2006).

    It showcases the limitations of a polynomial fitting of the baseline, and how sometimes
    certain powers will result in weird results. This makes use of poly_corrected_baseline(),
    which is not used in future versions of the code.
    """
    x, noise, signal = sl.load_raman()

    b = pd.poly_baseline_corrected(noise, x, 9)
    new_noise = noise - b
    ds, cs = pd.corrected_diff_spectrum(new_noise, 5, 23)
    smooth = sf.convo_filter_n(new_noise, 5, 10)
    result_diff, result_original = pd.detect_peaks(new_noise, cs)

    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(x, noise, alpha=0.6, label="Noise")
    ax[0].plot(x, b, c='C1', label="Baseline")
    # ax[0].plot(x, sf.convo_filter_n(noise, 101, 30))
    ax[0].plot(x, np.zeros_like(x), c='k', label="Zero")
    ax[0].legend()

    peaks = result_original["peaks"]
    prom = result_original["prom"]
    print(x[peaks])
    print(prom)

    ax[1].plot(x, new_noise, alpha=0.6, label="Noise")
    ax[1].plot(x, smooth, c='k', label="Smooth")
    ax[1].scatter(x[peaks], smooth[peaks], color='m', marker="s", label="Peaks", zorder=6)
    ax[1].vlines(x=x[peaks], ymin=smooth[peaks] - prom, ymax=smooth[peaks], color='C1', zorder=5, label="Prominence")
    ax[1].set_xticks(np.arange(round(x[0], -2), x[-1] + 1, 100), minor=True)
    ax[1].grid(which="both")
    plt.legend()
    plt.show()


def test_case_4():
    """

    """
    x, noise, signal = sl.load_raman()

    # x, noise, signal = sl.load_nuclear("data/49.npz")

    # wavenumbers, signal = sl.read_spectrum("data/4.csv")
    # wavenumbers = np.flip(wavenumbers)
    # x = wavenumbers
    # signal, correct_peaks = gs.generate_random(x)
    # print(correct_peaks)
    # rand = np.random.randn(x.size) * np.amax(signal) / 20
    # noise = signal + rand

    b, weights = pd.auto_als_baseline(sf.convo_filter_n(noise), 0.05)
    new_noise = noise - b
    ds, cs = pd.corrected_diff_spectrum(noise, 5, 23)
    smooth = sf.convo_filter_n(new_noise)
    result_diff, result_original = pd.detect_peaks(noise, cs)

    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(x, noise)
    ax[0].plot(x, b)
    ax[0].scatter(x, weights, s=0.2, alpha=0.7, color='r', zorder=10)
    ax[0].plot(x, np.zeros_like(x), alpha=0.5, color='k')

    ax[1].plot(x, smooth)
    ax[1].plot(x, new_noise, alpha=0.5, label="Noise")

    peaks = result_original["peaks"]
    prom = new_noise[peaks]
    print(x[peaks])
    print(prom)

    ax[1].scatter(x[peaks], new_noise[peaks], color='m', marker="s", label="Peaks", zorder=5)
    ax[1].vlines(x=x[peaks], ymin=0, ymax=prom, color='k', zorder=10, label="Prominence")
    ax[1].set_xticks(np.arange(round(x[0], -2), x[-1] + 1, 100), minor=True)
    ax[1].grid(which="both")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_case_2()
