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

    # x, noise, signal = sl.load_raman("data/NA20.csv")

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].plot(x, signal)
    ax[0, 0].set_title("True Signal")
    ax[0, 0].set_xlabel("Wavenumbers / cm^-1")
    ax[0, 0].set_ylabel("Intensity")
    ax[0, 1].plot(x, noise)
    ax[0, 1].set_title("Raw Spectrum")
    ax[0, 1].set_xlabel("Wavenumbers / cm^-1")
    ax[0, 1].set_ylabel("Intensity")
    convolved = sf.iter_convo_filter(noise, 5)
    ax[1, 0].plot(x, convolved)
    ax[1, 0].set_title("Iterative Convolution Smoothing")
    ax[1, 0].set_xlabel("Wavenumbers / cm^-1")
    ax[1, 0].set_ylabel("Intensity")
    ax[1, 1].plot(x, sf.convo_filter_n(noise, 5, 20))
    ax[1, 1].set_title("Convolution filter (20)")
    ax[1, 1].set_xlabel("Wavenumbers / cm^-1")
    ax[1, 1].set_ylabel("Intensity")

    # ds, cs = pd.corrected_diff_spectrum(noise, 5, 53)
    # ax[0, 2].plot(x[:-1], ds, color='C1')
    # ax[0, 2].set_title("Differentiated")
    # ax[1, 2].plot(x[:-1], cs, color='C1')
    # ax[1, 2].set_title("Corrected")

    # new_x, new_y = pd.detect_peaks(cs, x[:-1])
    # print(new_x)
    # print(new_y)
    # ax[1, 1].plot(new_x, new_y, color='b', marker="x", markersize=6)
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
    a = np.linspace(1000, 2000, 1000)
    b = ((a/200 - 7.5) ** 3) + gs.lorentzian(a, 1800, 10, 5) + np.random.normal(size=a.size) / 1.75
    ds, cs = pd.corrected_diff_spectrum(b, 5, 53)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].plot(a, b, c='b')
    ax[0, 0].set_title("Noisy Spectrum")
    ax[0, 0].set_xlabel("Wavenumbers / cm^-1")
    ax[0, 0].set_ylabel("Intensity")
    ax[0, 1].plot(a, sf.convo_filter_n(b, 5, 20), c='b')
    ax[0, 1].set_title("Convo smoothed")
    ax[0, 1].set_xlabel("Wavenumbers / cm^-1")
    ax[0, 1].set_ylabel("Intensity")
    x = a[:-1]
    ax[1, 0].plot(x, ds, c='b', label="Differentiated Spectrum")
    ax[1, 0].plot(x, np.diff(((a/200 - 7.5) ** 3)), c='r', label="Differentiated Baseline")
    ax[1, 0].set_title("Differentiated")
    ax[1, 0].set_xlabel("Wavenumbers / cm^-1")
    ax[1, 0].set_ylabel("Intensity")
    ax[1, 0].legend()
    ax[1, 1].plot(x, cs, c='b', label="Corrected spectrum")
    ax[1, 1].plot(x, np.zeros_like(x), c='r', label="Baseline")
    ax[1, 1].set_title("Corrected")
    ax[1, 1].set_xlabel("Wavenumbers / cm^-1")
    ax[1, 1].set_ylabel("Intensity")
    ax[1, 1].legend()

    plt.subplots_adjust(hspace=0.4)
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
    # signal, correct_peaks = gs.generate_random(x, seed)
    signal = gs.populate(x, gs.RSIGNAL)
    # print(correct_peaks)
    rand = np.random.randn(x.size) * np.amax(signal) / 20
    noise = signal + rand

    # x, noise, signal = sl.load_raman("data/23.csv")

    # x, noise, signal = sl.load_nuclear("data/50.npz")

    ds, cs = pd.corrected_diff_spectrum(noise, 5, 23)
    result_diff, result_original = pd.detect_peaks(noise, cs)

    fig, ax = plt.subplots(nrows=2, ncols=2)
    # ax.plot(x, sf.convo_filter_n(noise, 5, 20)/4, color='k', alpha=0.3)
    ax[0, 0].plot(x, signal, color='C1')
    ax[0, 0].set_title("Signal Spectrum")
    ax[0, 0].set_xlabel("Wavenumbers / cm^-1")
    ax[0, 0].set_ylabel("Intensity")

    ax[1, 0].plot(x[:-1], cs, color='C1')
    ax[1, 0].scatter(x[result_diff["zeros"]], cs[result_diff["zeros"]], color='b', marker="o", label="Zeros", zorder=5)
    ax[1, 0].scatter(x[result_diff["highs"]], cs[result_diff["highs"]], color='r', marker="x", label="Highs", zorder=5)
    ax[1, 0].scatter(x[result_diff["lows"]], cs[result_diff["lows"]], color='g', marker="x", label="Lows", zorder=5)
    ax[1, 0].set_title("Differentiated Spectrum (Corrected)")
    ax[1, 0].set_xlabel("Wavenumbers / cm^-1")
    ax[1, 0].set_ylabel("Intensity")
    ax[1, 0].legend()

    g = ax[0, 1].get_gridspec()
    ax[0, 1].remove()
    ax[1, 1].remove()
    big = fig.add_subplot(g[0:, -1])

    sm = sf.convo_filter_n(noise, 5, 20)
    big.plot(x, noise, color='k', alpha=0.5, label="Noisy spectrum")
    big.plot(x, sm, color='C1', linewidth=2, label="Smoothed spectrum")
    big.set_title("Raman Spectrum")
    big.set_xlabel("Wavenumbers / cm^-1")
    big.set_ylabel("Intensity")
    big.scatter(x[result_original["peaks"]], sm[result_original["peaks"]], color='m', marker="s", label="Peaks", zorder=10)

    # peaks = sig.find_peaks_cwt(sm, np.arange(3, 40), min_snr=1.5)
    # peaks, _ = sig.find_peaks(sm, distance=5, prominence=(0.5, None))
    # big.scatter(x[peaks], sm[peaks], color='m', marker="s", label="Peaks", zorder=10)

    big.set_title("Original (Noisy) Spectrum")
    big.legend()

    plt.show()


def test_case_3():
    """
    This is a test case for Baseline correction by improved iterative polynomial fitting with
    automatic threshold by Gan et. al. (2006).

    It showcases the limitations of a polynomial fitting of the baseline, and how sometimes
    certain powers will result in weird results. This makes use of poly_corrected_baseline(),
    which is not used in future versions of the code.
    """
    x, noise, signal = sl.load_raman("data/23.csv")

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
    np.set_printoptions(suppress=True)
    print(np.vstack((np.array([x[peaks]]), np.array([prom]))).T)

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
    # Valid csv data for Raman - 4, 23, 41, MS39, MS41, MS42, NA19, NA20, NA21
    # x, noise, signal = sl.load_raman("data/4.csv")
    # t = "raman"

    # Valid npz data for Gamma - 48, 49, 50, 51, 52, 334, 354
    x, noise, signal = sl.load_nuclear("data/48.npz")
    t = "gamma"

    # wavenumbers, signal = sl.read_spectrum("data/4.csv")
    # wavenumbers = np.flip(wavenumbers)
    # x = wavenumbers
    # signal, correct_peaks = gs.generate_random(x)
    # print(correct_peaks)
    # rand = np.random.randn(x.size) * np.amax(signal) / 20
    # noise = signal + rand
    # for i in correct_peaks:
    #     pass
    #     #check accuracy of correction

    b, weights = pd.auto_als_baseline(sf.convo_filter_n(noise), 0.05)
    new_noise = noise - b
    ds, cs = pd.corrected_diff_spectrum(noise, 5, 23)
    smooth = sf.convo_filter_n(new_noise)
    result_diff, result_original = pd.detect_peaks(noise, cs, t=t)

    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(x, noise, label="Original Spectrum")
    ax[0].plot(x, b, label="Baseline")
    # ax[0].scatter(x, weights, s=0.2, alpha=0.7, color='r', zorder=10, label="Weights")
    ax[0].plot(x, np.zeros_like(x), alpha=0.5, color='k')
    ax[0].set_title("Raw Spectrum")
    ax[0].legend()

    ax[1].plot(x, new_noise, c='C1', alpha=0.5, label="Noise")
    ax[1].plot(x, smooth, c='C0', alpha=0.7, label="Smooth")

    peaks = result_original["peaks"]
    prom = new_noise[peaks]
    np.set_printoptions(suppress=True)
    print(np.vstack((np.array([x[peaks]]), np.array([prom]))).T)
    # print(prom)

    ax[1].scatter(x[peaks], new_noise[peaks], color='m', marker="s", label="Peaks", zorder=5)
    ax[1].vlines(x=x[peaks], ymin=0, ymax=prom, color='k', zorder=10, label="Prominence")
    ax[1].set_xticks(np.arange(round(x[0], -2), x[-1] + 1, 100), minor=True)
    ax[1].grid(which="both")
    ax[1].set_title("Corrected Spectrum")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_case_2()
