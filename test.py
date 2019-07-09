import numpy as np
import matplotlib.pyplot as plt
from numba import *
from scipy import signal as sig
import saveLoadCSV as sl
import smoothing_functions as sf
import peak_detection as pd
import generate_spectrum as gs


def graph():
    x = np.load("data/spectrum_clean.npz")["wavenumbers"]
    x = np.flip(x)
    signal, correct_peaks = gs.generate_random(x)
    # signal = gs.populate(x, gs.LSIGNAL)
    rand = np.random.randn(x.size) * np.amax(signal) / 20
    noise = rand + signal

    # wavenumbers, signal = sl.read_spectrum("data/4.csv")
    # wavenumbers = np.flip(wavenumbers)
    # x = wavenumbers
    # signal = np.flip(signal)
    # _, noise = sl.read_spectrum("data/23.csv")
    # noise = np.flip(noise)

    # r = np.load("data/48.npz")
    # x = r["wavenumbers"]
    # signal = r["values"]
    # signal[signal < 0.001] = 0.001
    # noise = signal

    ds, cs = pd.corrected_diff_spectrum(noise, 5, 23)
    result_diff, result_original = pd.detect_peaks(noise, cs, x[:-1])

    fig, ax = plt.subplots(nrows=2, ncols=2)
    # ax.plot(x, sf.convo_filter_n(noise, 5, 20)/4, color='k', alpha=0.3)
    ax[0, 0].plot(x, signal, color='C1')
    ax[0, 0].set_title("Signal Spectrum")

    ax[1, 0].plot(x[:-1], cs, color='C1')
    ax[1, 0].scatter(result_diff["zeros_x"], result_diff["zeros_y"], color='b', marker="o", label="Zeros", zorder=5)
    ax[1, 0].scatter(result_diff["highs_x"], result_diff["highs_y"], color='r', marker="x", label="Highs", zorder=5)
    ax[1, 0].scatter(result_diff["lows_x"], result_diff["lows_y"], color='g', marker="x", label="Lows", zorder=5)
    ax[1, 0].set_title("Differentiated Spectrum (Corrected)")
    ax[1, 0].legend()

    g = ax[0, 1].get_gridspec()
    ax[0, 1].remove()
    ax[1, 1].remove()
    big = fig.add_subplot(g[0:, -1])

    sm = sf.convo_filter_n(noise, 5, 20)
    big.semilogy(x, noise, color='k', alpha=0.5, label="Noisy spectrum")
    big.semilogy(x, sm, color='C1', linewidth=2, label="Smoothed spectrum")
    big.scatter(result_original["peaks"], noise[result_original["peaks"]], color='m', marker="s", label="Peaks", zorder=10)

    # TODO Look at InterSpec, Sandia Nat Lab and see how to improve from there
    # peaks = sig.find_peaks_cwt(sm, np.arange(3, 40), min_snr=1.5)
    # peaks, _ = sig.find_peaks(sm, distance=5, prominence=(0.5, None))
    # big.scatter(x[peaks], sm[peaks], color='m', marker="s", label="Peaks", zorder=10)

    big.set_title("Original (Noisy) Spectrum")
    big.set_yscale("log")
    big.legend()

    plt.show()


def generate_spectra():
    x = np.load("data/spectrum_clean.npz")["wavenumbers"]
    x = np.flip(x)

    fig, ax = plt.subplots(nrows=3, ncols=3)
    for i in range(3):
        for j in range(3):
            ax[i, j].plot(x, gs.generate_random(x))
    plt.show()


def test_case_1():
    wavenumbers, signal = sl.read_spectrum("data/4.csv")
    wavenumbers = np.flip(wavenumbers)
    x = wavenumbers
    signal = np.flip(signal)
    _, noise = sl.read_spectrum("data/23.csv")
    noise = np.flip(noise)

    # signal, correct_peaks = gs.generate_random(x)
    # rand = np.random.randn(x.size) * np.amax(signal) / 20
    # noise = signal + rand

    # r = np.load("data/48.npz")
    # x = r["wavenumbers"] * 0.3692
    # signal = r["values"]
    # signal[signal < 0.01] = 0.01
    # noise = signal

    ds, cs = pd.corrected_diff_spectrum(noise, 5, 23)
    smooth = sf.convo_filter_n(noise, 5, 10)
    result_diff, result_original = pd.detect_peaks(noise, cs)

    fig, ax = plt.subplots()
    ax.plot(x, smooth)
    ax.plot(x, noise, alpha=0.5, label="Noise")

    peaks = result_original["peaks"]
    prom = result_original["prom"]
    print(x[peaks])
    print(prom)

    ax.scatter(x[peaks], smooth[peaks], color='m', marker="s", label="Peaks", zorder=5)
    ax.vlines(x=x[peaks], ymin=smooth[peaks]-prom, ymax=smooth[peaks], color='k', zorder=10, label="Prominence")
    ax.set_xticks(np.arange(round(x[0], -2), x[-1]+1, 100), minor=True)
    ax.grid(which="both")
    plt.legend()
    plt.show()


def test_case_2():
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


def test_case_3():
    """
    This is a test case for Baseline correction by improved iterative polynomial fitting with
    automatic threshold by Gan et. al. (2006).

    It showcases the limitations of a polynomial fitting of the baseline, and how sometimes
    certain powers will result in weird results. This makes use of poly_corrected_baseline(),
    which is not used in future versions of the code.
    """
    wavenumbers, signal = sl.read_spectrum("data/4.csv")
    wavenumbers = np.flip(wavenumbers)
    x = wavenumbers
    signal = np.flip(signal)
    _, noise = sl.read_spectrum("data/23.csv")
    noise = np.flip(noise)

    new_spec, b = pd.poly_baseline_corrected(noise, wavenumbers, 9)
    new_noise = noise - b
    ds, cs = pd.corrected_diff_spectrum(new_noise, 5, 23)
    smooth = sf.convo_filter_n(new_noise, 5, 10)
    result_diff, result_original = pd.detect_peaks(new_noise, cs)

    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(wavenumbers, noise)
    ax[0].plot(wavenumbers, b, alpha=0.5)
    # ax[0].plot(wavenumbers, sf.convo_filter_n(noise, 101, 30))
    ax[0].plot(wavenumbers, np.zeros_like(wavenumbers), alpha=0.5, color='k')

    ax[1].plot(x, smooth)
    ax[1].plot(x, new_noise, alpha=0.5, label="Noise")

    peaks = result_original["peaks"]
    prom = result_original["prom"]
    print(x[peaks])
    print(prom)

    ax[1].scatter(x[peaks], smooth[peaks], color='m', marker="s", label="Peaks", zorder=5)
    ax[1].vlines(x=x[peaks], ymin=smooth[peaks] - prom, ymax=smooth[peaks], color='k', zorder=10, label="Prominence")
    ax[1].set_xticks(np.arange(round(x[0], -2), x[-1] + 1, 100), minor=True)
    ax[1].grid(which="both")
    plt.legend()
    plt.show()


def test_case_4():
    """

    """
    wavenumbers, signal = sl.read_spectrum("data/4.csv")
    wavenumbers = np.flip(wavenumbers)
    x = wavenumbers
    signal = np.flip(signal)
    _, noise = sl.read_spectrum("data/23.csv")
    noise = np.flip(noise)

    b, weights = pd.als_baseline(sf.convo_filter_n(noise), 1000000)
    # TODO how to automate finding of lambda?
    new_noise = noise - b
    ds, cs = pd.corrected_diff_spectrum(noise, 5, 23)
    smooth = sf.convo_filter_n(new_noise)
    result_diff, result_original = pd.detect_peaks(noise, cs)

    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(wavenumbers, noise)
    ax[0].plot(wavenumbers, b)
    ax[0].scatter(wavenumbers, weights, s=0.2, alpha=0.7, color='r', zorder=10)
    ax[0].plot(wavenumbers, np.zeros_like(wavenumbers), alpha=0.5, color='k')

    ax[1].plot(x, smooth)
    ax[1].plot(x, new_noise, alpha=0.5, label="Noise")

    peaks = result_original["peaks"]
    prom = sig.peak_prominences(smooth, peaks)[0]
    print(x[peaks])
    print(prom)

    ax[1].scatter(x[peaks], smooth[peaks], color='m', marker="s", label="Peaks", zorder=5)
    ax[1].vlines(x=x[peaks], ymin=smooth[peaks] - prom, ymax=smooth[peaks], color='k', zorder=10, label="Prominence")
    ax[1].set_xticks(np.arange(round(x[0], -2), x[-1] + 1, 100), minor=True)
    ax[1].grid(which="both")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_case_4()
