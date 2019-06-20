import numpy as np
import matplotlib.pyplot as plt
from numba import *
from scipy import signal as sig
import saveLoadCSV as sl
import smoothing_functions as sf
import peak_detection as pd
import generate_spectrum as gs


def main():
    # x = np.load("data/spectrum_clean.npz")["wavenumbers"]
    # x = np.flip(x)
    # signal = gs.populate(x, gs.LSIGNAL)
    wavenumbers, signal = sl.read_spectrum("data/4.csv")
    wavenumbers = np.flip(wavenumbers)
    signal = np.flip(signal)
    _, noise = sl.read_spectrum("data/23.csv")
    noise = np.flip(noise)
    x = wavenumbers

    # np.random.seed(3141592653)
    # rand = np.random.randn(x.size) * np.amax(signal) / 20
    # noise = rand + signal

    ds, cs = pd.get_corrected_spectrum(noise, 5, 23)
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

    sm = sf.convo_filter_n(noise, 5, 3)
    big.plot(x, noise, color='k', alpha=0.5, label="Noisy spectrum")
    big.plot(x, sm, color='C1', linewidth=2, label="Smoothed spectrum")
    big.scatter(result_original["peaks_x"], result_original["peaks_y"], color='m', marker="s", label="Peaks", zorder=10)
    big.set_title("Original (Noisy) Spectrum")
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


if __name__ == "__main__":
    main()
