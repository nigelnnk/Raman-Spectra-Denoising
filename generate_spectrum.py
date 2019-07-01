import math
import random
import numpy as np
import matplotlib.pyplot as plt
from numba import *
from scipy import signal as sig
import saveLoadCSV as sl
import smoothing_functions as sf
import peak_detection as pd

LSIGNAL = np.array([[500, 40, 0.5], [825, 10, 1], [900, 25, 3], [1200, 10, 5]])

# The three functions below describe different types of curves found in Raman spectra
@vectorize(nopython=True)
def lorentzian(x, position, width, height):
    y = height * width**2/((x-position)**2 + width**2)
    return y


@vectorize(nopython=True)
def gaussian(x, position, width, height):
    y = height * np.e ** (-math.log(2) * (x-position)**2 / width**2)
    return y


@vectorize(nopython=True)
def pseudoVoigt(x, position, width, height, alpha):
    y = alpha * lorentzian(x, position, width, height) + (1 - alpha) * gaussian(x, position, width, height)
    return y


@njit
def populate(x, list_of_signals):
    i = np.zeros(x.size)
    for a in range(list_of_signals.shape[0]):
        i += lorentzian(x, list_of_signals[a, 0], list_of_signals[a, 1], list_of_signals[a, 2])
    return i


def generate_random(wavenumbers):
    n = random.randint(2, 7)
    spectrum = np.zeros(wavenumbers.size)
    peaks = []
    for i in range(n):
        p = random.choice(wavenumbers)
        w = random.randint(5, 30)
        h = random.randint(1, 5)
        # print("{}\t{}\t{}".format(p, w, h))
        peaks.append((p, w, h))
        spectrum += lorentzian(wavenumbers, p, w, h)
    return spectrum, peaks


def main():
    """
    For debugging purposes, to print out certain functions and see how good they are
    :return: A plot of the signal, noisy spectrum, and some functions
    """
    x = np.load("data/spectrum_clean.npz")["wavenumbers"]
    x = np.flip(x)
    signal = populate(x, LSIGNAL)
    # wavenumbers, signal = sl.read_spectrum("data/4.csv")
    # _, noise = sl.read_spectrum("data/23.csv")
    # x = wavenumbers

    np.random.seed(3141592653)
    rand = np.random.randn(x.size) * np.amax(signal) / 20
    # print("rrrrr\t{}\t{}".format(np.mean(rand), np.std(rand)))
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


if __name__ == "__main__":
    main()
