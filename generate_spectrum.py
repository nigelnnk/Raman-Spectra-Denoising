import math
import numpy as np
import matplotlib.pyplot as plt
from numba import *
from scipy import signal as sig
import saveLoadCSV as sl
import smoothing_functions as sf
import peak_fitting as pf

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


def main():
    """
    For debugging purposes, to print out certain functions and see how good they are
    :return: A plot of the signal, noisy spectrum, and some functions
    """
    x = np.load("data/spectrum_clean.npz")["wavenumbers"]
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
    ax[0, 2].plot(x, convolved)
    ax[0, 2].set_title("Iterative SG convolution")

    ds, cs = pf.get_corrected_spectrum(noise, 5, 53)
    ax[1, 0].plot(x[:-1], ds, color='C1')
    ax[1, 0].set_title("Differentiated")
    ax[1, 1].plot(x[:-1], cs, color='C1')
    ax[1, 1].set_title("Corrected")

    n = sf.get_noise(noise)
    ax[1, 2].plot(np.arange(n.size), n, color='C1')
    ax[1, 2].set_title("Noise")

    plt.show()


if __name__ == "__main__":
    main()
