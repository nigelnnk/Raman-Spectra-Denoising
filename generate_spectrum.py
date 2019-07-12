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


def generate_random(wavenumbers, seed=-1):
    if seed != -1:
        np.random.seed(seed)
    wave = wavenumbers.copy()
    n = random.randint(2, 7)
    spectrum = np.zeros(wave.size)
    space = np.mean(np.diff(wave))
    peaks = []
    for i in range(n):
        p = random.randint(10, wave.size-10)
        w = random.randint(5, 30)
        h = random.randint(1, 5)
        # print("{}\t{}\t{}".format(p, w, h))
        peaks.append((wave[p], w, h))
        spectrum += lorentzian(wavenumbers, wave[p], w, h)
        remove = wave[np.arange(p-3, p+3)]
        wave = np.setdiff1d(wave, remove)
    peaks.sort(key=lambda x: x[0])
    return spectrum, peaks

