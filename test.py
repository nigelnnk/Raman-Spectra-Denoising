import numpy as np
import matplotlib.pyplot as plt
from numba import *
from scipy import signal as sig
import saveLoadCSV as sl
import smoothing_functions as sf
import peak_fitting as pf
import generate_spectrum as gs


def main():
    x = np.load("data/spectrum_clean.npz")["wavenumbers"]
    signal = gs.populate(x, gs.LSIGNAL)
    # wavenumbers, signal = sl.read_spectrum("data/4.csv")
    # _, noise = sl.read_spectrum("data/23.csv")
    # x = wavenumbers

    # np.random.seed(3141592653)
    rand = np.random.randn(x.size) * np.amax(signal) / 20
    # print("rrrrr\t{}\t{}".format(np.mean(rand), np.std(rand)))
    noise = rand + signal

    ds, cs = pf.get_corrected_spectrum(noise, 5, 53)
    new_x, new_y, h_x, h_y, l_x, l_y = pf.detect_peaks(cs, x[:-1])

    fig, ax = plt.subplots()
    # ax.plot(x, sf.convo_filter_n(noise, 5, 20)/4, color='k', alpha=0.3)
    ax.plot(x[:-1], cs, color='C1')
    ax.plot(new_x, new_y, color='b', marker="o")
    ax.scatter(h_x, h_y, color='r', marker="x")
    ax.scatter(l_x, l_y, color='g', marker="x")
    plt.show()


if __name__ == "__main__":
    main()
