import numpy as np
import matplotlib.pyplot as plt
import csv


def read_spectrum(filename):
    wavenumber = []
    value = []
    with open(filename, "r") as f:
        data = csv.reader(f, delimiter=',')
        for w, v in data:
            wavenumber.append(float(w))
            value.append(float(v))
    wa = np.array(wavenumber)
    va = np.array(value)
    return wa, va


def main():
    wa, va = read_spectrum("data/4.csv")
    ax = plt.figure().gca()

    plt.axvline(x=1738, c='r', alpha=0.7, label="Ethyl Acetate\n381, 637, 849,\n1117, 1450, 1738")
    plt.axvline(x=1450, c='r', alpha=0.7)
    plt.axvline(x=1117, c='r', alpha=0.7)
    plt.axvline(x=849, c='r', alpha=0.7)
    plt.axvline(x=637, c='r', alpha=0.7)
    plt.axvline(x=381, c='r', alpha=0.7)

    plt.plot(wa, va, c='black', linewidth=1.0)
    ax.set_xticks(np.arange(200, 3100, 100))
    plt.xlabel("Wavenumbers / cm^-1")
    # ax.invert_xaxis()
    plt.legend()
    plt.grid()
    plt.show()
    # np.savez("data/spectrum_noisy.npz", wavenumbers=wa, values=va)


if __name__ == "__main__":
    main()
