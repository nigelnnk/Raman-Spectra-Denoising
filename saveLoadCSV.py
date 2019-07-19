import numpy as np
import matplotlib.pyplot as plt
import csv
import re
import os


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


def read_txt(filename, a=0.0, b=1.0, c=0.0):
    wavenumber = []
    value = []
    pattern = re.compile("\W+")
    with open(filename, "r") as f:
        for line in f:
            l = re.split(pattern, line)
            w, v = float(l[0]), float(l[1])
            w = a*w**2 + b * w + c
            wavenumber.append(w)
            value.append(v)
    return np.array(wavenumber), np.array(value)


def ethyl_acetate():
    wa, va = read_spectrum("data/4.csv")
    ax = plt.figure().gca()

    plt.axvline(x=1738, c='C0', label="Ethyl Acetate\n381, 637, 849,\n1117, 1450, 1738")
    plt.axvline(x=1117, c='C0')
    plt.axvline(x=849, c='C0')
    plt.axvline(x=637, c='C0')
    plt.axvline(x=381, c='C0')

    plt.axvline(x=1450, c='C1', label="Methyl Salicylate\n810, 1033,\n1250, 1450")
    plt.axvline(x=1250, c='C1')
    plt.axvline(x=1033, c='C1')
    plt.axvline(x=810, c='C1')

    plt.plot(wa, va, c='black', linewidth=1.0)
    ax.set_xticks(np.arange(500, 3100, 500))
    ax.set_xticks(np.arange(200, 3100, 100), minor=True)
    plt.xlabel("Wavenumbers / cm^-1")
    plt.ylabel("Intensity")
    plt.title("Ethyl Acetate with Methyl Salicylate")
    plt.legend()
    plt.grid(which="both")
    plt.show()


def load_nuclear(filename):
    r = np.load(filename)
    x = r["wavenumbers"]
    signal = np.trim_zeros(r["values"], trim='f')
    x = x[x.size - signal.size:]
    # x = x[x.size - signal.size+5:]
    # signal = signal[5:]
    # signal[signal < 0.01] = 0.01
    noise = signal
    return x, noise, signal


def load_raman(filename):
    wavenumbers, signal = read_spectrum(filename)
    wavenumbers = np.flip(wavenumbers)
    x = wavenumbers
    signal = np.flip(signal)
    _, noise = read_spectrum(filename)
    noise = np.flip(noise)
    return x, noise, signal


def threebythree():
    i = 99999
    for file in os.listdir("data/"):
        if file.endswith(".txt"):
            print(file)
            wa, va = read_txt("data/" + file)
            print(va)
            np.savez("data/" + str(i) + ".npz", wavenumbers=wa, values=va)
            i += 1


def baseline():
    import scipy.signal as sig
    import matplotlib.pyplot as plt
    s = sig.windows.hann(50)
    x = np.linspace(0, 50)
    b = x/50
    fig, ax = plt.subplots(ncols=2)

    ax[0].plot(x, s)
    ax[1].plot(x, s+b, label="Modified Signal")
    ax[1].plot(x, b, alpha=0.5, label="Baseline")
    ax[1].legend()

    plt.show()


def main():
    # , a=3.3817, b=0.3692, c=0.00004
    # w, v = read_txt("data/Original/RadSeeker_CS_75307_D20141013_T165449_E0354_N42-0001.txt")
    # np.savez("data/354.npz", wavenumbers=w, values=v)
    x, noise, s = load_raman("data/4.csv")
    fig, ax = plt.subplots()
    ax.plot(x, s, c='k', linewidth=1)
    ax.set_xticks(np.arange(round(x[0], -2), x[-1] + 1, 100), minor=True)
    ax.set_xlabel("Wavenumbers / cm^-1")
    ax.set_ylabel("Intensity")
    ax.set_title("Ethyl Acetate with Methyl Salicylate")
    ax.grid(which="both")
    plt.show()


if __name__ == "__main__":
    ethyl_acetate()
