import numpy as np
import peak_detection as pd
import smoothing_functions as sf
import generate_spectrum as gs


def fit_peaks(spectrum, wavenumbers):
    diff_spectrum = pd.corrected_diff_spectrum(spectrum)
    noise_spectrum = sf.get_noise(spectrum)
    noise_mean, noise_stdd = np.mean(noise_spectrum), np.std(noise_spectrum)
    peak_results = pd.detect_peaks(spectrum, diff_spectrum)

    combined_spectrum = np.zeros_like(wavenumbers)
    peak_positions = peak_results["peaks"]
    peak_prom = peak_results["prom"]
    peak_widths = peak_results["widths"]
    width = []
    for w in peak_widths:
        width.append(w[1] - w[0])
    width = np.array([width])

    first_pass = np.hstack((np.array([peak_positions]).T, np.array([peak_prom]).T, width.T))
    sigma = 2 * np.ones_like(first_pass)
    peak_list = []

    for peak in peak_positions:
        pass


def optimize_peaks():
    pass


def extend_peak():
    pass
