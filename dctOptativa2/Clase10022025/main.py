import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.linalg import svd

def music(signal, fs, num_sources, resolution=1000):
    N = len(signal)
    M = N // 2  
    X = np.array([signal[i:M+i] for i in range(M)]).T  
    R = X @ X.T / M
    U, S, Vh = svd(R)
    Un = U[:, num_sources:]  
    freq_range = np.linspace(0, fs/2, resolution)
    spectrum = np.zeros_like(freq_range)

    for i, f in enumerate(freq_range):
        steering_vector = np.exp(-2j * np.pi * f * np.arange(M) / fs)
        spectrum[i] = 1 / np.abs(steering_vector.conj().T @ Un @ Un.conj().T @ steering_vector)

    spectrum = 10 * np.log10(spectrum / np.max(spectrum))
    peak_indices = np.argsort(spectrum)[-num_sources:]
    freqs_detected = np.sort(freq_range[peak_indices])

    return freqs_detected, spectrum, freq_range

mat_data = scipy.io.loadmat("senal_2.mat")
senal_2 = mat_data["senal_2"].flatten()
fs = 500  
num_frecuencias = 2  

frequencies, spectrum, freq_range = music(senal_2, fs, num_frecuencias)

plt.figure(figsize=(8, 5))
plt.plot(freq_range, spectrum)
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (dB)")
plt.title("Espectro MUSIC")
plt.grid()
plt.show()

print("Frecuencias detectadas:", frequencies)