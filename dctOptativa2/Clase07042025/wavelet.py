import numpy as np
import matplotlib.pyplot as plt
import pywt

#%% Señal
fs = 1250
T = 2
t = np.arange(0, T, 1/fs)
f1 = 80
f2 = 40
f3 = 5
s1 = np.cos(2*np.pi*f1*t)
s2 = np.cos(2*np.pi*f2*t)
s3 = np.cos(2*np.pi*f3*t)
signal = s1 + s2 + s3

#%% Wavelet
max_level = 7
aproximacion = np.zeros((max_level, 2))
nyquist = fs/2
for nivel in range(1, max_level+1):
    limite_superior_A = nyquist / (2**nivel)
    limite_inferior_A = 0
    aproximacion[nivel-1, :] = [limite_inferior_A, limite_superior_A]
    print(f'A{nivel}: [{limite_inferior_A:.2f} - {limite_superior_A:.2f}] Hz')

#%% Obtener coeficientes
wavelet = 'dmey'
coeffs = pywt.wavedec(signal, wavelet, level=max_level)
cA = coeffs[0]
cD = coeffs[1:]

#%% Senal reconstruida
DCz = []
for i in range(1, max_level+1):
    coeff_list = [np.zeros_like(cA)] + [np.zeros_like(cD[j]) for j in range(max_level)] 
    coeff_list[i] = cD[i-1]
    DCz.append(pywt.waverec(coeff_list, wavelet))

coeff_list = [cA] + [np.zeros_like(cD[j]) for j in range(max_level)]
APz = pywt.waverec(coeff_list, wavelet)

length = len(signal)
for i in range(len(DCz)):
    if len(DCz[i]) > length:
        DCz[i] = DCz[i][:length]
if len(APz) > length:
    APz = APz[:length]

#%% Visualización
plt.figure(figsize=(12, 8))
plt.subplot(max_level+2, 1, 1)
plt.plot(t, signal)
plt.title('Señal Original')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')

for i in range(max_level):
    plt.subplot(max_level+2, 1, i+2)
    plt.plot(t, DCz[i])
    plt.title(f'Detalle D{i+1}')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')

plt.subplot(max_level+2, 1, max_level+2)
plt.plot(t, APz)
plt.title(f'Aproximación A{max_level}')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.tight_layout()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t, signal, 'b')
plt.title('Señal Original')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')

plt.subplot(1, 2, 2)
reconstruida = APz.copy()
for i in range(max_level):
    reconstruida = reconstruida + DCz[i]
plt.plot(t, reconstruida, 'r')
plt.title('Señal Reconstruida')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.tight_layout()

plt.show()
