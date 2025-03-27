import numpy as np
import matplotlib.pyplot as plt

def generar_senal_coseno(amplitud, frecuencia, duracion, frecuencia_muestreo):
    t = np.arange(0, duracion, 1/frecuencia_muestreo)
    senal = amplitud * np.cos(2 * np.pi * frecuencia * t)
    return t, senal

def generar_senal_seno(amplitud, frecuencia, duracion, frecuencia_muestreo):
    t = np.arange(0, duracion, 1/frecuencia_muestreo)
    senal = amplitud * np.sin(2 * np.pi * frecuencia * t)
    return t, senal

def sumar_senales(senal1, senal2):
    return senal1 + senal2

def sumar_senales_diferente_tiempo(t1,t2,senal1,senal2):
    # Rellenar con ceros lo que falte para igulalar tamanos
    if len(t1) > len(t2):
        senal2 = np.concatenate((senal2, np.zeros(len(t1)-len(t2))))
        t = t1
    elif len(t2) > len(t1):
        senal1 = np.concatenate((senal1, np.zeros(len(t2)-len(t1))))
        t = t2
    return t, senal1 + senal2

def generar_senal_no_estacionaria(duracion_total, frecuencia_muestreo, num_senales, frecuencias, amplitudes, tipo='coseno'):
    duracion_por_senal = duracion_total / num_senales
    t_total = np.arange(0, duracion_total, 1/frecuencia_muestreo)
    senal_total = np.array([])

    for i in range(num_senales):
        t = np.arange(0, duracion_por_senal, 1/frecuencia_muestreo)
        if tipo == 'coseno':
            senal = amplitudes[i] * np.cos(2 * np.pi * frecuencias[i] * t)
        else:
            senal = amplitudes[i] * np.sin(2 * np.pi * frecuencias[i] * t)
        senal_total = np.concatenate((senal_total, senal))

    return t_total, senal_total

def generar_senal_no_estacionaria_v2(duraciones, frecuencia_muestreo, num_senales, frecuencias, amplitudes, tipo='coseno'):
    t_total = np.array([])
    senal_total = np.array([])

    for i in range(num_senales):
        t = np.arange(0, duraciones[i], 1/frecuencia_muestreo)
        if tipo == 'coseno':
            senal = amplitudes[i] * np.cos(2 * np.pi * frecuencias[i] * t)
        else:
            senal = amplitudes[i] * np.sin(2 * np.pi * frecuencias[i] * t)
        
        if t_total.size == 0:
            t_total = t
        else:
            t_total = np.concatenate((t_total, t + t_total[-1] + 1/frecuencia_muestreo))
        senal_total = np.concatenate((senal_total, senal))
    
    return t_total, senal_total

def generar_disturbios_v2(t, senal, frecuencia_muestreo, num_disturbios, amplitud_disturbio, duracion_disturbio, frecuencias_disturbio, tipo='coseno'):
    for i in range(num_disturbios):
        inicio = duracion_disturbio[i][0]
        fin = duracion_disturbio[i][1]
        t_disturbio = np.arange(inicio, fin, 1/frecuencia_muestreo)
        
        if tipo == 'coseno':
            disturbio = amplitud_disturbio[i] * np.cos(2 * np.pi * frecuencias_disturbio[i] * t_disturbio)
        else:
            disturbio = amplitud_disturbio[i] * np.sin(2 * np.pi * frecuencias_disturbio[i] * t_disturbio)
        
        idx_inicio = np.searchsorted(t, inicio)
        idx_fin = np.searchsorted(t, fin)
        
        senal[idx_inicio:idx_fin] += disturbio[:idx_fin-idx_inicio]
    
    return t, senal

def imprimirsenal(t, senal):
    plt.plot(t, senal)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.show()

def agregar_ruido_awgn(senal, snr_db):
    potencia_senal = np.mean(senal ** 2)
    snr_lineal = 10 ** (snr_db / 10)
    potencia_ruido = potencia_senal / snr_lineal
    ruido = np.sqrt(potencia_ruido) * np.random.normal(size=senal.shape)
    senal_con_ruido = senal + ruido
    return senal_con_ruido

def imprimir_multiples_senales_unica_ventana(t, senales, etiquetas):
    plt.figure(figsize=(10, 8))
    num_senales = len(senales)
    for i in range(num_senales):
        plt.subplot(num_senales, 1, i+1)
        plt.plot(t, senales[i])
        plt.title(etiquetas[i])
    plt.tight_layout()
    plt.show()

def transformadaFourier(tiempo, senal):
    N = len(tiempo)
    frecuencia = np.fft.fftfreq(N, tiempo[1] - tiempo[0])
    transformada = np.fft.fft(senal)
    frecuencia = frecuencia[:N // 2]
    transformada = transformada[:N // 2]
    magnitud = (2 / N) * np.abs(transformada)  
    imprimir_transformada(frecuencia, magnitud)

def imprimir_transformada(frecuencia, magnitud):
    plt.plot(frecuencia, magnitud)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud')
    plt.title('Transformada de Fourier')
    plt.grid()
    plt.show()