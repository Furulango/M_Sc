import funciones as fn
import matplotlib.pyplot as plt

# Generar señal con 3 perturbaciones
# Señal principal:
# | Frecuencia (Hz) | Amplitud | Duración (s) |
# |-----------------|----------|--------------|
# | 60              | 1        | 10           |
#
# Perturbaciones:
# | Perturbación | Inicio (s) | Fin (s) | Frecuencia (Hz) | Amplitud |
# |--------------|------------|---------|-----------------|----------|
# | 1            | 2          | 3       | 100             | 1        |
# | 2            | 4          | 4.5     | 150             | 0.75     |
# | 3            | 8          | 8.1     | 200             | 1.5      |
# Se crearan 3 senales con ruido Gaussiano una con ruido de 1 db, otra con 10 y otra con 20
# Se mostraran las 4 senales en un solo grafico 

amplitud = 1
frecuencia = 60
duracion = 10
frecuencia_muestreo = 3000
num_disturbios = 3
amplitud_disturbio = [1, 0.75, 1.5]
duracion_disturbio = [[2, 3], [4, 4.5], [8, 8.1]]
frecuencias_disturbio = [100, 150, 200]

t1, senal = fn.generar_senal_seno(amplitud, frecuencia, duracion, frecuencia_muestreo)
t, senal1 = fn.generar_disturbios_v2(t1, senal, frecuencia_muestreo, num_disturbios, amplitud_disturbio, duracion_disturbio, frecuencias_disturbio, tipo='coseno')

senal2 = fn.agregar_ruido_awgn(senal1, 1)
senal3 = fn.agregar_ruido_awgn(senal1, 10)
senal4 = fn.agregar_ruido_awgn(senal1, 20)

conjunto_senales = [senal1, senal2, senal3, senal4]
etiquetas_senales = ['Señal original', 'Señal con ruido 1 dB', 'Señal con ruido 10 dB', 'Señal con ruido 20 dB']
fn.imprimir_multiples_senales_unica_ventana(t, conjunto_senales, etiquetas_senales)

fn.transformadaFourier(t,senal4)
