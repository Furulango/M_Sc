import funciones as f

num_senales = 4
fs = 1000
frecuencias = [10, 15, 50, 100]
amplitudes = [1, 1.5, 0.75, 1.2]
duraciones = [2, 2.5, 3.5, 2]
tipo = 'seno'

t, senal = f.generar_senal_no_estacionaria_v2(duraciones, fs, num_senales, frecuencias, amplitudes, tipo)
f.imprimirsenal(t, senal)
