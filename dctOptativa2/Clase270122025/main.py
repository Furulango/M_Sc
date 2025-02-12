
import funciones as f

amplitud = 1
frecuencia = 50
duracion = 10
frecuencia_muestreo = 5000

num_disturbios = 3
amplitud_disturbio = [0.5, 0.3, 0.7]
duracion_disturbio = [[2, 3], [4, 4.5], [8, 8.1]]
frecuencias_disturbio = [100, 200, 50]

t1, senal1 = f.generar_senal_coseno(amplitud, frecuencia, duracion,frecuencia_muestreo)

t, senal = f.generar_disturbios_v2(t1,senal1, frecuencia_muestreo,
            num_disturbios, amplitud_disturbio, duracion_disturbio, frecuencias_disturbio, tipo='coseno')

f.imprimirsenal(t, senal)    
