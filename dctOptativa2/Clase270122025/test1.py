import funciones as f

amplitud = 1
frecuencia = 10
duracion = 1
amplitud2 = 2
frecuencia2 = 20
duracion2 = 1
frecuencia_muestreo = 40

t, senal1 = f.generar_senal_coseno(amplitud, frecuencia, duracion, frecuencia_muestreo)
t, senal2 = f.generar_senal_seno(amplitud2, frecuencia2, duracion2, frecuencia_muestreo)
senal = f.sumar_senales(senal1, senal2)

f.imprimirsenal(t, senal)
