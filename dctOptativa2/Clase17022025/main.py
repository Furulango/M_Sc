import funciones as fn
#Generar senal y usar funcion de trasnfromada de furier

amplitud = 1 
frecuencia = 10
duracion = 10
frecuenciaMuestreo = 100

t,senal = fn.generar_senal_seno(amplitud, frecuencia, duracion, frecuenciaMuestreo)

fn.transformadaFourier(t, senal)