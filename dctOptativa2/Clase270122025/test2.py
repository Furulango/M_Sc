import funciones as f
##Verificar impresion de valores

#Senal 1 
amplitud = 1
frecuencia = 10
duracion = 1

#Senal 2
amplitud2 = 2
frecuencia2 = 20
duracion2 = 1.1

frecuencia_muestreo = 40

t1, senal1 = f.generar_senal_seno(amplitud, frecuencia, duracion, frecuencia_muestreo)
t2, senal2 = f.generar_senal_seno(amplitud2, frecuencia2, duracion2, frecuencia_muestreo)

tt, senal = f.sumar_senales_diferente_tiempo(t1,t2,senal1,senal2)

f.imprimirsenal(tt, senal)

