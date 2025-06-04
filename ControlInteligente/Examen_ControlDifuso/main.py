# Controlador difusso para obtener el valor de la fuerza
# e(t) = 1.390 , e'(t) = -0.0982
# Se utilizara el producto en la etapa de inferencia y
#  el desfusificador para hallar el valor de la fuerza.
# 
# force |Change in error e'(t)
# 	u   |-2.0 | -1.0 | 0.0 | 1.0 | 2.0
# ---------------------------------------
#  -2.0 | 2.0 | 2.0 | 2.0 | 1.0 | 0.0
# -1.0 | 2.0 | 2.0 | 1.0 | 0.0 | -1.0
#  0.0 | 2.0 | 1.0 | 0.0 | -1.0 | -2.0
#  1.0 | 1.0 | 0.0 | -1.0 | -2.0 | -2.0
# 2.0 | 0.0 | -1.0 | -2.0 | -2.0 | -2.0

# Ai = wi(hi - hi^2/2)
# F = sum(Ai*bi)/sum(Ai)

#Valor en el centroide
#e(t)       | neglarge = -pi/2; negsmall = -pi/4; zero = 0; possmall = pi/4; poslarge = pi/2
#d/dt e(t)  | neglarge = -pi/4; negsmall = -pi/8; zero = 0; possmall = pi/8; poslarge = pi/4
#u(t),N     | neglarge = -20; negsmall = -10; zero = 0; possmall = 10; poslarge = 20

import numpy as np

def fuzzificar_error(error_rad):
    error_rad = max(-np.pi/2, min(np.pi/2, error_rad))
    F = [0.0, 0.0, 0.0, 0.0, 0.0]
    if error_rad <= -np.pi/2:
        F[0] = 1.0
    elif error_rad <= -np.pi/4:
        ratio = (error_rad - (-np.pi/2)) / (np.pi/4)
        F[0] = 1 - ratio
        F[1] = ratio
    elif error_rad <= 0:
        ratio = (error_rad - (-np.pi/4)) / (np.pi/4)
        F[1] = 1 - ratio
        F[2] = ratio
    elif error_rad <= np.pi/4:
        ratio = (error_rad - 0) / (np.pi/4)
        F[2] = 1 - ratio
        F[3] = ratio
    elif error_rad <= np.pi/2:
        ratio = (error_rad - (np.pi/4)) / (np.pi/4)
        F[3] = 1 - ratio
        F[4] = ratio
    else:
        F[4] = 1.0
    total = sum(F)
    if total > 0:
        F = [x/total for x in F]
    return F
    
def fuzzificar_cambio_error(cambio_error_rad):
    cambio_error_rad = max(-np.pi/2, min(np.pi/2, cambio_error_rad))
    F = [0.0, 0.0, 0.0, 0.0, 0.0]
    if cambio_error_rad <= -np.pi/4:
        F[0] = 1.0
    elif cambio_error_rad <= -np.pi/8:
        ratio = (cambio_error_rad - (-np.pi/4)) / (np.pi/8)
        F[0] = 1 - ratio
        F[1] = ratio
    elif cambio_error_rad <= 0:
        ratio = (cambio_error_rad - (-np.pi/8)) / (np.pi/8)
        F[1] = 1 - ratio
        F[2] = ratio
    elif cambio_error_rad <= np.pi/8:
        ratio = (cambio_error_rad - 0) / (np.pi/8)
        F[2] = 1 - ratio
        F[3] = ratio
    elif cambio_error_rad <= np.pi/4:
        ratio = (cambio_error_rad - (np.pi/8)) / (np.pi/8)
        F[3] = 1 - ratio
        F[4] = ratio
    else:
        F[4] = 1.0
    total = sum(F)
    if total > 0:
        F = [x/total for x in F]
    return F

def inferencia_fuzzy(error_rad, cambio_error_rad):
    tabla_reglas = [
        [2, 2, 2, 1, 0],
        [2, 2, 1, 0, -1],
        [2, 1, 0, -1, -2],
        [1, 0, -1, -2, -2],
        [0, -1, -2, -2, -2]
    ]
    membresia_error = fuzzificar_error(error_rad)
    membresia_cambio_error = fuzzificar_cambio_error(cambio_error_rad)
    membresia_fuerza = [0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(5):
        for j in range(5):
            activacion = membresia_error[i] * membresia_cambio_error[j]
            indice_salida = tabla_reglas[i][j]
            indice_mapeo = indice_salida + 2
            membresia_fuerza[indice_mapeo] = max(membresia_fuerza[indice_mapeo], activacion)
    return membresia_fuerza


def defuzzificador(membresia_fuerza, metodo):
    centros = [-20, -10, 0, 10, 20]
    ancho = 20

    if metodo == 'COG':
        numerador = 0
        denominador = 0
        
        for i, (centro, h) in enumerate(zip(centros, membresia_fuerza)):
            area = ancho * (h - (h**2)/2)
            numerador += centro * area
            denominador += area
        
        if denominador == 0:
            return 0 
        
        fuerza = numerador / denominador
    elif metodo == 'ImplicacionProducto':
        numerador = sum([centro * 0.5 * ancho * membresia for centro, membresia in zip(centros, membresia_fuerza)])
        denominador = sum([0.5 * ancho * membresia for membresia in membresia_fuerza])
        if denominador == 0:
            return 0 
        fuerza = numerador / denominador
    elif metodo == 'PromedioCentros':
        numerador = sum([centro * membresia for centro, membresia in zip(centros, membresia_fuerza)])
        denominador = sum(membresia_fuerza) 
        if denominador == 0:
            return 0 
        fuerza = numerador / denominador
    else:
        raise ValueError("Método no válido")
    return fuerza
    
def probar_inferencia(error_rad, cambio_error_rad):
    print(f"Error: {np.degrees(error_rad):.2f}° | Cambio Error: {np.degrees(cambio_error_rad):.2f}°")
    membresia_error = fuzzificar_error(error_rad)
    membresia_cambio_error = fuzzificar_cambio_error(cambio_error_rad)
    print(f"Membresía Error: {[round(x, 4) for x in membresia_error]}")
    print(f"Membresía Cambio Error: {[round(x, 4) for x in membresia_cambio_error]}")
    membresia_fuerza = inferencia_fuzzy(error_rad, cambio_error_rad)
    print(f"Membresía Fuerza: {[round(x, 4) for x in membresia_fuerza]}")
    return membresia_fuerza

if __name__ == "__main__":
    error_rad = 1.390 
    cambio_error_rad = -0.0982 
    #metodos = ['COG', 'ImplicacionProducto', 'PromedioCentros']
    metodos = ['COG']
    membresia_fuerza = probar_inferencia(error_rad, cambio_error_rad)
    
    print("\nResultados de diferentes métodos de defuzzificación:")
    for metodo in metodos:
        fuerza = defuzzificador(membresia_fuerza, metodo)
        print(f"Método {metodo}: Fuerza de Control = {fuerza:.4f}")
