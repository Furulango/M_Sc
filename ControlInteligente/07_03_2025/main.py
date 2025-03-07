import numpy as np

def controlador_difuso(error_rad):
    """
    Controlador de lógica difusa que convierte un valor de error (en radianes) a un vector de membresía F.
    
    Args:
        error_rad (float): Valor de error en radianes
        
    Returns:
        list: Vector F = [neggrande, negpequeño, cero, pospequeño, posgrande] con valores de membresía
                que suman exactamente 1
    """
    # Asegurar que el error esté dentro de los límites
    error_rad = max(-np.pi/2, min(np.pi/2, error_rad))
    
    # Inicializar valores de membresía
    F = [0.0, 0.0, 0.0, 0.0, 0.0]  # [negativogrande, negativopequeño, cero, positivopequeño, positivogrande]
    
    # Calcular membresías basadas en el diagrama
    if error_rad <= -np.pi/2:
        # Completamente neggrande
        F[0] = 1.0
    elif error_rad <= -np.pi/4:
        # Entre neggrande y negpequeño
        ratio = (error_rad - (-np.pi/2)) / (np.pi/4)  # 0 en -pi/2, 1 en -pi/4
        F[0] = 1 - ratio
        F[1] = ratio
    elif error_rad <= 0:
        # Entre negpequeño y cero
        ratio = (error_rad - (-np.pi/4)) / (np.pi/4)  # 0 en -pi/4, 1 en 0
        F[1] = 1 - ratio
        F[2] = ratio
    elif error_rad <= np.pi/4:
        # Entre cero y pospequeño
        ratio = (error_rad - 0) / (np.pi/4)  # 0 en 0, 1 en pi/4
        F[2] = 1 - ratio
        F[3] = ratio
    elif error_rad <= np.pi/2:
        # Entre pospequeño y posgrande
        ratio = (error_rad - (np.pi/4)) / (np.pi/4)  # 0 en pi/4, 1 en pi/2
        F[3] = 1 - ratio
        F[4] = ratio
    else:
        # Completamente posgrande
        F[4] = 1.0
    
    # Asegurar que la suma sea exactamente 1 (manejar errores de punto flotante)
    total = sum(F)
    if total > 0:
        F = [x/total for x in F]
    
    return F

# Ejemplo de uso
if __name__ == "__main__":
    # Probar con diferentes valores de error
    errores_prueba = [-np.pi/2, -np.pi/3, -np.pi/4, -np.pi/8, 0, np.pi/8, np.pi/4, np.pi/3, np.pi/2]
    
    for error in errores_prueba:
        vector_membresia = controlador_difuso(error)
        print(f"Error: {error:.4f} rad, F = {[round(x, 4) for x in vector_membresia]}")