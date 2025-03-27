def visualizar_activacion_reglas(error_rad, cambio_error_rad):
    membresia_error = fuzzificar_error(error_rad)
    membresia_cambio_error = fuzzificar_cambio_error(cambio_error_rad)
    membresia_fuerza, fuerza = inferencia_fuzzy(error_rad, cambio_error_rad)
    error_grados = np.degrees(error_rad)
    cambio_error_grados = np.degrees(cambio_error_rad)
    tabla_reglas = [
        [2, 2, 2, 1, 0],
        [2, 2, 1, 0, -1],
        [2, 1, 0, -1, -2],
        [1, 0, -1, -2, -2],
        [0, -1, -2, -2, -2]
    ]
    etiquetas_error = ["NegGrande", "NegPequeño", "Cero", "PosPequeño", "PosGrande"]
    etiquetas_cambio = ["NegGrande", "NegPequeño", "Cero", "PosPequeño", "PosGrande"]
    etiquetas_fuerza = ["NegGrande", "NegPequeño", "Cero", "PosPequeño", "PosGrande"]
    plt.figure(figsize=(15, 12))
    plt.subplot(3, 1, 1)
    plt.bar(etiquetas_error, membresia_error, color='skyblue')
    plt.title(f'Membresía del Error: {error_grados:.2f}°')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.subplot(3, 1, 2)
    plt.bar(etiquetas_cambio, membresia_cambio_error, color='lightgreen')
    plt.title(f'Membresía del Cambio de Error: {cambio_error_grados:.2f}°')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.subplot(3, 1, 3)
    plt.bar(etiquetas_fuerza, membresia_fuerza, color='salmon')
    plt.title(f'Membresía de la Fuerza de Control: {fuerza:.4f}')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("\nDetalle de Activación de Reglas:")
    print("--------------------------------")
    print(f"Error: {error_grados:.2f}° | Cambio de Error: {cambio_error_grados:.2f}°")
    print("--------------------------------")
    tabla_activacion = []
    for i in range(5):
        for j in range(5):
            activacion = min(membresia_error[i], membresia_cambio_error[j])
            if activacion > 0:
                indice_salida = tabla_reglas[i][j]
                fuerza_etiqueta = etiquetas_fuerza[indice_salida + 2]
                tabla_activacion.append([
                    etiquetas_error[i], 
                    etiquetas_cambio[j], 
                    fuerza_etiqueta, 
                    activacion
                ])
    tabla_activacion.sort(key=lambda x: x[3], reverse=True)
    df_activacion = pd.DataFrame(
        tabla_activacion, 
        columns=["Error", "Cambio Error", "Fuerza", "Activación"]
    )
    if not df_activacion.empty:
        print(df_activacion)
    else:
        print("No hay reglas activadas con valor significativo.")
    print("--------------------------------")
    print(f"Fuerza defuzzificada: {fuerza:.4f}")
