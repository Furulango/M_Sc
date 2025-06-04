import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def load_data(path):
    """Carga los datos desde un archivo CSV."""
    try:
        print(f"Cargando datos desde: {path}")
        df = pd.read_csv(path)
        print(f"Forma original: {df.shape}")
        print(f"Distribución de clases original:")
        if 'Carpeta' in df.columns:
            print(df['Carpeta'].value_counts())
        return df
    except Exception as e:
        print(f"Error cargando CSV: {e}")
        return None

def bootstrap_augmentation(df, n_bootstraps=5, sample_size=None):
    """
    Realiza data augmentation usando bootstrap.
    
    Args:
        df: DataFrame original
        n_bootstraps: Número de muestras bootstrap a generar
        sample_size: Tamaño de cada muestra bootstrap (por defecto, mismo tamaño que df)
    
    Returns:
        DataFrame aumentado
    """
    if df is None:
        return None
    
    if 'Carpeta' not in df.columns:
        print("Error: Columna 'Carpeta' no encontrada.")
        return df
    
    # Si no se especifica un tamaño de muestra, usar el mismo tamaño que el dataset original
    if sample_size is None:
        sample_size = len(df)
    
    # Guardar el DataFrame original
    augmented_df = df.copy()
    
    # Identificar las columnas de características
    feature_cols = [col for col in df.columns if col not in ['Carpeta', 'Imagen']]
    
    # Realizar bootstrap por clase para mantener la distribución
    classes = df['Carpeta'].unique()
    
    print(f"Generando {n_bootstraps} muestras bootstrap por clase...")
    
    for class_name in classes:
        # Obtener solo los datos de esta clase
        class_data = df[df['Carpeta'] == class_name]
        
        # Calcular el tamaño proporcional para esta clase
        class_proportion = len(class_data) / len(df)
        class_sample_size = int(sample_size * class_proportion)
        
        # Generar n_bootstraps muestras bootstrap para esta clase
        for i in range(n_bootstraps):
            # Remuestreo con reemplazo
            bootstrap_sample = resample(
                class_data, 
                replace=True,
                n_samples=class_sample_size,
                random_state=42+i
            )
            
            # Añadir pequeña variación gaussiana a las características numéricas para evitar duplicados exactos
            noise_factor = 0.01  # Factor de ruido, ajustar según sea necesario
            for col in feature_cols:
                if pd.api.types.is_numeric_dtype(bootstrap_sample[col]):
                    # Calcular la desviación estándar de la columna
                    col_std = bootstrap_sample[col].std()
                    # Si la desviación estándar es 0, usar un valor pequeño para evitar multiplicar por 0
                    if col_std == 0:
                        col_std = 0.001
                    # Añadir ruido gaussiano proporcional a la desviación estándar
                    noise = np.random.normal(0, noise_factor * col_std, size=len(bootstrap_sample))
                    bootstrap_sample[col] = bootstrap_sample[col] + noise
            
            # Añadir identificador a la muestra bootstrap para seguimiento
            bootstrap_sample = bootstrap_sample.copy()
            if 'Imagen' in bootstrap_sample.columns:
                # Convertir la columna a string primero y luego concatenar
                bootstrap_sample['Imagen'] = bootstrap_sample['Imagen'].astype(str) + f"_b{i+1}"
            
            # Añadir la muestra bootstrap al DataFrame aumentado
            augmented_df = pd.concat([augmented_df, bootstrap_sample], ignore_index=True)
    
    print(f"Forma después de augmentation: {augmented_df.shape}")
    print(f"Distribución de clases después de augmentation:")
    print(augmented_df['Carpeta'].value_counts())
    
    return augmented_df

def save_augmented_data(df, original_path, output_dir=None):
    """Guarda los datos aumentados en un nuevo archivo CSV."""
    if df is None:
        return None
    
    # Crear directorio de salida si no existe
    if output_dir is None:
        output_dir = os.path.dirname(original_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar nombre del archivo de salida
    filename = os.path.basename(original_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_augmented{ext}")
    
    # Guardar DataFrame aumentado
    df.to_csv(output_path, index=False)
    print(f"Datos aumentados guardados en: {output_path}")
    
    return output_path

def main():
    # Obtener ruta al archivo CSV
    csv_path = input("Ingrese la ruta al archivo CSV: ")
    
    # Cargar datos
    df = load_data(csv_path)
    
    if df is not None:
        # Preguntar parámetros de augmentation
        print("\nParámetros para data augmentation:")
        n_bootstraps = input("Número de muestras bootstrap por clase (por defecto: 5): ")
        n_bootstraps = int(n_bootstraps) if n_bootstraps.strip() else 5
        
        use_default_size = input("¿Usar tamaño de muestra proporcional al original? (s/n, por defecto: s): ")
        if use_default_size.lower() != 'n':
            sample_size = None
        else:
            sample_size = input("Tamaño total de cada muestra bootstrap: ")
            sample_size = int(sample_size) if sample_size.strip() else len(df)
        
        output_dir = input("Carpeta para guardar el dataset aumentado (por defecto: misma carpeta): ")
        output_dir = output_dir if output_dir.strip() else None
        
        # Realizar data augmentation
        print("\nRealizando data augmentation mediante bootstrap...")
        augmented_df = bootstrap_augmentation(df, n_bootstraps, sample_size)
        
        # Guardar datos aumentados
        if augmented_df is not None:
            save_augmented_data(augmented_df, csv_path, output_dir)
            
            # Mostrar estadísticas de aumento
            original_count = len(df)
            augmented_count = len(augmented_df)
            increase_percent = ((augmented_count / original_count) - 1) * 100
            
            print("\nEstadísticas del aumento de datos:")
            print(f"- Cantidad original de registros: {original_count}")
            print(f"- Cantidad después del aumento: {augmented_count}")
            print(f"- Incremento: {increase_percent:.2f}%")
            
            print("\nEl dataset aumentado está listo para ser utilizado con los modelos.")

if __name__ == "__main__":
    main()