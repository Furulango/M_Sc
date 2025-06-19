import json
import pandas as pd

def convert_pwm_to_velocity_json(input_file='datos_16062025.txt', 
                                output_file='datos_velocidades_reales.json'):
    """
    Convierte archivo JSON de PWM a velocidades reales calculadas
    
    Lógica:
    - vel_d_act = vel_d_ref - vel_d_err  (velocidad real motor derecho)
    - vel_i_act = vel_i_ref - vel_i_err  (velocidad real motor izquierdo)
    """
    
    print(f"🔄 Convirtiendo {input_file} a velocidades reales...")
    
    converted_data = []
    
    try:
        with open(input_file, 'r') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:
                    try:
                        # Leer dato original
                        original_data = json.loads(line)
                        
                        # Crear nueva entrada con velocidades calculadas
                        new_data = original_data.copy()
                        
                        # CALCULAR VELOCIDADES REALES
                        # vel_act = vel_ref - vel_err (porque error = ref - actual)
                        new_data['vel_d_act'] = original_data['vel_d_ref'] - original_data['vel_d_err']
                        new_data['vel_i_act'] = original_data['vel_i_ref'] - original_data['vel_i_err']
                        
                        converted_data.append(new_data)
                        
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Error en línea {line_num}: {e}")
                        continue
        
        # Guardar archivo convertido
        with open(output_file, 'w') as file:
            for data_point in converted_data:
                json.dump(data_point, file)
                file.write('\n')
        
        print(f"✅ Conversión completada!")
        print(f"   • Datos originales: {len(converted_data)} muestras")
        print(f"   • Archivo generado: {output_file}")
        
        # Mostrar ejemplos de conversión
        print(f"\n📊 EJEMPLOS DE CONVERSIÓN:")
        print(f"="*60)
        
        for i in range(min(3, len(converted_data))):
            original_line = None
            with open(input_file, 'r') as f:
                for j, line in enumerate(f):
                    if j == i:
                        original_line = json.loads(line.strip())
                        break
            
            converted = converted_data[i]
            
            print(f"\nMuestra {i+1} (t={converted['t']}):")
            print(f"  Motor Derecho:")
            print(f"    Referencia: {converted['vel_d_ref']}")
            print(f"    Error: {converted['vel_d_err']}")
            print(f"    PWM original: {original_line['vel_d_act']}")
            print(f"    → Velocidad real: {converted['vel_d_act']}")
            
            print(f"  Motor Izquierdo:")
            print(f"    Referencia: {converted['vel_i_ref']}")
            print(f"    Error: {converted['vel_i_err']}")
            print(f"    PWM original: {original_line['vel_i_act']}")
            print(f"    → Velocidad real: {converted['vel_i_act']}")
        
        # Estadísticas de conversión
        df = pd.DataFrame(converted_data)
        
        print(f"\n📈 ESTADÍSTICAS DE VELOCIDADES REALES:")
        print(f"="*60)
        print(f"Motor Derecho:")
        print(f"  • Rango velocidad: {df['vel_d_act'].min():.1f} a {df['vel_d_act'].max():.1f}")
        print(f"  • Velocidad promedio: {df['vel_d_act'].mean():.1f}")
        print(f"  • Desviación estándar: {df['vel_d_act'].std():.1f}")
        
        print(f"\nMotor Izquierdo:")
        print(f"  • Rango velocidad: {df['vel_i_act'].min():.1f} a {df['vel_i_act'].max():.1f}")
        print(f"  • Velocidad promedio: {df['vel_i_act'].mean():.1f}")
        print(f"  • Desviación estándar: {df['vel_i_act'].std():.1f}")
        
        return output_file
        
    except FileNotFoundError:
        print(f"❌ Error: Archivo {input_file} no encontrado")
        return None
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return None

def validate_conversion(original_file, converted_file):
    """
    Valida que la conversión sea correcta
    """
    print(f"\n🔍 VALIDANDO CONVERSIÓN...")
    
    # Leer archivos
    original_data = []
    with open(original_file, 'r') as f:
        for line in f:
            if line.strip():
                original_data.append(json.loads(line.strip()))
    
    converted_data = []
    with open(converted_file, 'r') as f:
        for line in f:
            if line.strip():
                converted_data.append(json.loads(line.strip()))
    
    # Validar algunos puntos
    errors = 0
    for i in range(min(10, len(original_data))):
        orig = original_data[i]
        conv = converted_data[i]
        
        # Verificar cálculo motor derecho
        expected_d = orig['vel_d_ref'] - orig['vel_d_err']
        if abs(conv['vel_d_act'] - expected_d) > 0.1:
            print(f"❌ Error en muestra {i+1}, motor derecho")
            errors += 1
        
        # Verificar cálculo motor izquierdo
        expected_i = orig['vel_i_ref'] - orig['vel_i_err']
        if abs(conv['vel_i_act'] - expected_i) > 0.1:
            print(f"❌ Error en muestra {i+1}, motor izquierdo")
            errors += 1
    
    if errors == 0:
        print(f"✅ Validación exitosa - Conversión correcta")
    else:
        print(f"⚠️  Se encontraron {errors} errores en la validación")
    
    return errors == 0

def create_comparison_sample(original_file, converted_file, num_samples=5):
    """
    Crea un archivo de muestra comparando original vs convertido
    """
    print(f"\n📝 Creando archivo de comparación...")
    
    # Leer datos
    with open(original_file, 'r') as f:
        original_lines = [line.strip() for line in f if line.strip()]
    
    with open(converted_file, 'r') as f:
        converted_lines = [line.strip() for line in f if line.strip()]
    
    # Crear archivo de comparación
    comparison_file = 'comparacion_pwm_vs_velocidad.txt'
    with open(comparison_file, 'w') as f:
        f.write("COMPARACIÓN: PWM ORIGINAL vs VELOCIDADES REALES\n")
        f.write("="*80 + "\n\n")
        
        for i in range(min(num_samples, len(original_lines))):
            orig = json.loads(original_lines[i])
            conv = json.loads(converted_lines[i])
            
            f.write(f"MUESTRA {i+1} (t={orig['t']}):\n")
            f.write("-"*40 + "\n")
            f.write("ORIGINAL (PWM):\n")
            f.write(f"{original_lines[i]}\n\n")
            f.write("CONVERTIDO (Velocidades):\n") 
            f.write(f"{converted_lines[i]}\n\n")
            f.write("CAMBIOS:\n")
            f.write(f"  vel_d_act: {orig['vel_d_act']} (PWM) → {conv['vel_d_act']} (velocidad)\n")
            f.write(f"  vel_i_act: {orig['vel_i_act']} (PWM) → {conv['vel_i_act']} (velocidad)\n")
            f.write("="*80 + "\n\n")
    
    print(f"✅ Archivo de comparación creado: {comparison_file}")
    return comparison_file

# FUNCIÓN PRINCIPAL PARA USO DIRECTO
def convert_to_velocity_json(input_file='datos_16062025.txt', 
                           output_file='datos_velocidades_reales.json',
                           validate=True,
                           create_comparison=True):
    """
    Función completa de conversión con validación
    """
    print("🚗 CONVERTIDOR PWM → VELOCIDADES REALES")
    print("="*60)
    
    # Realizar conversión
    result_file = convert_pwm_to_velocity_json(input_file, output_file)
    
    if result_file:
        # Validar si se solicita
        if validate:
            validate_conversion(input_file, result_file)
        
        # Crear comparación si se solicita
        if create_comparison:
            create_comparison_sample(input_file, result_file)
        
        print(f"\n🎉 ¡CONVERSIÓN COMPLETADA!")
        print(f"   Archivo con velocidades reales: {result_file}")
        print(f"   Ahora puedes usar este archivo con cualquier monitor")
        
        return result_file
    
    return None

# EJEMPLO DE USO DIRECTO
if __name__ == "__main__":
    print("🔄 CONVERSIÓN AUTOMÁTICA PWM → VELOCIDADES")
    print("="*50)
    
    # Ejecutar conversión completa
    nuevo_archivo = convert_to_velocity_json()
    
    if nuevo_archivo:
        print(f"\n💡 CÓMO USAR EL ARCHIVO CONVERTIDO:")
        print(f"   1. Usar con monitor normal: monitor('{nuevo_archivo}')")
        print(f"   2. Las gráficas ahora mostrarán velocidades reales")
        print(f"   3. vel_d_act y vel_i_act están en unidades de velocidad")