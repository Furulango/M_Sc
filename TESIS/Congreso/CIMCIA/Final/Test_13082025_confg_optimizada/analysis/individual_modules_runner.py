#!/usr/bin/env python3
"""
Ejecutor de Módulos Individuales
Sistema Modular de Análisis - Gemelo Digital Adaptativo

Este script permite ejecutar módulos específicos de análisis
sin tener que correr todo el sistema completo.

Uso:
    python individual_modules.py --module 1
    python individual_modules.py --module 2
    python individual_modules.py --all
    python individual_modules.py --help

Autor: [Tu Nombre]
Fecha: 2024
"""

import argparse
import sys
from pathlib import Path

# Importar la clase principal (asume que está en el mismo directorio)
from adaptive_analysis_system import AdaptiveDigitalTwinAnalyzer

def run_module_1(analyzer):
    """Ejecutar solo Módulo 1: Análisis de Adaptabilidad"""
    print("🔥 EJECUTANDO MÓDULO 1: ANÁLISIS DE ADAPTABILIDAD")
    return analyzer.module_1_adaptability_analysis()

def run_module_2(analyzer):
    """Ejecutar solo Módulo 2: Heatmap de Parámetros DQ"""
    print("📊 EJECUTANDO MÓDULO 2: HEATMAP DE PARÁMETROS DQ")
    return analyzer.module_2_parameter_heatmap()

def run_module_3(analyzer):
    """Ejecutar solo Módulo 3: Dashboard Comparativo"""
    print("⚡ EJECUTANDO MÓDULO 3: DASHBOARD COMPARATIVO")
    return analyzer.module_3_comparative_dashboard()

def run_module_4(analyzer):
    """Ejecutar solo Módulo 4: Análisis Estadístico"""
    print("📈 EJECUTANDO MÓDULO 4: ANÁLISIS ESTADÍSTICO ROBUSTO")
    return analyzer.module_4_statistical_analysis()

def run_module_5(analyzer):
    """Ejecutar solo Módulo 5: Análisis de Convergencia"""
    print("🎯 EJECUTANDO MÓDULO 5: ANÁLISIS DE CONVERGENCIA")
    return analyzer.module_5_convergence_analysis()

def run_quick_analysis(analyzer):
    """Ejecutar análisis rápido (solo módulos 1, 2, 3)"""
    print("⚡ EJECUTANDO ANÁLISIS RÁPIDO (Módulos 1, 2, 3)")
    results = {}
    results['module_1'] = run_module_1(analyzer)
    results['module_2'] = run_module_2(analyzer)
    results['module_3'] = run_module_3(analyzer)
    return results

def run_statistical_focus(analyzer):
    """Ejecutar enfoque estadístico (módulos 4 y 5)"""
    print("📊 EJECUTANDO ENFOQUE ESTADÍSTICO (Módulos 4, 5)")
    results = {}
    results['module_4'] = run_module_4(analyzer)
    results['module_5'] = run_module_5(analyzer)
    return results

def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Ejecutor de Módulos Individuales - Sistema de Análisis Adaptativo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Ejecutar módulo específico
  python individual_modules.py --module 1
  
  # Ejecutar múltiples módulos
  python individual_modules.py --module 1 2 3
  
  # Ejecutar todos los módulos
  python individual_modules.py --all
  
  # Análisis rápido (módulos principales)
  python individual_modules.py --quick
  
  # Análisis estadístico
  python individual_modules.py --statistical
  
  # Especificar ruta de datos
  python individual_modules.py --module 1 --data_path ./data/

Descripción de módulos:
  1 - Análisis de Adaptabilidad (Fase 1 vs Fase 2)
  2 - Heatmap de Parámetros DQ  
  3 - Dashboard Comparativo
  4 - Análisis Estadístico Robusto
  5 - Análisis de Convergencia
        """
    )
    
    # Argumentos principales
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--module', '-m', type=int, nargs='+', choices=[1,2,3,4,5],
                      help='Número(s) de módulo(s) a ejecutar (1-5)')
    group.add_argument('--all', '-a', action='store_true',
                      help='Ejecutar todos los módulos')
    group.add_argument('--quick', '-q', action='store_true',
                      help='Análisis rápido (módulos 1, 2, 3)')
    group.add_argument('--statistical', '-s', action='store_true',
                      help='Análisis estadístico (módulos 4, 5)')
    
    # Argumentos opcionales
    parser.add_argument('--data_path', '-d', type=str, default='./',
                       help='Ruta donde están los archivos CSV (default: ./)')
    parser.add_argument('--output', '-o', type=str, default='results_analysis',
                       help='Directorio de salida (default: results_analysis)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Salida detallada')
    
    args = parser.parse_args()
    
    # Verificar que los archivos CSV existen
    data_path = Path(args.data_path)
    required_files = ['BFO_adaptive_results.csv', 'PSO_adaptive_results.csv', 'Chaotic_PSODSO_adaptive_results.csv']
    
    missing_files = [f for f in required_files if not (data_path / f).exists()]
    if missing_files:
        print(f"❌ ERROR: Archivos CSV no encontrados:")
        for f in missing_files:
            print(f"   - {data_path / f}")
        print(f"\n💡 Asegúrate de que los archivos CSV estén en: {data_path.absolute()}")
        sys.exit(1)
    
    # Configurar verbosidad
    if not args.verbose:
        import warnings
        warnings.filterwarnings('ignore')
    
    print("🔬 EJECUTOR DE MÓDULOS INDIVIDUALES")
    print("Sistema de Análisis de Gemelo Digital Adaptativo")
    print("=" * 60)
    print(f"📂 Datos: {data_path.absolute()}")
    print(f"📁 Salida: {args.output}")
    print("=" * 60)
    
    try:
        # Inicializar analizador
        analyzer = AdaptiveDigitalTwinAnalyzer(data_path=args.data_path)
        
        # Ejecutar módulos según argumentos
        results = {}
        
        if args.all:
            print("\n🚀 EJECUTANDO TODOS LOS MÓDULOS")
            results = analyzer.run_all_modules()
            
        elif args.quick:
            results = run_quick_analysis(analyzer)
            
        elif args.statistical:
            results = run_statistical_focus(analyzer)
            
        elif args.module:
            print(f"\n🎯 EJECUTANDO MÓDULOS: {args.module}")
            
            module_functions = {
                1: run_module_1,
                2: run_module_2,
                3: run_module_3,
                4: run_module_4,
                5: run_module_5
            }
            
            for module_num in args.module:
                if module_num in module_functions:
                    results[f'module_{module_num}'] = module_functions[module_num](analyzer)
                else:
                    print(f"⚠️ Módulo {module_num} no válido")
        
        # Resumen final
        print(f"\n✅ EJECUCIÓN COMPLETADA")
        print(f"📊 Módulos ejecutados: {len(results)}")
        print(f"📁 Resultados en: {analyzer.results_path}")
        
        # Mostrar archivos generados
        if analyzer.results_path.exists():
            generated_files = list(analyzer.results_path.rglob("*"))
            if generated_files:
                print(f"\n📄 Archivos generados:")
                for file in sorted(generated_files):
                    if file.is_file():
                        rel_path = file.relative_to(analyzer.results_path)
                        print(f"   • {rel_path}")
        
        return analyzer, results
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR INESPERADO: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    analyzer, results = main()
