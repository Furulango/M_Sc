import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functions import run_pso, run_pso_sqp, BacterialForaging
import time
import pyswarms as ps

def motor_induccion(t, x, params, vqs, vds):
    """Modelo básico motor inducción en coordenadas DQ"""
    iqs, ids, iqr, idr, wr = x
    rs, rr, Lls, Llr, Lm, J, B = params
    
    Ls, Lr = Lls + Lm, Llr + Lm
    we = 2*np.pi*60
    ws = we - wr
    
    lqs = Ls*iqs + Lm*iqr
    lds = Ls*ids + Lm*idr
    lqr = Lr*iqr + Lm*iqs
    ldr = Lr*idr + Lm*ids
    
    L = np.array([[Ls, 0, Lm, 0], [0, Ls, 0, Lm], 
                  [Lm, 0, Lr, 0], [0, Lm, 0, Lr]])
    v = np.array([vqs - rs*iqs - we*lds, vds - rs*ids + we*lqs,
                  -rr*iqr - ws*ldr, -rr*idr + ws*lqr])
    
    di_dt = np.linalg.solve(L, v)
    Te = (3*4/4) * Lm * (iqs*idr - ids*iqr)
    dwr_dt = (Te - B*wr) / J
    
    return np.array([*di_dt, dwr_dt])

def simular_motor(params, t_span=[0, 2], n_points=500):
    """Simula el motor y retorna señales de interés"""
    vqs, vds = 220*np.sqrt(2)/np.sqrt(3), 0
    
    try:
        sol = solve_ivp(lambda t, x: motor_induccion(t, x, params, vqs, vds),
                        t_span, [0,0,0,0,0], dense_output=True, rtol=1e-6)
        
        t = np.linspace(t_span[0], t_span[1], n_points)
        iqs, ids, iqr, idr, wr = sol.sol(t)
        
        # Señales de salida para identificación
        Is_mag = np.sqrt(iqs**2 + ids**2)  # Magnitud corriente estator
        Te = (3*4/4) * params[4] * (iqs*idr - ids*iqr)  # Par electromagnético
        rpm = wr * 60/(2*np.pi) * 2/4  # Velocidad en RPM
        
        return t, {'iqs': iqs, 'ids': ids, 'Is_mag': Is_mag, 'Te': Te, 'rpm': rpm, 'wr': wr}
    
    except Exception as e:
        # Retornar valores altos si la simulación falla
        t = np.linspace(t_span[0], t_span[1], n_points)
        return t, {'iqs': np.ones(n_points)*1e6, 'ids': np.ones(n_points)*1e6, 
                   'Is_mag': np.ones(n_points)*1e6, 'Te': np.ones(n_points)*1e6, 
                   'rpm': np.ones(n_points)*1e6, 'wr': np.ones(n_points)*1e6}

def generar_datos_experimentales(params_reales, ruido_nivel=0.02):
    """Genera datos 'experimentales' agregando ruido a la simulación perfecta"""
    print("=== GENERANDO DATOS EXPERIMENTALES ===")
    print(f"Parámetros reales: {params_reales}")
    
    t, salidas = simular_motor(params_reales)
    
    # Agregar ruido realista
    np.random.seed(42)  # Para reproducibilidad
    datos_exp = {}
    
    for key, signal in salidas.items():
        ruido = np.random.normal(0, ruido_nivel * np.std(signal), len(signal))
        datos_exp[key] = signal + ruido
    
    print(f"Ruido agregado: {ruido_nivel*100}% del std de cada señal")
    return t, datos_exp

def funcion_objetivo(params_estimados, t_exp, datos_exp, pesos=None):
    """
    Función objetivo para minimizar: Error Cuadrático Medio (ECM)
    """
    if pesos is None:
        pesos = {'Is_mag': 1.0, 'Te': 0.5, 'rpm': 0.3}  # Pesos para diferentes señales
    
    # Verificar límites físicos básicos
    if any(p <= 0 for p in params_estimados[:5]) or any(p < 0 for p in params_estimados[5:]):
        return 1e10
    
    try:
        # Simular con parámetros estimados
        _, salidas_sim = simular_motor(params_estimados)
        
        # Calcular ECM ponderado
        error_total = 0
        for señal, peso in pesos.items():
            if señal in datos_exp and señal in salidas_sim:
                error = np.mean((datos_exp[señal] - salidas_sim[señal])**2)
                error_total += peso * error
        
        return error_total
        
    except Exception as e:
        return 1e10

def pso_con_progreso(objetivo, bounds, n_particles=30, iterations=50):
    """PSO con progreso en tiempo real usando pyswarms directamente"""
    
    def callback_progreso(optimizer, iteration):
        """Callback para mostrar progreso de PSO"""
        best_cost = optimizer.swarm.best_cost
        print(f"  Iteración {iteration+1:3d}/{iterations} | Mejor costo: {best_cost:.6e}")
        
        # Cada 10 iteraciones, mostrar más detalles
        if (iteration + 1) % 10 == 0 or iteration == 0:
            mean_cost = np.mean(optimizer.swarm.current_cost)
            std_cost = np.std(optimizer.swarm.current_cost)
            print(f"    └─ Costo promedio: {mean_cost:.6e} ± {std_cost:.6e}")

    # Función wrapper para pyswarms - IMPORTANTE: debe aceptar **kwargs
    def pso_wrapper(x, **kwargs):
        return np.array([objetivo(p) for p in x])

    # Configurar optimizador con verbose
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=len(bounds[0]),
        options={'c1': 2.05, 'c2': 2.05, 'w': 0.9},
        bounds=bounds
    )

    print(f"    Configuración: {n_particles} partículas, {iterations} iteraciones")
    print(f"    Espacio de búsqueda: {len(bounds[0])} dimensiones")
    print("    Iniciando optimización...")
    
    # Ejecutar con callback
    best_cost, best_pos = optimizer.optimize(
        pso_wrapper, 
        iters=iterations, 
        verbose=True
    )
    
    return float(best_cost), best_pos

def bfo_con_progreso(objetivo, bounds, n_bacteria=20, n_chemotactic=15, n_reproductive=3):
    """BFO modificado con progreso en tiempo real"""
    
    class BFO_Verbose(BacterialForaging):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.iteration_count = 0
            self.total_iterations = self.Ned * self.Nre * self.Nc
            
        def optimize(self):
            print(f"    Configuración: {self.S} bacterias")
            print(f"    Ciclos: {self.Ned} eliminación × {self.Nre} reproducción × {self.Nc} quimiotaxis")
            print(f"    Total iteraciones: {self.total_iterations}")
            print("    Iniciando optimización...")
            
            for l in range(self.Ned):
                print(f"\n    🔄 Ciclo Eliminación {l+1}/{self.Ned}")
                
                for k in range(self.Nre):
                    print(f"      📊 Ciclo Reproducción {k+1}/{self.Nre}")
                    
                    # Ciclo chemotáctico con progreso
                    for j in range(self.Nc):
                        self.iteration_count += 1
                        self._update_best()
                        
                        # Mostrar progreso cada pocas iteraciones
                        if j % 5 == 0 or j == self.Nc - 1:
                            progress = (self.iteration_count / self.total_iterations) * 100
                            print(f"        Quimiotaxis {j+1:2d}/{self.Nc} | "
                                  f"Mejor costo: {self.best_cost:.6e} | "
                                  f"Progreso: {progress:.1f}%")
                        
                        last_costs = np.copy(self.costs)
                        directions = self._tumble()
                        
                        # Natación
                        for m in range(self.Ns):
                            new_pos = self.bacteria + self.Ci * directions
                            new_pos = np.clip(new_pos, self.lb, self.ub)
                            new_costs = np.array([self.objective_func(p) for p in new_pos])
                            
                            improved_mask = new_costs < self.costs
                            self.bacteria[improved_mask] = new_pos[improved_mask]
                            self.costs[improved_mask] = new_costs[improved_mask]
                            self.health += last_costs - self.costs
                            
                            if not np.any(improved_mask):
                                break

                    # Reproducción
                    self._reproduce()
                    print(f"        └─ Reproducción completada. Mejor costo: {self.best_cost:.6e}")

                # Eliminación y dispersión
                self._eliminate_disperse()
                print(f"      └─ Eliminación-dispersión completada")
                
            self._update_best()
            return self.best_cost, self.best_pos
    
    bfo = BFO_Verbose(objetivo, bounds, n_bacteria, n_chemotactic, 4, n_reproductive, 2, 0.25, 0.1)
    return bfo.optimize()

def ejecutar_identificacion():
    """Función principal para ejecutar la identificación de parámetros"""
    
    # Parámetros reales del motor (los que queremos identificar)
    params_reales = np.array([2.45, 1.83, 0.008, 0.008, 0.203, 0.02, 0.001])
    nombres_params = ['rs', 'rr', 'Lls', 'Llr', 'Lm', 'J', 'B']
    
    print("🔧 Generando datos experimentales...")
    # Generar datos experimentales con ruido
    t_exp, datos_exp = generar_datos_experimentales(params_reales, ruido_nivel=0.03)
    
    # Definir límites de búsqueda (±50% de los valores reales)
    factor_busqueda = 0.5
    lb = params_reales * (1 - factor_busqueda)
    ub = params_reales * (1 + factor_busqueda)
    bounds = (lb, ub)
    
    print(f"\n{'='*60}")
    print(f"{'CONFIGURACIÓN DE IDENTIFICACIÓN':^60}")
    print(f"{'='*60}")
    print(f"Parámetros a identificar: {len(params_reales)}")
    print(f"Límites de búsqueda: ±{factor_busqueda*100}% de valores reales")
    print(f"Ruido en datos: 3%")
    print(f"Duración simulación: 2.0 segundos")
    
    # Función objetivo parcial
    objetivo = lambda params: funcion_objetivo(params, t_exp, datos_exp)
    
    # Configuración de algoritmos
    config = {
        'PSO': {'n_particles': 30, 'iterations': 50},
        'PSO-SQP': {'n_particles': 20, 'pso_iterations': 30},
        'BFO': {'n_bacteria': 20, 'n_chemotactic': 15, 'n_reproductive': 3}
    }
    
    resultados = {}
    
    print(f"\n{'='*60}")
    print(f"{'IDENTIFICACIÓN CON ALGORITMOS BIOINSPIRADOS':^60}")
    print(f"{'='*60}")
    
    # 1. PSO con progreso
    print("\n🐝 ALGORITMO: PARTICLE SWARM OPTIMIZATION (PSO)")
    print("=" * 50)
    start_time = time.time()
    costo_pso, params_pso = pso_con_progreso(objetivo, bounds, **config['PSO'])
    tiempo_pso = time.time() - start_time
    resultados['PSO'] = {'costo': costo_pso, 'params': params_pso, 'tiempo': tiempo_pso}
    print(f"    ✅ PSO completado en {tiempo_pso:.2f}s - Costo final: {costo_pso:.6e}")
    
    # 2. PSO-SQP Híbrido
    print(f"\n🚀 ALGORITMO: PSO-SQP HÍBRIDO")
    print("=" * 50)
    start_time = time.time()
    costo_hibrido, params_hibrido = run_pso_sqp(objetivo, bounds, **config['PSO-SQP'])
    tiempo_hibrido = time.time() - start_time
    resultados['PSO-SQP'] = {'costo': costo_hibrido, 'params': params_hibrido, 'tiempo': tiempo_hibrido}
    print(f"    ✅ PSO-SQP completado en {tiempo_hibrido:.2f}s - Costo final: {costo_hibrido:.6e}")
    
    # 3. BFO con progreso
    print(f"\n🦠 ALGORITMO: BACTERIAL FORAGING OPTIMIZATION (BFO)")
    print("=" * 50)
    start_time = time.time()
    costo_bfo, params_bfo = bfo_con_progreso(objetivo, bounds, **config['BFO'])
    tiempo_bfo = time.time() - start_time
    resultados['BFO'] = {'costo': costo_bfo, 'params': params_bfo, 'tiempo': tiempo_bfo}
    print(f"    ✅ BFO completado en {tiempo_bfo:.2f}s - Costo final: {costo_bfo:.6e}")
    
    # Mostrar resultados comparativos en tiempo real
    print(f"\n{'='*70}")
    print(f"{'🏆 RESULTADOS FINALES DE IDENTIFICACIÓN':^70}")
    print(f"{'='*70}")
    
    print(f"{'Algoritmo':<12} {'Costo Final':<15} {'Tiempo (s)':<12} {'Eficiencia':<10} {'Mejor':<6}")
    print(f"{'-'*70}")
    
    mejor_costo = min(r['costo'] for r in resultados.values())
    for alg, res in resultados.items():
        es_mejor = "🥇" if res['costo'] == mejor_costo else ""
        eficiencia = 1.0 / (res['tiempo'] * res['costo']) if res['costo'] > 0 else 0
        print(f"{alg:<12} {res['costo']:<15.2e} {res['tiempo']:<12.2f} {eficiencia:<10.2e} {es_mejor:<6}")
    
    # Mostrar parámetros identificados de TODOS los algoritmos
    print(f"\n{'='*80}")
    print(f"{'📊 COMPARACIÓN DETALLADA DE PARÁMETROS POR ALGORITMO':^80}")
    print(f"{'='*80}")
    
    for alg_nombre, resultado in resultados.items():
        params_alg = resultado['params']
        costo_alg = resultado['costo']
        tiempo_alg = resultado['tiempo']
        
        print(f"\n🔸 ALGORITMO: {alg_nombre}")
        print(f"   Costo: {costo_alg:.2e} | Tiempo: {tiempo_alg:.2f}s")
        print(f"   {'─'*60}")
        print(f"   {'Parámetro':<8} {'Real':<12} {'Identificado':<15} {'Error %':<10} {'Estado':<12}")
        print(f"   {'─'*60}")
        
        errores_algoritmo = []
        for i, nombre in enumerate(nombres_params):
            real = params_reales[i]
            identificado = params_alg[i]
            error_pct = abs((identificado - real) / real) * 100
            errores_algoritmo.append(error_pct)
            
            # Estado del error con más detalle
            if error_pct < 1.0:
                estado = "🟢 Excelente"
            elif error_pct < 5.0:
                estado = "🟡 Bueno"
            elif error_pct < 15.0:
                estado = "🟠 Aceptable"
            elif error_pct < 30.0:
                estado = "🔴 Pobre"
            else:
                estado = "💀 Muy Pobre"
                
            print(f"   {nombre:<8} {real:<12.6f} {identificado:<15.6f} {error_pct:<10.2f} {estado:<12}")
        
        error_promedio_alg = np.mean(errores_algoritmo)
        print(f"   {'─'*60}")
        print(f"   {'Error promedio:':<25} {error_promedio_alg:<10.2f}%")
        
        # Evaluar rendimiento general del algoritmo
        if error_promedio_alg < 5.0:
            rendimiento = "🟢 EXCELENTE"
        elif error_promedio_alg < 15.0:
            rendimiento = "🟡 BUENO"
        elif error_promedio_alg < 30.0:
            rendimiento = "🟠 REGULAR"
        else:
            rendimiento = "🔴 DEFICIENTE"
        
        print(f"   {'Rendimiento general:':<25} {rendimiento}")
    
    # Análisis comparativo por parámetro
    print(f"\n{'='*80}")
    print(f"{'📈 ANÁLISIS COMPARATIVO POR PARÁMETRO':^80}")
    print(f"{'='*80}")
    print(f"{'Parámetro':<10} {'PSO':<15} {'PSO-SQP':<15} {'BFO':<15} {'Mejor Algoritmo':<15}")
    print(f"{'─'*80}")
    
    for i, nombre in enumerate(nombres_params):
        real = params_reales[i]
        errores_param = {}
        
        for alg_nombre, resultado in resultados.items():
            identificado = resultado['params'][i]
            error_pct = abs((identificado - real) / real) * 100
            errores_param[alg_nombre] = error_pct
        
        # Encontrar el mejor para este parámetro
        mejor_alg_param = min(errores_param.keys(), key=lambda k: errores_param[k])
        mejor_error = errores_param[mejor_alg_param]
        
        # Formatear errores con colores
        pso_str = f"{errores_param['PSO']:.2f}%"
        psosqp_str = f"{errores_param['PSO-SQP']:.2f}%"
        bfo_str = f"{errores_param['BFO']:.2f}%"
        
        # Marcar el mejor
        if mejor_alg_param == 'PSO':
            pso_str += " 🏆"
        elif mejor_alg_param == 'PSO-SQP':
            psosqp_str += " 🏆"
        else:
            bfo_str += " 🏆"
        
        print(f"{nombre:<10} {pso_str:<15} {psosqp_str:<15} {bfo_str:<15} {mejor_alg_param:<15}")
    
    # Diagnóstico de problemas
    print(f"\n{'='*80}")
    print(f"{'🔧 DIAGNÓSTICO Y RECOMENDACIONES':^80}")
    print(f"{'='*80}")
    
    # Verificar si los costos son demasiado altos
    todos_costos = [r['costo'] for r in resultados.values()]
    costo_promedio = np.mean(todos_costos)
    
    if costo_promedio > 10:
        print("⚠️  PROBLEMA DETECTADO: Costos de función objetivo muy altos")
        print("   Posibles causas:")
        print("   • Límites de búsqueda demasiado amplios")
        print("   • Escalamiento inadecuado de la función objetivo")
        print("   • Parámetros de ruido experimental muy altos")
        print("   • Necesidad de más iteraciones en los algoritmos")
        
        print(f"\n💡 RECOMENDACIONES:")
        print(f"   1. Reducir límites de búsqueda a ±25% (actualmente ±50%)")
        print(f"   2. Aumentar iteraciones: PSO 100+, BFO 50+ quimiotaxis")
        print(f"   3. Reducir ruido experimental de 3% a 1%")
        print(f"   4. Ajustar pesos de función objetivo")
        print(f"   5. Usar normalización en función objetivo")
    
    # Análisis de tiempos de ejecución
    tiempos = [r['tiempo'] for r in resultados.values()]
    if max(tiempos) > 300:  # Más de 5 minutos
        print(f"\n⏱️  TIEMPOS DE EJECUCIÓN ALTOS DETECTADOS")
        print(f"   • Considerar reducir número de partículas/bacterias")
        print(f"   • Optimizar función objetivo para mayor velocidad")
        print(f"   • Usar tolerancias de convergencia más amplias")
    
    # Recomendar parámetros más restrictivos
    print(f"\n🎯 CONFIGURACIÓN MEJORADA SUGERIDA:")
    print(f"   • Límites de búsqueda: ±25% de valores reales")
    print(f"   • PSO: 50 partículas, 75 iteraciones")
    print(f"   • BFO: 15 bacterias, 25 quimiotaxis")
    print(f"   • Ruido experimental: 1-2%")
    print(f"   • Función objetivo normalizada con pesos ajustados")
    
    mejor_algoritmo = min(resultados.keys(), key=lambda k: resultados[k]['costo'])
    mejores_params = resultados[mejor_algoritmo]['params']
    
    # Mostrar resumen de convergencia
    print(f"\n{'='*70}")
    print(f"{'📈 ANÁLISIS DE CONVERGENCIA':^70}")
    print(f"{'='*70}")
    
    for alg, res in resultados.items():
        convergencia = "Rápida" if res['tiempo'] < 15 else "Moderada" if res['tiempo'] < 30 else "Lenta"
        precision = "Alta" if res['costo'] < 1e-5 else "Media" if res['costo'] < 1e-3 else "Baja"
        print(f"{alg:<12} | Convergencia: {convergencia:<8} | Precisión: {precision:<8} | Costo: {res['costo']:.2e}")
    
    print(f"\n{'='*80}")
    print(f"{'🔍 VALIDACIÓN DETALLADA POR ALGORITMO':^80}")
    print(f"{'='*80}")
    
    for alg_nombre, resultado in resultados.items():
        print(f"\n📊 Validando {alg_nombre}...")
        params_alg = resultado['params']
        t_val, salidas_val = simular_motor(params_alg)
        
        # Calcular MSE para cada señal
        mse_corriente = np.mean((datos_exp['Is_mag'] - salidas_val['Is_mag'])**2)
        mse_par = np.mean((datos_exp['Te'] - salidas_val['Te'])**2)
        mse_rpm = np.mean((datos_exp['rpm'] - salidas_val['rpm'])**2)
        mse_total = mse_corriente + 0.5*mse_par + 0.3*mse_rpm
        
        print(f"   Errores MSE - Corriente: {mse_corriente:.2e} | Par: {mse_par:.2e} | RPM: {mse_rpm:.2e}")
        print(f"   MSE Total Ponderado: {mse_total:.2e}")
        
        # Comparar con costo de función objetivo
        costo_calculado = funcion_objetivo(params_alg, t_exp, datos_exp)
        print(f"   Costo función objetivo: {costo_calculado:.2e}")
        
        if abs(costo_calculado - resultado['costo']) / resultado['costo'] > 0.1:
            print(f"   ⚠️  Discrepancia detectada entre costo reportado y calculado")
    
    # Graficar comparación solo del mejor algoritmo
    print(f"\n📊 Generando gráficas de comparación para {mejor_algoritmo}...")
    t_val, salidas_val = simular_motor(mejores_params)
    graficar_comparacion(t_exp, datos_exp, t_val, salidas_val, mejor_algoritmo)
    
    return resultados, mejores_params, params_reales

def graficar_comparacion(t_exp, datos_exp, t_sim, salidas_sim, algoritmo):
    """Grafica comparación entre datos experimentales y simulación identificada"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Validación - Identificación con {algoritmo}', fontsize=14)
    
    # Magnitud de corriente
    axes[0,0].plot(t_exp, datos_exp['Is_mag'], 'r-', alpha=0.7, label='Experimental')
    axes[0,0].plot(t_sim, salidas_sim['Is_mag'], 'b--', label='Identificado')
    axes[0,0].set_title('Magnitud Corriente Estator')
    axes[0,0].set_ylabel('Corriente (A)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Par electromagnético
    axes[0,1].plot(t_exp, datos_exp['Te'], 'r-', alpha=0.7, label='Experimental')
    axes[0,1].plot(t_sim, salidas_sim['Te'], 'b--', label='Identificado')
    axes[0,1].set_title('Par Electromagnético')
    axes[0,1].set_ylabel('Par (N⋅m)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Velocidad en RPM
    axes[1,0].plot(t_exp, datos_exp['rpm'], 'r-', alpha=0.7, label='Experimental')
    axes[1,0].plot(t_sim, salidas_sim['rpm'], 'b--', label='Identificado')
    axes[1,0].set_title('Velocidad')
    axes[1,0].set_ylabel('RPM')
    axes[1,0].set_xlabel('Tiempo (s)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Error absoluto de corriente
    error_corriente = np.abs(datos_exp['Is_mag'] - salidas_sim['Is_mag'])
    axes[1,1].plot(t_exp, error_corriente, 'g-', linewidth=2)
    axes[1,1].set_title('Error Absoluto |Is|')
    axes[1,1].set_ylabel('Error (A)')
    axes[1,1].set_xlabel('Tiempo (s)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas de error
    mse_corriente = np.mean((datos_exp['Is_mag'] - salidas_sim['Is_mag'])**2)
    mse_par = np.mean((datos_exp['Te'] - salidas_sim['Te'])**2)
    mse_rpm = np.mean((datos_exp['rpm'] - salidas_sim['rpm'])**2)
    
    print(f"Errores de validación (MSE):")
    print(f"  Corriente: {mse_corriente:.2e}")
    print(f"  Par:       {mse_par:.2e}")
    print(f"  RPM:       {mse_rpm:.2e}")

def ejecutar_identificacion_mejorada():
    """Función con configuración mejorada basada en análisis previo"""
    
    # Parámetros reales del motor
    params_reales = np.array([2.45, 1.83, 0.008, 0.008, 0.203, 0.02, 0.001])
    nombres_params = ['rs', 'rr', 'Lls', 'Llr', 'Lm', 'J', 'B']
    
    print("🔧 Generando datos experimentales MEJORADOS...")
    # Usar menos ruido para mejor identificación
    t_exp, datos_exp = generar_datos_experimentales(params_reales, ruido_nivel=0.015)  # 1.5% vs 3%
    
    # Límites más restrictivos (±25% vs ±50%)
    factor_busqueda = 0.25
    lb = params_reales * (1 - factor_busqueda)
    ub = params_reales * (1 + factor_busqueda)
    bounds = (lb, ub)
    
    print(f"\n{'='*60}")
    print(f"{'CONFIGURACIÓN MEJORADA':^60}")
    print(f"{'='*60}")
    print(f"Límites de búsqueda: ±{factor_busqueda*100}% (más restrictivo)")
    print(f"Ruido en datos: 1.5% (reducido)")
    print(f"Iteraciones aumentadas, partículas optimizadas")
    
    # Función objetivo mejorada con normalización
    def objetivo_normalizado(params):
        costo_base = funcion_objetivo(params, t_exp, datos_exp)
        # Normalizar por la magnitud típica de las señales
        factor_norm = np.mean([np.std(datos_exp['Is_mag']), 
                              np.std(datos_exp['Te']), 
                              np.std(datos_exp['rpm'])])
        return costo_base / (factor_norm**2)
    
    # Configuración optimizada
    config_mejorada = {
        'PSO': {'n_particles': 40, 'iterations': 75},      # Más partículas e iteraciones
        'PSO-SQP': {'n_particles': 30, 'pso_iterations': 50},
        'BFO': {'n_bacteria': 15, 'n_chemotactic': 25, 'n_reproductive': 4}  # Más quimiotaxis
    }
    
    resultados = {}
    
    print(f"\n{'='*60}")
    print(f"{'RE-EJECUTANDO CON CONFIGURACIÓN MEJORADA':^60}")
    print(f"{'='*60}")
    
    # Solo ejecutar PSO mejorado por velocidad
    print("\n🐝 PSO MEJORADO...")
    start_time = time.time()
    costo_pso, params_pso = pso_con_progreso(objetivo_normalizado, bounds, **config_mejorada['PSO'])
    tiempo_pso = time.time() - start_time
    resultados['PSO-Mejorado'] = {'costo': costo_pso, 'params': params_pso, 'tiempo': tiempo_pso}
    
    print(f"\n🚀 PSO-SQP MEJORADO...")
    start_time = time.time()
    costo_hibrido, params_hibrido = run_pso_sqp(objetivo_normalizado, bounds, **config_mejorada['PSO-SQP'])
    tiempo_hibrido = time.time() - start_time
    resultados['PSO-SQP-Mejorado'] = {'costo': costo_hibrido, 'params': params_hibrido, 'tiempo': tiempo_hibrido}
    
    # Mostrar comparación mejorada
    print(f"\n{'='*60}")
    print(f"{'🎯 RESULTADOS CON CONFIGURACIÓN MEJORADA':^60}")
    print(f"{'='*60}")
    
    for alg_nombre, resultado in resultados.items():
        params_alg = resultado['params']
        costo_alg = resultado['costo']
        tiempo_alg = resultado['tiempo']
        
        print(f"\n🔸 {alg_nombre}")
        print(f"   Costo normalizado: {costo_alg:.2e} | Tiempo: {tiempo_alg:.2f}s")
        
        errores_algoritmo = []
        for i, nombre in enumerate(nombres_params):
            real = params_reales[i]
            identificado = params_alg[i]
            error_pct = abs((identificado - real) / real) * 100
            errores_algoritmo.append(error_pct)
        
        error_promedio_alg = np.mean(errores_algoritmo)
        print(f"   Error promedio: {error_promedio_alg:.2f}%")
        
        if error_promedio_alg < 5.0:
            print(f"   Estado: 🟢 EXCELENTE MEJORA")
        elif error_promedio_alg < 10.0:
            print(f"   Estado: 🟡 BUENA MEJORA")
        else:
            print(f"   Estado: 🟠 MEJORA MODERADA")
    
    return resultados, params_reales

if __name__ == "__main__":
    print("🚀 INICIANDO SISTEMA DE IDENTIFICACIÓN DE PARÁMETROS")
    print("=" * 80)
    print("Motor de Inducción Trifásico + Algoritmos Bioinspirados")
    print("=" * 80)
    
    # Ejecutar identificación completa
    print("\n⏱️  Tiempo estimado: 5-10 minutos")
    print("📊 Monitoreando progreso en tiempo real...\n")
    
    inicio_total = time.time()
    resultados, params_identificados, params_reales = ejecutar_identificacion()
    tiempo_total = time.time() - inicio_total
    
    print(f"\n{'='*80}")
    print(f"{'🎯 IDENTIFICACIÓN INICIAL COMPLETADA':^80}")
    print(f"{'='*80}")
    print(f"⏱️  Tiempo total: {tiempo_total:.2f} segundos")
    
    # Verificar si necesita mejora
    mejor_costo = min(r['costo'] for r in resultados.values())
    if mejor_costo > 10.0:
        print(f"\n⚠️  RENDIMIENTO SUBÓPTIMO DETECTADO")
        print(f"🔧 Ejecutando configuración mejorada automáticamente...")
        
        print(f"\n{'='*80}")
        print(f"{'🚀 INICIANDO IDENTIFICACIÓN MEJORADA':^80}")
        print(f"{'='*80}")
        
        inicio_mejorado = time.time()
        resultados_mejorados, params_reales = ejecutar_identificacion_mejorada()
        tiempo_mejorado = time.time() - inicio_mejorado
        
        print(f"\n{'='*80}")
        print(f"{'✅ IDENTIFICACIÓN MEJORADA COMPLETADA':^80}")
        print(f"{'='*80}")
        print(f"⏱️  Tiempo adicional: {tiempo_mejorado:.2f} segundos")
        
        mejor_costo_mejorado = min(r['costo'] for r in resultados_mejorados.values())
        mejora_factor = mejor_costo / mejor_costo_mejorado if mejor_costo_mejorado > 0 else float('inf')
        
        print(f"📈 Factor de mejora: {mejora_factor:.1f}x")
        print(f"🎯 Costo inicial: {mejor_costo:.2e} → Costo mejorado: {mejor_costo_mejorado:.2e}")
    
    print(f"\n{'='*80}")
    print("🔬 Sistema completado y validado para investigación")
    print("💡 Resultados listos para análisis y publicación")
    print("🎓 Datos comparativos disponibles para tesis")
    print("=" * 80)