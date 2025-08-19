# 🔬 Sistema Modular de Análisis
## Gemelo Digital Adaptativo - Algoritmos Bio-Inspirados

Sistema completo de análisis para la evaluación de algoritmos bio-inspirados en la identificación de parámetros DQ de motores de inducción con enfoque de gemelo digital adaptativo.

## 📋 **Características Principales**

### **Enfoque de 2 Fases:**
- **Fase 1:** Calibración completa con multi-señal (operación normal)
- **Fase 2:** Adaptación de campo con solo corriente (alta temperatura y condiciones severas)

### **Algoritmos Analizados:**
- **BFO** (Bacterial Foraging Optimization)
- **PSO** (Particle Swarm Optimization)  
- **Chaotic PSO-DSO** (Chaotic Particle Swarm with Dynamic Social Learning)

### **Parámetros DQ del Motor:**
- `rs` - Resistencia del estator
- `rr` - Resistencia del rotor
- `Lls` - Inductancia de dispersión del estator
- `Llr` - Inductancia de dispersión del rotor
- `Lm` - Inductancia magnetizante
- `J` - Inercia
- `B` - Amortiguamiento

---

## 🚀 **Instalación y Requisitos**

### **Dependencias:**
```bash
pip install pandas numpy matplotlib seaborn scipy
```

### **Estructura de Archivos:**
```
proyecto/
├── adaptive_analysis_system.py    # Sistema principal
├── individual_modules.py          # Ejecutor individual
├── README.md                      # Este archivo
├── BFO_adaptive_results.csv       # Datos BFO
├── PSO_adaptive_results.csv       # Datos PSO
└── Chaotic_PSODSO_adaptive_results.csv  # Datos Chaotic PSO-DSO
```

---

## 📊 **Módulos de Análisis**

### **🔥 Módulo 1: Análisis de Adaptabilidad**
- **Objetivo:** Comparar rendimiento Fase 1 vs Fase 2
- **Métricas:** Degradación de error, mejora de tiempo, score de adaptabilidad
- **Salida:** Ranking de adaptabilidad, análisis por parámetro

### **📊 Módulo 2: Heatmap de Parámetros DQ**
- **Objetivo:** Error por parámetro × Algoritmo × Escenario
- **Métricas:** Dificultad de identificación, especialización algorítmica
- **Salida:** Heatmap visual, ranking de dificultad

### **⚡ Módulo 3: Dashboard Comparativo**
- **Objetivo:** Análisis integral de precisión + tiempo + robustez
- **Métricas:** Score general, trade-offs, robustez
- **Salida:** Ranking general, visualizaciones comparativas

### **📈 Módulo 4: Análisis Estadístico Robusto**
- **Objetivo:** ANOVA + Tests post-hoc + Intervalos de confianza
- **Métricas:** Significancia estadística, tamaño del efecto
- **Salida:** Tests de normalidad, comparaciones pairwise

### **🎯 Módulo 5: Análisis de Convergencia**
- **Objetivo:** Eficiencia temporal y estabilidad
- **Métricas:** Velocidad de convergencia, estabilidad, costo-efectividad
- **Salida:** Análisis de eficiencia, métricas de estabilidad

---

## 🔧 **Uso del Sistema**

### **1. Análisis Completo**
```bash
# Ejecutar todos los módulos
python adaptive_analysis_system.py
```

### **2. Módulos Individuales**
```bash
# Módulo específico
python individual_modules.py --module 1

# Múltiples módulos
python individual_modules.py --module 1 2 3

# Todos los módulos
python individual_modules.py --all

# Análisis rápido (módulos principales)
python individual_modules.py --quick

# Análisis estadístico
python individual_modules.py --statistical
```

### **3. Opciones Avanzadas**
```bash
# Especificar ruta de datos
python individual_modules.py --module 1 --data_path ./data/

# Cambiar directorio de salida
python individual_modules.py --all --output ./resultados/

# Salida detallada
python individual_modules.py --module 1 --verbose
```

---

## 📁 **Estructura de Resultados**

```
results_analysis/
├── plots/
│   ├── parameter_heatmap.png           # Heatmap de parámetros
│   ├── dashboard_comparative.png       # Dashboard comparativo
│   └── convergence_analysis.png        # Análisis de convergencia
├── tables/
│   └── [tablas CSV generadas]
├── module_1_adaptability_results.json  # Resultados Módulo 1
├── module_2_heatmap_results.json       # Resultados Módulo 2
├── module_3_dashboard_results.json     # Resultados Módulo 3
├── module_4_statistical_results.json   # Resultados Módulo 4
├── module_5_convergence_results.json   # Resultados Módulo 5
└── executive_summary.json              # Resumen ejecutivo
```

---

## 📊 **Interpretación de Resultados**

### **Métricas Clave:**

#### **Score de Adaptabilidad:**
- **> 80:** Excelente adaptabilidad
- **60-80:** Buena adaptabilidad  
- **40-60:** Adaptabilidad moderada
- **< 40:** Adaptabilidad pobre

#### **Dificultad de Parámetros:**
- **> 25%:** Muy alto (crítico)
- **15-25%:** Alto (problemático)
- **10-15%:** Medio (monitoreable)
- **< 10%:** Bajo (estable)

#### **Success Rate:**
- **> 80%:** Excelente performance
- **60-80%:** Buena performance
- **40-60%:** Performance moderada
- **< 40%:** Performance pobre

### **Interpretación Estadística:**
- **p < 0.05:** Diferencia significativa
- **Cohen's d > 0.8:** Efecto grande
- **CV < 0.2:** Baja variabilidad (buena robustez)

---

## 🎯 **Casos de Uso Específicos**

### **Para Investigación Académica:**
```bash
# Análisis estadístico completo
python individual_modules.py --statistical --verbose

# Verificar significancia
python individual_modules.py --module 4
```

### **Para Implementación Industrial:**
```bash
# Análisis rápido para decisión
python individual_modules.py --quick

# Foco en adaptabilidad
python individual_modules.py --module 1 3
```

### **Para Optimización de Algoritmos:**
```bash
# Análisis de convergencia detallado
python individual_modules.py --module 5

# Análisis de parámetros problemáticos
python individual_modules.py --module 2
```

---

## 🔍 **Ejemplos de Insights Típicos**

### **Adaptabilidad:**
```
🏆 MEJOR adaptabilidad: BFO (Score: 47.75)
📉 Degradación promedio: 124.5% (multi-señal → solo corriente)
⚡ Mejora en tiempo: 19.2% (adaptación más eficiente)
```

### **Parámetros Críticos:**
```
🔴 Parámetro MÁS difícil: B (23.1% error)
🟢 Parámetro MÁS fácil: Lm (4.1% error)
👁️ Mayor pérdida observabilidad: B (320.7% degradación)
```

### **Rendimiento General:**
```
🏆 MEJOR algoritmo general: Chaotic_PSODSO (Score: 54.33)
🛡️ MÁS robusto: BFO (Robustez: 1.65)
⚡ MÁS rápido: Chaotic_PSODSO (503.0s promedio)
```

---

## 🚨 **Solución de Problemas**

### **Error: Archivos CSV no encontrados**
```bash
# Verificar que los archivos están en el directorio correcto
ls *.csv

# Especificar ruta correcta
python individual_modules.py --module 1 --data_path ./ruta/correcta/
```

### **Error: Dependencias faltantes**
```bash
# Instalar dependencias
pip install pandas numpy matplotlib seaborn scipy

# Verificar versiones
python -c "import pandas; print(pandas.__version__)"
```

### **Error: Permisos de escritura**
```bash
# Cambiar directorio de salida
python individual_modules.py --all --output ./mi_directorio/

# Verificar permisos
chmod 755 ./results_analysis/
```

---

## 📈 **Recomendaciones de Uso**

### **Para Papers/Conferencias:**
1. **Ejecutar análisis completo** con `--all`
2. **Generar todas las visualizaciones** para figuras
3. **Revisar resumen ejecutivo** para conclusions
4. **Usar análisis estadístico** para rigor científico

### **Para Implementación Industrial:**
1. **Ejecutar análisis rápido** para decisión inicial
2. **Foco en módulo 1** para entender adaptabilidad
3. **Revisar módulo 2** para identificar parámetros críticos
4. **Usar módulo 3** para selección final de algoritmo

### **Para Debugging de Algoritmos:**
1. **Módulo 5** para analizar convergencia
2. **Módulo 2** para identificar parámetros problemáticos
3. **Módulo 4** para validación estadística

---

## 🤝 **Soporte y Contribuciones**

### **Formato de Datos CSV Requerido:**
```csv
scenario,phase,run,cost,error,time,identified_rs,identified_rr,...,error_rs,error_rr,...
Normal_Operation,1,1,0.000243,11.40,684.20,2.408,2.062,...,6.81,2.07,...
```

### **Columnas Obligatorias:**
- `scenario`: Normal_Operation, High_Temperature, Severe_Conditions
- `phase`: 1 (Calibración), 2 (Adaptación)
- `run`: Número de ejecución
- `cost`: Función de costo
- `error`: Error promedio de parámetros (%)
- `time`: Tiempo de optimización (s)
- `identified_*`: Parámetros identificados
- `true_*`: Parámetros verdaderos
- `error_*`: Error por parámetro (%)

---

## 📚 **Referencias y Citas**

Si usas este sistema en tu investigación, por favor cita:

```bibtex
@inproceedings{adaptive_digital_twin_2024,
  title={Adaptive Digital Twin System: Bio-Inspired Algorithms for Real-Time Parameter Adaptation},
  author={[Tu Nombre]},
  booktitle={Mechatronics, Control \& AI Conference},
  year={2024}
}
```

---

## 📝 **Changelog**

### **v1.0.0 - 2024**
- ✅ Implementación inicial de 5 módulos
- ✅ Sistema modular completo
- ✅ Ejecutor individual
- ✅ Visualizaciones automáticas
- ✅ Análisis estadístico robusto
- ✅ Documentación completa

---

## 📞 **Contacto**

Para preguntas, sugerencias o reportar bugs:
- **Email:** [tu-email@universidad.edu]
- **GitHub:** [tu-github-username]
- **ORCID:** [tu-orcid-id]

---

*🔬 Sistema desarrollado para el análisis de algoritmos bio-inspirados en gemelos digitales adaptativos para motores de inducción.*