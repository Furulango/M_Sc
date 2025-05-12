
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import os
import pandas as pd
import seaborn as sns

class AppClasificadorFallas:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Fallas en Sistemas de Refrigeración")
        self.root.geometry("900x680")
        self.root.configure(bg="#f0f0f0")
        
        # Ruta relativa a la carpeta de modelos
        carpeta_actual = os.path.dirname(os.path.abspath(__file__))
        carpeta_base = os.path.dirname(os.path.dirname(carpeta_actual))
        modelo_path = os.path.join(carpeta_base, 'Generables', 'modelos', 'random_forest_model.pkl')
        
        if not os.path.exists(modelo_path):
            messagebox.showerror("Error", f"Modelo no encontrado en: {modelo_path}")
            self.root.destroy()
            return
            
        self.modelo = joblib.load(modelo_path)
        self.mapeo_inverso = {0: 'FALLA ALTA', 1: 'FALLA BAJA', 2: 'SANO'}
        
        # Título principal
        titulo = tk.Label(root, text="Sistema de Clasificación de Fallas", 
                         font=("Arial", 18, "bold"), bg="#f0f0f0", pady=10)
        titulo.pack(fill="x")
        
        # Frame para entradas
        frame_entrada = ttk.LabelFrame(root, text="Datos de Gradientes Térmicos")
        frame_entrada.pack(fill="both", expand=False, padx=20, pady=10)
        
        # Crear widgets para entrada de datos
        self.crear_widgets_entrada(frame_entrada)
        
        # Frame para resultados
        self.frame_resultados = ttk.LabelFrame(root, text="Resultados de la Clasificación")
        self.frame_resultados.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Inicializar frame de gráficos
        self.frame_grafico = ttk.Frame(self.frame_resultados)
        self.frame_grafico.pack(side=tk.TOP, fill="both", expand=True, padx=10, pady=10)
        
        # Texto inicial en el área de resultados
        self.resultado_text = tk.Label(self.frame_resultados, 
                                      text="Los resultados aparecerán aquí",
                                      font=("Arial", 12), pady=10)
        self.resultado_text.pack()
        
        # Inicializar figura vacía
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_grafico)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Botones de acción
        frame_botones = ttk.Frame(root)
        frame_botones.pack(fill="x", padx=20, pady=10)
        
        # Botón para predecir
        btn_predecir = ttk.Button(frame_botones, text="Predecir", command=self.predecir)
        btn_predecir.pack(side=tk.LEFT, padx=10)
        
        # Botón para limpiar
        btn_limpiar = ttk.Button(frame_botones, text="Limpiar", command=self.limpiar)
        btn_limpiar.pack(side=tk.LEFT, padx=10)
        
        # Botón para salir
        btn_salir = ttk.Button(frame_botones, text="Salir", command=root.destroy)
        btn_salir.pack(side=tk.RIGHT, padx=10)
    
    def crear_widgets_entrada(self, frame):
        # Crear 3 entradas para los valores de gradiente
        labels = ["Válvula de Expansión:", "Compresor:", "Condensador:"]
        self.entradas = []
        
        for i, label in enumerate(labels):
            # Frame para cada entrada
            frame_entrada = ttk.Frame(frame)
            frame_entrada.pack(fill="x", padx=10, pady=5)
            
            # Etiqueta
            lbl = ttk.Label(frame_entrada, text=label, width=20)
            lbl.pack(side=tk.LEFT, padx=5)
            
            # Campo de entrada
            var = tk.DoubleVar(value=0.0)
            entrada = ttk.Entry(frame_entrada, textvariable=var, width=10)
            entrada.pack(side=tk.LEFT, padx=5)
            
            # Slider
            slider = ttk.Scale(frame_entrada, from_=-5.0, to=5.0, 
                              orient="horizontal", variable=var, 
                              length=400)
            slider.pack(side=tk.LEFT, fill="x", expand=True, padx=10)
            
            self.entradas.append(var)
    
    def predecir(self):
        try:
            # Obtener valores de las entradas
            valvula = self.entradas[0].get()
            compresor = self.entradas[1].get()
            condensador = self.entradas[2].get()
            
            # Preparar datos para predicción
            X = np.array([[valvula, compresor, condensador]])
            
            # Realizar predicción
            y_pred = self.modelo.predict(X)[0]
            y_probs = self.modelo.predict_proba(X)[0]
            
            # Obtener clase predicha
            clase = self.mapeo_inverso[y_pred]
            
            # Actualizar texto de resultado
            resultado = f"Clasificación: {clase}

"
            resultado += "Probabilidades:
"
            
            for i, prob in enumerate(y_probs):
                resultado += f"{self.mapeo_inverso[i]}: {prob:.4f}
"
            
            self.resultado_text.config(text=resultado)
            
            # Crear gráfico de barras para las probabilidades
            self.ax.clear()
            clases = [self.mapeo_inverso[i] for i in range(len(y_probs))]
            colors = ['red' if clase == 'FALLA ALTA' else 'orange' if clase == 'FALLA BAJA' else 'green' 
                    for clase in clases]
            
            bars = self.ax.bar(clases, y_probs, color=colors)
            self.ax.set_ylim(0, 1)
            self.ax.set_ylabel('Probabilidad')
            self.ax.set_title('Probabilidades de Predicción')
            
            # Añadir etiquetas con los valores
            for bar in bars:
                height = bar.get_height()
                self.ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.4f}', ha='center', va='bottom')
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en la predicción: {str(e)}")
    
    def limpiar(self):
        # Restaurar valores predeterminados
        for var in self.entradas:
            var.set(0.0)
        
        # Limpiar resultados
        self.resultado_text.config(text="Los resultados aparecerán aquí")
        
        # Limpiar gráfico
        self.ax.clear()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = AppClasificadorFallas(root)
    root.mainloop()
