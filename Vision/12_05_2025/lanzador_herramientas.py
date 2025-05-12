import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

class LaunchPad:
    def __init__(self, root):
        self.root = root
        self.root.title("Lanzador de Herramientas de Clasificación")
        self.root.geometry("600x400")
        self.root.configure(bg="#f0f0f0")
        
        # Título principal
        titulo = tk.Label(root, text="Sistema de Clasificación de Fallas en Refrigeración", 
                         font=("Arial", 16, "bold"), bg="#f0f0f0", pady=15)
        titulo.pack(fill="x")
        
        # Crear frame principal
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # Crear etiqueta de instrucciones
        instrucciones = ttk.Label(main_frame, 
                                text="Seleccione la herramienta que desea ejecutar:", 
                                font=("Arial", 12))
        instrucciones.pack(pady=10)
        
        # Crear botones para las diferentes herramientas
        self.create_button(main_frame, "Ejecutar Aplicación de Clasificación", 
                          self.ejecutar_app_clasificacion)
        
        self.create_button(main_frame, "Ejecutar Predicción por Consola", 
                          self.ejecutar_prediccion_consola)
        
        self.create_button(main_frame, "Abrir Carpeta de Imágenes", 
                          self.abrir_carpeta_imagenes)
        
        self.create_button(main_frame, "Ver Carpeta de Generables", 
                          self.abrir_carpeta_generables)
        
        # Botón de salida
        exit_btn = ttk.Button(main_frame, text="Salir", command=root.destroy)
        exit_btn.pack(pady=10)
        
        # Verificar existencia de carpetas y archivos necesarios
        self.verificar_estructura()
    
    def create_button(self, parent, text, command):
        """Crea un botón estilizado"""
        btn = ttk.Button(parent, text=text, command=command)
        btn.pack(fill="x", pady=5)
        return btn
    
    def verificar_estructura(self):
        """Verifica que existan las carpetas y archivos necesarios"""
        # Verificar carpeta principal
        if not os.path.exists("Generables"):
            messagebox.showwarning("Advertencia", 
                                  "No se encontró la carpeta 'Generables'. Se creará la estructura necesaria.")
            self.crear_estructura()
        
        # Verificar subcarpetas
        subcarpetas = ["scripts", "modelos", "imagenes"]
        for subcarpeta in subcarpetas:
            ruta = os.path.join("Generables", subcarpeta)
            if not os.path.exists(ruta):
                os.makedirs(ruta, exist_ok=True)
                print(f"Se ha creado la carpeta: {ruta}")
    
    def crear_estructura(self):
        """Crea la estructura básica de carpetas"""
        os.makedirs("Generables", exist_ok=True)
        os.makedirs("Generables/scripts", exist_ok=True)
        os.makedirs("Generables/modelos", exist_ok=True)
        os.makedirs("Generables/imagenes", exist_ok=True)
        messagebox.showinfo("Información", "Se ha creado la estructura de carpetas básica.")
    
    def ejecutar_app_clasificacion(self):
        """Ejecuta la aplicación de clasificación"""
        app_path = os.path.join("Generables", "scripts", "app_clasificador_fallas.py")
        
        if not os.path.exists(app_path):
            messagebox.showerror("Error", f"No se encontró la aplicación en {app_path}")
            return
        
        try:
            subprocess.Popen([sys.executable, app_path])
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo ejecutar la aplicación: {str(e)}")
    
    def ejecutar_prediccion_consola(self):
        """Ejecuta el script de predicción por consola"""
        script_path = os.path.join("Generables", "scripts", "predecir_falla.py")
        
        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"No se encontró el script en {script_path}")
            return
        
        try:
            subprocess.Popen([sys.executable, script_path])
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo ejecutar el script: {str(e)}")
    
    def abrir_carpeta_imagenes(self):
        """Abre la carpeta de imágenes generadas"""
        path = os.path.join("Generables", "imagenes")
        self.abrir_carpeta(path)
    
    def abrir_carpeta_generables(self):
        """Abre la carpeta principal de generables"""
        self.abrir_carpeta("Generables")
    
    def abrir_carpeta(self, ruta):
        """Abre una carpeta en el explorador de archivos"""
        if not os.path.exists(ruta):
            messagebox.showerror("Error", f"No se encontró la carpeta: {ruta}")
            return
        
        try:
            # Intenta abrir la carpeta según el sistema operativo
            if sys.platform == 'win32':
                os.startfile(ruta)
            elif sys.platform == 'darwin':  # macOS
                subprocess.call(['open', ruta])
            else:  # Linux
                subprocess.call(['xdg-open', ruta])
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la carpeta: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LaunchPad(root)
    root.mainloop()