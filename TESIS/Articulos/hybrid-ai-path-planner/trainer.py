"""
Entrenador para la DCGAN de laberintos con monitoreo completo
Coordina el entrenamiento, validación y visualización
"""

import tensorflow as tf
import numpy as np
import time
import os
from typing import Tuple, Dict, Optional

# Importar nuestros módulos
from data_generator import MazeDataGenerator
from gan_model import DCGANMazeGenerator
from validator import MazeValidator
from visualizer import MazeVisualizer

class MazeGANTrainer:
    def __init__(self, 
                 image_size: int = 32,
                 batch_size: int = 32,
                 learning_rate: float = 0.0002,
                 noise_dim: int = 100,
                 checkpoint_dir: str = "checkpoints",
                 output_dir: str = "training_outputs"):
        """
        Inicializa el entrenador de la DCGAN
        
        Args:
            image_size: Tamaño de las imágenes de laberintos
            batch_size: Tamaño del batch para entrenamiento
            learning_rate: Tasa de aprendizaje
            noise_dim: Dimensión del vector de ruido
            checkpoint_dir: Directorio para checkpoints
            output_dir: Directorio para outputs de entrenamiento
        """
        self.image_size = image_size
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        
        # Crear directorios
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Inicializar componentes
        self.data_generator = MazeDataGenerator(size=image_size)
        self.dcgan = DCGANMazeGenerator(
            image_size=image_size,
            noise_dim=noise_dim,
            learning_rate=learning_rate
        )
        self.validator = MazeValidator()
        self.visualizer = MazeVisualizer(save_dir=output_dir)
        
        # Métricas de entrenamiento
        self.training_metrics = {
            'generator_losses': [],
            'discriminator_losses': [],
            'solvability_rates': [],
            'epochs': [],
            'best_solvability': 0.0,
            'best_epoch': 0
        }
        
        print("🏗️  Entrenador inicializado correctamente!")
        print(f"📁 Checkpoints: {checkpoint_dir}")
        print(f"📊 Outputs: {output_dir}")
    
    def prepare_dataset(self, 
                       num_samples: int = 10000,
                       dataset_path: Optional[str] = None) -> tf.data.Dataset:
        """
        Prepara el dataset de entrenamiento
        
        Args:
            num_samples: Número de laberintos a generar
            dataset_path: Ruta de dataset existente (opcional)
            
        Returns:
            dataset: Dataset de TensorFlow preparado
        """
        print("📦 Preparando dataset...")
        
        if dataset_path and os.path.exists(dataset_path):
            print(f"📂 Cargando dataset existente: {dataset_path}")
            maze_data = self.data_generator.load_dataset(dataset_path)
        else:
            print(f"🔨 Generando {num_samples} laberintos nuevos...")
            save_path = f"{self.output_dir}/training_mazes_{num_samples}.npy"
            maze_data = self.data_generator.generate_dataset(
                num_mazes=num_samples,
                save_path=save_path
            )
        
        # Crear dataset de TensorFlow
        dataset = tf.data.Dataset.from_tensor_slices(maze_data)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        print(f"✅ Dataset preparado: {maze_data.shape}")
        return dataset
    
    def validate_generated_mazes(self, num_samples: int = 100) -> Dict:
        """
        Valida laberintos generados por el modelo actual
        
        Args:
            num_samples: Número de laberintos a validar
            
        Returns:
            dict: Resultados de validación
        """
        # Generar laberintos
        generated_mazes = self.dcgan.generate_mazes(num_samples=num_samples)
        
        # Validar
        validation_results = self.validator.validate_batch(
            generated_mazes, verbose=False
        )
        
        return validation_results
    
    def train_step_with_metrics(self, batch: tf.Tensor) -> Tuple[float, float]:
        """
        Ejecuta un paso de entrenamiento y retorna métricas
        
        Args:
            batch: Batch de imágenes reales
            
        Returns:
            gen_loss, disc_loss: Pérdidas del generador y discriminador
        """
        gen_loss, disc_loss = self.dcgan.train_step(batch, self.batch_size)
        return float(gen_loss), float(disc_loss)
    
    def train(self, 
              dataset: tf.data.Dataset,
              epochs: int = 100,
              validation_freq: int = 10,
              checkpoint_freq: int = 20,
              target_solvability: float = 0.75) -> None:
        """
        Entrena la DCGAN con monitoreo completo
        
        Args:
            dataset: Dataset de entrenamiento
            epochs: Número de épocas
            validation_freq: Frecuencia de validación (cada N épocas)
            checkpoint_freq: Frecuencia de checkpoints
            target_solvability: Objetivo de solvabilidad (75%)
        """
        print(f"🚀 Iniciando entrenamiento por {epochs} épocas...")
        print(f"🎯 Objetivo: {target_solvability:.0%} de solvabilidad")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Entrenar una época
            epoch_gen_losses = []
            epoch_disc_losses = []
            
            for batch in dataset:
                gen_loss, disc_loss = self.train_step_with_metrics(batch)
                epoch_gen_losses.append(gen_loss)
                epoch_disc_losses.append(disc_loss)
            
            # Promediar pérdidas de la época
            avg_gen_loss = np.mean(epoch_gen_losses)
            avg_disc_loss = np.mean(epoch_disc_losses)
            
            # Validación periódica
            solvability_rate = 0.0
            if epoch % validation_freq == 0 or epoch == epochs - 1:
                print(f"\n🔍 Validando en época {epoch}...")
                validation_results = self.validate_generated_mazes(num_samples=200)
                solvability_rate = validation_results['solvability_rate']
                
                # Actualizar mejor modelo
                if solvability_rate > self.training_metrics['best_solvability']:
                    self.training_metrics['best_solvability'] = solvability_rate
                    self.training_metrics['best_epoch'] = epoch
                    
                    # Guardar mejor modelo
                    best_model_dir = f"{self.checkpoint_dir}/best_model"
                    self.dcgan.save_models(best_model_dir)
                    print(f"💾 Nuevo mejor modelo guardado (solvabilidad: {solvability_rate:.2%})")
            
            # Actualizar métricas
            self.training_metrics['epochs'].append(epoch)
            self.training_metrics['generator_losses'].append(avg_gen_loss)
            self.training_metrics['discriminator_losses'].append(avg_disc_loss)
            self.training_metrics['solvability_rates'].append(solvability_rate)
            self.visualizer.update_metrics(epoch, avg_gen_loss, avg_disc_loss, solvability_rate)
            
            # Mostrar progreso
            epoch_time = time.time() - epoch_start
            print(f"Época {epoch:3d}/{epochs} | "
                  f"Gen: {avg_gen_loss:.4f} | "
                  f"Disc: {avg_disc_loss:.4f} | "
                  f"Solv: {solvability_rate:.2%} | "
                  f"Tiempo: {epoch_time:.1f}s")
            
            # Visualización periódica
            if epoch % validation_freq == 0:
                self._create_epoch_visualizations(epoch, validation_results)
            
            # Checkpoint periódico
            if epoch % checkpoint_freq == 0 and epoch > 0:
                checkpoint_path = f"{self.checkpoint_dir}/epoch_{epoch}"
                self.dcgan.save_models(checkpoint_path)
                print(f"💾 Checkpoint guardado en época {epoch}")
            
            # Verificar si se alcanzó el objetivo
            if solvability_rate >= target_solvability:
                print(f"\n🎉 ¡OBJETIVO ALCANZADO! Solvabilidad: {solvability_rate:.2%}")
                print(f"🏆 Meta de {target_solvability:.0%} superada en época {epoch}")
                break
        
        # Finalizar entrenamiento
        total_time = (time.time() - start_time) / 60
        print(f"\n✅ Entrenamiento completado en {total_time:.2f} minutos")
        
        # Guardar modelo final
        final_model_dir = f"{self.checkpoint_dir}/final_model"
        self.dcgan.save_models(final_model_dir)
        
        # Crear resumen final
        self._create_final_summary(total_time)
    
    def _create_epoch_visualizations(self, epoch: int, validation_results: Dict) -> None:
        """
        Crea visualizaciones para una época específica
        
        Args:
            epoch: Número de época
            validation_results: Resultados de validación
        """
        # Generar laberintos para visualizar
        sample_mazes = self.dcgan.generate_mazes(num_samples=16)
        
        # Validar los muestras para visualización
        sample_solvability = []
        for i in range(len(sample_mazes)):
            maze_2d = sample_mazes[i, :, :, 0]
            is_solvable = self.validator.validate_maze(maze_2d)['is_solvable']
            sample_solvability.append(is_solvable)
        
        # Guardar grilla de laberintos
        maze_grid_path = f"{self.output_dir}/epoch_{epoch:03d}_mazes.png"
        self.visualizer.plot_maze_grid(
            sample_mazes,
            solvability=sample_solvability,
            save_path=maze_grid_path,
            grid_size=(4, 4)
        )
        
        # Guardar métricas actuales
        metrics_path = f"{self.output_dir}/epoch_{epoch:03d}_metrics.png"
        self.visualizer.plot_training_metrics(
            self.training_metrics['generator_losses'],
            self.training_metrics['discriminator_losses'],
            self.training_metrics['solvability_rates'],
            self.training_metrics['epochs'],
            save_path=metrics_path
        )
        
        print(f"📊 Visualizaciones guardadas para época {epoch}")
    
    def _create_final_summary(self, training_time: float) -> None:
        """
        Crea el resumen final del entrenamiento
        
        Args:
            training_time: Tiempo total de entrenamiento en minutos
        """
        final_solvability = self.training_metrics['solvability_rates'][-1]
        total_epochs = len(self.training_metrics['epochs'])
        best_epoch = self.training_metrics['best_epoch']
        
        # Crear resumen con visualizador
        self.visualizer.save_training_summary(
            final_solvability=final_solvability,
            total_epochs=total_epochs,
            best_epoch=best_epoch,
            training_time=training_time
        )
        
        # Validación final detallada
        print("\n🔬 Ejecutando validación final detallada...")
        final_validation = self.validate_generated_mazes(num_samples=1000)
        
        # Visualizar estadísticas finales
        final_stats_path = f"{self.output_dir}/final_validation_stats.png"
        self.visualizer.plot_validation_stats(
            final_validation,
            save_path=final_stats_path
        )
        
        # Generar laberintos finales de muestra
        final_samples = self.dcgan.generate_mazes(num_samples=16)
        final_solvability = []
        for i in range(len(final_samples)):
            maze_2d = final_samples[i, :, :, 0]
            is_solvable = self.validator.validate_maze(maze_2d)['is_solvable']
            final_solvability.append(is_solvable)
        
        final_samples_path = f"{self.output_dir}/final_sample_mazes.png"
        self.visualizer.plot_maze_grid(
            final_samples,
            solvability=final_solvability,
            save_path=final_samples_path,
            grid_size=(4, 4)
        )
        
        print(f"📈 Resumen final guardado")
        print(f"🎯 Solvabilidad final: {final_solvability:.2%}")
        print(f"🏆 Mejor solvabilidad: {self.training_metrics['best_solvability']:.2%} (época {best_epoch})")
    
    def resume_training(self, checkpoint_path: str) -> None:
        """
        Reanuda el entrenamiento desde un checkpoint
        
        Args:
            checkpoint_path: Ruta del checkpoint a cargar
        """
        print(f"🔄 Reanudando entrenamiento desde: {checkpoint_path}")
        self.dcgan.load_models(checkpoint_path)
        print("✅ Modelos cargados correctamente")

# Ejemplo de uso
if __name__ == "__main__":
    # Verificar GPU
    print("🔧 Verificando configuración...")
    print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Crear entrenador
    trainer = MazeGANTrainer(
        image_size=32,
        batch_size=32,
        learning_rate=0.0002,
        noise_dim=100
    )
    
    # Preparar dataset
    dataset = trainer.prepare_dataset(num_samples=5000)  # Dataset pequeño para prueba
    
    # Entrenar
    trainer.train(
        dataset=dataset,
        epochs=50,  # Pocas épocas para prueba
        validation_freq=5,
        checkpoint_freq=10,
        target_solvability=0.75
    )
    
    print("🎉 Entrenamiento de prueba completado!")
