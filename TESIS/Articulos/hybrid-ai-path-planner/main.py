"""
GENERADOR DE LABERINTOS CON DCGAN - Punto de entrada principal
Proyecto de Machine Learning para generar laberintos solucionables usando Deep Learning

Autor: [Tu nombre]
Fecha: 2025
Objetivo: Lograr 75% de laberintos solucionables generados por IA
"""

import os
import argparse
import tensorflow as tf
from datetime import datetime

# Importar nuestros módulos
from data_generator import MazeDataGenerator
from gan_model import DCGANMazeGenerator
from validator import MazeValidator
from visualizer import MazeVisualizer
from trainer import MazeGANTrainer

def setup_gpu():
    """
    Configura la GPU para entrenamiento óptimo
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU configurada correctamente: {len(gpus)} dispositivo(s)")
            return True
        except RuntimeError as e:
            print(f"❌ Error configurando GPU: {e}")
            return False
    else:
        print("⚠️  No se detectaron GPUs. Usando CPU.")
        return False

def main():
    """
    Función principal del programa
    """
    print("🏰 GENERADOR DE LABERINTOS CON DCGAN")
    print("="*50)
    print("🎯 Objetivo: Generar laberintos solucionables con IA")
    print("📊 Meta: 75% de tasa de solvabilidad")
    print("="*50)
    
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Generador de Laberintos con DCGAN")
    parser.add_argument("--mode", type=str, default="train", 
                       choices=["train", "generate", "validate", "demo"],
                       help="Modo de ejecución")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Número de épocas de entrenamiento")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Tamaño del batch")
    parser.add_argument("--dataset_size", type=int, default=10000,
                       help="Tamaño del dataset de entrenamiento")
    parser.add_argument("--learning_rate", type=float, default=0.0002,
                       help="Tasa de aprendizaje")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Directorio para checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Directorio para outputs")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                       help="Cargar modelo desde checkpoint")
    parser.add_argument("--num_generate", type=int, default=16,
                       help="Número de laberintos a generar")
    
    args = parser.parse_args()
    
    # Configurar GPU
    gpu_available = setup_gpu()
    
    # Crear timestamp para esta ejecución
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = f"{args.output_dir}/run_{timestamp}"
    
    if args.mode == "train":
        print(f"\n🚀 MODO ENTRENAMIENTO")
        print(f"📅 Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Crear entrenador
        trainer = MazeGANTrainer(
            image_size=32,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=run_output_dir
        )
        
        # Cargar checkpoint si se especifica
        if args.load_checkpoint:
            trainer.resume_training(args.load_checkpoint)
        
        # Preparar dataset
        print(f"\n📦 Preparando dataset de {args.dataset_size} laberintos...")
        dataset_path = f"training_data/mazes_{args.dataset_size}.npy"
        dataset = trainer.prepare_dataset(
            num_samples=args.dataset_size,
            dataset_path=dataset_path if os.path.exists(dataset_path) else None
        )
        
        # Entrenar modelo
        print(f"\n🏋️  Iniciando entrenamiento...")
        trainer.train(
            dataset=dataset,
            epochs=args.epochs,
            validation_freq=max(1, args.epochs // 10),  # Validar 10 veces durante el entrenamiento
            checkpoint_freq=max(5, args.epochs // 5),   # Guardar 5 checkpoints
            target_solvability=0.75
        )
        
        print(f"\n✅ Entrenamiento completado!")
        print(f"📁 Resultados en: {run_output_dir}")
    
    elif args.mode == "generate":
        print(f"\n🎨 MODO GENERACIÓN")
        
        if not args.load_checkpoint:
            print("❌ Error: Se requiere --load_checkpoint para generar laberintos")
            return
        
        # Cargar modelo entrenado
        dcgan = DCGANMazeGenerator(image_size=32)
        dcgan.load_models(args.load_checkpoint)
        
        # Crear visualizador
        visualizer = MazeVisualizer(save_dir=run_output_dir)
        validator = MazeValidator()
        
        # Generar laberintos
        print(f"🎲 Generando {args.num_generate} laberintos...")
        generated_mazes = dcgan.generate_mazes(num_samples=args.num_generate)
        
        # Validar laberintos generados
        print("🔍 Validando laberintos generados...")
        validation_results = validator.validate_batch(generated_mazes)
        
        # Crear visualizaciones
        solvability_list = []
        for i in range(len(generated_mazes)):
            maze_2d = generated_mazes[i, :, :, 0]
            is_solvable = validator.validate_maze(maze_2d)['is_solvable']
            solvability_list.append(is_solvable)
        
        # Guardar resultados
        output_path = f"{run_output_dir}/generated_mazes.png"
        visualizer.plot_maze_grid(
            generated_mazes,
            solvability=solvability_list,
            save_path=output_path,
            grid_size=(4, 4)
        )
        
        stats_path = f"{run_output_dir}/generation_stats.png"
        visualizer.plot_validation_stats(validation_results, save_path=stats_path)
        
        print(f"✅ Generación completada!")
        print(f"📊 Tasa de solvabilidad: {validation_results['solvability_rate']:.2%}")
        print(f"📁 Resultados en: {run_output_dir}")
    
    elif args.mode == "validate":
        print(f"\n🔬 MODO VALIDACIÓN")
        
        if not args.load_checkpoint:
            print("❌ Error: Se requiere --load_checkpoint para validar")
            return
        
        # Cargar modelo y validar exhaustivamente
        dcgan = DCGANMazeGenerator(image_size=32)
        dcgan.load_models(args.load_checkpoint)
        validator = MazeValidator()
        visualizer = MazeVisualizer(save_dir=run_output_dir)
        
        print("🧪 Ejecutando validación exhaustiva con 1000 laberintos...")
        test_mazes = dcgan.generate_mazes(num_samples=1000)
        validation_results = validator.validate_batch(test_mazes, verbose=True)
        
        # Crear reportes detallados
        stats_path = f"{run_output_dir}/detailed_validation.png"
        visualizer.plot_validation_stats(validation_results, save_path=stats_path)
        
        # Mostrar mejores y peores ejemplos
        best_mazes = []
        worst_mazes = []
        
        for i, maze in enumerate(test_mazes[:50]):  # Revisar primeros 50
            maze_2d = maze[:, :, 0]
            result = validator.validate_maze(maze_2d)
            
            if result['is_solvable'] and len(best_mazes) < 8:
                best_mazes.append((maze, result))
            elif not result['is_solvable'] and len(worst_mazes) < 8:
                worst_mazes.append((maze, result))
        
        # Guardar ejemplos
        if best_mazes:
            best_sample = np.array([m[0] for m in best_mazes])
            best_path = f"{run_output_dir}/best_examples.png"
            visualizer.plot_maze_grid(best_sample, save_path=best_path, grid_size=(2, 4))
        
        if worst_mazes:
            worst_sample = np.array([m[0] for m in worst_mazes])
            worst_path = f"{run_output_dir}/worst_examples.png"
            visualizer.plot_maze_grid(worst_sample, save_path=worst_path, grid_size=(2, 4))
        
        print(f"✅ Validación completada!")
        print(f"📊 Resultados detallados en: {run_output_dir}")
    
    elif args.mode == "demo":
        print(f"\n🎪 MODO DEMOSTRACIÓN")
        
        # Crear generador de datos para demostrar algoritmo DFS
        print("🔨 Generando laberintos con algoritmo DFS tradicional...")
        data_gen = MazeDataGenerator(size=32)
        validator = MazeValidator()
        visualizer = MazeVisualizer(save_dir=run_output_dir)
        
        # Generar algunos laberintos DFS
        dfs_mazes = []
        dfs_results = []
        
        for i in range(8):
            maze = data_gen.generate_maze_dfs()
            result = validator.validate_maze(maze)
            dfs_mazes.append(maze[:, :, np.newaxis])
            dfs_results.append(result['is_solvable'])
        
        dfs_mazes = np.array(dfs_mazes)
        
        # Visualizar
        dfs_path = f"{run_output_dir}/dfs_examples.png"
        visualizer.plot_maze_grid(
            dfs_mazes,
            solvability=dfs_results,
            save_path=dfs_path,
            grid_size=(2, 4)
        )
        
        # Mostrar estadísticas DFS
        print(f"📊 Laberintos DFS generados: {len(dfs_mazes)}")
        print(f"📊 Tasa de solvabilidad DFS: {np.mean(dfs_results):.2%}")
        
        print(f"✅ Demostración completada!")
        print(f"📁 Ejemplos en: {run_output_dir}")
    
    print(f"\n🎉 Programa ejecutado exitosamente!")
    print(f"⏰ Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n⚠️  Ejecución interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
