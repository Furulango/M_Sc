"""
Modelo DCGAN (Deep Convolutional GAN) para generar laberintos
Arquitectura optimizada para laberintos de 32x32 píxeles
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
import numpy as np

class DCGANMazeGenerator:
    def __init__(self, 
                 image_size: int = 32,
                 noise_dim: int = 100,
                 learning_rate: float = 0.0002,
                 beta_1: float = 0.5):
        """
        Inicializa la DCGAN para generar laberintos
        
        Args:
            image_size: Tamaño de las imágenes (32x32)
            noise_dim: Dimensión del vector de ruido
            learning_rate: Tasa de aprendizaje
            beta_1: Parámetro beta1 para Adam optimizer
        """
        self.image_size = image_size
        self.noise_dim = noise_dim
        self.lr = learning_rate
        self.beta_1 = beta_1
        
        # Construir modelos
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Optimizadores
        self.gen_optimizer = Adam(learning_rate=self.lr, beta_1=self.beta_1)
        self.disc_optimizer = Adam(learning_rate=self.lr, beta_1=self.beta_1)
        
        # Función de pérdida
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    def _build_generator(self) -> Model:
        """
        Construye el generador de la DCGAN
        
        Returns:
            generator: Modelo del generador
        """
        model = tf.keras.Sequential([
            # Input: vector de ruido (100,)
            layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(self.noise_dim,)),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            # Reshape a imagen 4x4x512
            layers.Reshape((4, 4, 512)),
            
            # Upsampling 1: 4x4 -> 8x8
            layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), 
                                 padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            # Upsampling 2: 8x8 -> 16x16
            layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), 
                                 padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            # Upsampling 3: 16x16 -> 32x32
            layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), 
                                 padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            # Capa final: output 32x32x1
            layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), 
                                 padding='same', use_bias=False, 
                                 activation='sigmoid')  # Valores entre 0 y 1
        ])
        
        return model
    
    def _build_discriminator(self) -> Model:
        """
        Construye el discriminador de la DCGAN
        
        Returns:
            discriminator: Modelo del discriminador
        """
        model = tf.keras.Sequential([
            # Input: imagen 32x32x1
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            
            # Downsampling 1: 32x32 -> 16x16
            layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            
            # Downsampling 2: 16x16 -> 8x8
            layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            
            # Downsampling 3: 8x8 -> 4x4
            layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            
            # Flatten y clasificación
            layers.Flatten(),
            layers.Dense(1)  # Sin activación (logits)
        ])
        
        return model
    
    def discriminator_loss(self, real_output, fake_output):
        """
        Calcula la pérdida del discriminador
        
        Args:
            real_output: Predicciones para imágenes reales
            fake_output: Predicciones para imágenes falsas
            
        Returns:
            total_loss: Pérdida total del discriminador
        """
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    def generator_loss(self, fake_output):
        """
        Calcula la pérdida del generador
        
        Args:
            fake_output: Predicciones del discriminador para imágenes generadas
            
        Returns:
            loss: Pérdida del generador
        """
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
    
    @tf.function
    def train_step(self, images, batch_size):
        """
        Un paso de entrenamiento de la GAN
        
        Args:
            images: Batch de imágenes reales
            batch_size: Tamaño del batch
            
        Returns:
            gen_loss, disc_loss: Pérdidas del generador y discriminador
        """
        noise = tf.random.normal([batch_size, self.noise_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
        
        gradients_of_generator = gen_tape.gradient(gen_loss, 
                                                  self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, 
                                                       self.discriminator.trainable_variables)
        
        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, 
                                              self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, 
                                               self.discriminator.trainable_variables))
        
        return gen_loss, disc_loss
    
    def generate_mazes(self, num_samples: int = 16, threshold: float = 0.5) -> np.ndarray:
        """
        Genera laberintos usando el generador entrenado
        
        Args:
            num_samples: Número de laberintos a generar
            threshold: Umbral para binarizar (0.5 = punto medio)
            
        Returns:
            mazes: Array con laberintos generados (valores 0 y 1)
        """
        noise = tf.random.normal([num_samples, self.noise_dim])
        generated_images = self.generator(noise, training=False)
        
        # Binarizar: >threshold = 1 (camino), <=threshold = 0 (pared)
        binary_mazes = (generated_images > threshold).numpy().astype(np.uint8)
        
        return binary_mazes
    
    def save_models(self, checkpoint_dir: str):
        """
        Guarda los modelos entrenados
        
        Args:
            checkpoint_dir: Directorio para guardar checkpoints
        """
        self.generator.save_weights(f"{checkpoint_dir}/generator")
        self.discriminator.save_weights(f"{checkpoint_dir}/discriminator")
        print(f"Modelos guardados en: {checkpoint_dir}")
    
    def load_models(self, checkpoint_dir: str):
        """
        Carga modelos previamente entrenados
        
        Args:
            checkpoint_dir: Directorio con los checkpoints
        """
        self.generator.load_weights(f"{checkpoint_dir}/generator")
        self.discriminator.load_weights(f"{checkpoint_dir}/discriminator")
        print(f"Modelos cargados desde: {checkpoint_dir}")
    
    def model_summary(self):
        """
        Muestra el resumen de los modelos
        """
        print("=== GENERADOR ===")
        self.generator.summary()
        print("\n=== DISCRIMINADOR ===")
        self.discriminator.summary()

# Ejemplo de uso
if __name__ == "__main__":
    # Crear modelo
    dcgan = DCGANMazeGenerator(image_size=32, noise_dim=100)
    
    # Mostrar arquitecturas
    dcgan.model_summary()
    
    # Generar algunos laberintos aleatorios (sin entrenar)
    random_mazes = dcgan.generate_mazes(num_samples=4)
    print(f"Laberintos generados: {random_mazes.shape}")
    print(f"Valores únicos: {np.unique(random_mazes)}")
