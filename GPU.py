import tensorflow as tf

# Agrega estas líneas para verificar
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))