from pyexpat import model
from re import A
import numpy as np
import cv2 as cv
import tensorflow as tf
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
import os 
import seaborn as sb
import matplotlib.pyplot as plt
import keras as ke
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def list_files(directory):
    lista = os.listdir(directory)
    files = list()
    for i in lista:
        path = os.path.join(directory, i)
        if os.path.isdir(path):
            files = files + list_files(path)
        else:
            files.append(path)
    return files

path_original = list_files("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/18_03_2025/Img/Originales")
path_augmented = list_files("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/18_03_2025/Img/Group1")
path_autoencoded = list_files("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/18_03_2025/Img/Group2")

tr = []
lb = []

for img_path in path_augmented:
    im = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    if im is None:
        continue
    im = cv.resize(im, (100, 100))
    tr.append(im)
    lab = os.path.split(os.path.split(img_path)[0])[1]
    lb.append(lab)

imgs = np.array(tr)
lb = np.array(lb)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(lb)
num_classes = len(label_encoder.classes_)
print(f"Classes found: {label_encoder.classes_}")
print(f"Number of classes: {num_classes}")

imgs = imgs/255.0

imgs = np.expand_dims(imgs, axis=-1)

tr_train, tr_test, lb_train, lb_test = sk.model_selection.train_test_split(imgs, encoded_labels, test_size=0.2, random_state=42)
tr_train, tr_val, lb_train, lb_val = sk.model_selection.train_test_split(tr_train, lb_train, test_size=0.2, random_state=42)

lb_train_onehot = ke.utils.to_categorical(lb_train, num_classes=num_classes)
lb_test_onehot = ke.utils.to_categorical(lb_test, num_classes=num_classes)
lb_val_onehot = ke.utils.to_categorical(lb_val, num_classes=num_classes)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=ke.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(tr_train, lb_train_onehot, epochs=10, batch_size=32, 
                    validation_data=(tr_val, lb_val_onehot))

test_loss, test_acc = model.evaluate(tr_test, lb_test_onehot)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# Plotting the training history
plt.figure(figsize=(18, 6))

# Accuracy
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Confusion Matrix
plt.subplot(1, 3, 3)
predicted_labels = model.predict(tr_test)
predicted_classes = np.argmax(predicted_labels, axis=1)
cm = confusion_matrix(lb_test, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
plt.title('Confusion Matrix')

plt.tight_layout()
plt.show()


