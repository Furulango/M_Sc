import numpy as np
import cv2 as cv
import seaborn as se
import sklearn as sk
import matplotlib.pyplot as plt

import os

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

#-------------------------#
k = 5
tr = []
lb = []

path_original = list_files("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/18_03_2025/Img/Originales") #Originales
path_augmented =list_files("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/18_03_2025/Img/Group1") #Aumentadas
path_autoencoded = list_files("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/18_03_2025/Img/Group2") #Autoencoder

#print(len(path_original), len(path_augmented), len(path_autoencoded))

for img_path in path_original:
    im = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    if im is None:
        continue
    im = cv.resize(im, (100, 100))  # Redimensionar la imagen correcta
    im1 = np.array(im).flatten()
    tr.append(im1)
    lab = os.path.split(os.path.split(img_path)[0])[1]  # Usar la ruta original
    lb.append(lab)


tr = np.array(tr)
lb = np.array(lb)
tr_train, tr_test, lb_train, lb_test = sk.model_selection.train_test_split(tr, lb, test_size=0.3)
knn = sk.neighbors.KNeighborsClassifier(n_neighbors=k)
knn.fit(tr_train, lb_train)
pred = knn.predict(tr_test)

# Matriz de confusion
acc = sk.metrics.accuracy_score(lb_test, pred)
conf_matrix = sk.metrics.confusion_matrix(lb_test, pred)
# Mostrar la matriz de confusión como un mapa de calor

plt.figure(figsize=(8, 6))
se.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(lb), 
           yticklabels=np.unique(lb),annot_kws={"size": 16})
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix (Accuracy: {acc:.2f})')
plt.show()
