import numpy as np
import cv2 as cv
import seaborn as se
import sklearn as sk
import matplotlib.pyplot as plt
import os

#Stadistical features
# Contrast, Entropy,RMS,Energy,Mean

def stadistical_features(img):
    """
    Calculate stadistical features of an image
    Args:
        img (np.array): Image in grayscale
    Returns:
        list: List of stadistical features
    """
    contrast = cv.Laplacian(img, cv.CV_64F).var()
    entropy = cv.calcHist([img], [0], None, [256], [0, 256])
    entropy = -np.sum(entropy*np.log2(entropy + 1e-6))
    rms = np.sqrt(np.mean(img**2))
    energy = np.sum(img**2)
    mean = np.mean(img)
    return [contrast, entropy, rms, energy, mean]


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
k_values = [3, 5, 7]
tr = []
lb = []

path_original = list_files("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/18_03_2025/Img/Originales") #Originales
path_augmented = list_files("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/18_03_2025/Img/Group1") #Aumentadas
path_autoencoded = list_files("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/18_03_2025/Img/Group2") #Autoencoder

for path, label in [(path_original, "Original"), (path_augmented, "Augmented")]:
    print(f"Processing {label} images...")
    tr = []
    lb = []
    for img_path in path:
        im = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if im is None:
            continue
        im = cv.resize(im, (100, 100))  # Resize the image
        im1 = np.array(im).flatten()
        tr.append(im1)
        lab = os.path.split(os.path.split(img_path)[0])[1]  # Extract label from path
        lb.append(lab)

    tr = np.array(tr)
    lb = np.array(lb)
    tr_train, tr_test, lb_train, lb_test = sk.model_selection.train_test_split(tr, lb, test_size=0.3)

    for degree in [3, 5, 7]:
        svm = sk.svm.SVC(kernel='poly', degree=degree)
        svm.fit(tr_train, lb_train)
        pred = svm.predict(tr_test)
        acc = sk.metrics.accuracy_score(lb_test, pred)
        pre = sk.metrics.precision_score(lb_test, pred, average='weighted')
        print(f'{label} - Degree {degree}: Accuracy: {acc:.2f}, Precision: {pre:.2f}')

