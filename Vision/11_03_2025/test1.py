import cv2 as cv
import numpy as np

img = cv.imread("C:/Users/gcmed/Downloads/Imagen/cats_set/renamed_images/3.jpg")
roi = cv.selectROI(img)
img_test = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
print("Imagen Original")
print("Alto: " + str(img.shape[0])+ " Ancho: " + str(img.shape[1]) + " Canales: " + str(img.shape[2]))
print("Imagen Recortada")
print("Alto: " + str(img_test.shape[0])+ " Ancho: " + str(img_test.shape[1]) + " Canales: " + str(img_test.shape[2]))
t1 = cv.flip(img_test, 1)
t2 = cv.add(img_test, np.ones(img_test.shape, dtype=np.uint8)*100)
t3 = cv.subtract(img_test, np.ones(img_test.shape, dtype=np.uint8)*100)
t4 = cv.resize(img_test[30:400, 20:400], (img_test.shape[1], img_test.shape[0]), interpolation=cv.INTER_AREA)
aux = cv.getRotationMatrix2D((img_test.shape[1]/2,img_test.shape[0]/2),45,1)
t5 = cv.warpAffine(img_test,aux,(img_test.shape[1],img_test.shape[0]))

resize_width = 200
aspect_ratio = resize_width / img.shape[1]
dimensions = (resize_width, int(img.shape[0] * aspect_ratio))

img_resized = cv.resize(img_test, dimensions, interpolation=cv.INTER_AREA)
t1_resized = cv.resize(t1, dimensions, interpolation=cv.INTER_AREA)
t2_resized = cv.resize(t2, dimensions, interpolation=cv.INTER_AREA)
t3_resized = cv.resize(t3, dimensions, interpolation=cv.INTER_AREA)
t4_resized = cv.resize(t4, dimensions, interpolation=cv.INTER_AREA)
t5_resized = cv.resize(t5, dimensions, interpolation=cv.INTER_AREA)


combined = np.hstack((img_resized, t1_resized, t2_resized, t3_resized, t4_resized, t5_resized))
cv.imshow("Combined Image", combined)
cv.waitKey(0)
cv.destroyAllWindows()
