import cv2 as cv
import numpy as np

def resize_image(image, width):
    aspect_ratio = width / image.shape[1]
    dimensions = (width, int(image.shape[0] * aspect_ratio))
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

img = cv.imread("C:/Users/gcmed/Downloads/Imagen/cats_set/renamed_images/2.jpg")

h, w, c = img.shape
print("Alto: " + str(h)+ " Ancho: " + str(w) + " Canales: " + str(c))

t1 = cv.flip(img, 1)
t2 = cv.add(img, np.ones(img.shape, dtype=np.uint8)*100)
t3 = cv.subtract(img, np.ones(img.shape, dtype=np.uint8)*100)
t4 = cv.resize(img[30:400, 20:400], (img.shape[1], img.shape[0]), interpolation=cv.INTER_AREA)

roi = cv.selectROI(img)
roi = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]


resize_width = 200

img_resized = resize_image(img, resize_width)
t1_resized = resize_image(t1, resize_width)
t2_resized = resize_image(t2, resize_width)
t3_resized = resize_image(t3, resize_width)
t4_resized = resize_image(t4,resize_width)
roi_resized = resize_image(roi,resize_width)

combined = np.hstack((img_resized, t1_resized, t2_resized, t3_resized, t4_resized, roi_resized))
cv.imshow("Combined Image", combined)
cv.waitKey(0)
cv.destroyAllWindows()
