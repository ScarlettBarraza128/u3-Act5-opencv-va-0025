import cv2
import numpy as np

# 1. Leer y mostrar una imagen
image = cv2.imread('Fresas.jpg')

img_mostrar = image.copy()
cv2.putText(img_mostrar, 'Emily Barraza NC 0025', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv2.imshow('Imagen Original - Emily Barraza NC 0025', img_mostrar)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Conversión de espacio de color
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_mostrar = gray_image.copy()
cv2.putText(gray_mostrar, 'Emily Barraza NC 0025', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.imshow('Imagen en Escala de Grises - Emily Barraza NC 0025', gray_mostrar)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. Redimensionamiento de imágenes
resized_image = cv2.resize(image, (300, 300))

resized_mostrar = resized_image.copy()
cv2.putText(resized_mostrar, 'Emily Barraza NC 0025', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
cv2.imshow('Imagen Redimensionada - Emily Barraza NC 0025', resized_mostrar)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4. Desenfoque y suavizado
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

blurred_mostrar = blurred_image.copy()
cv2.putText(blurred_mostrar, 'Emily Barraza NC 0025', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv2.imshow('Imagen Desenfocada - Emily Barraza NC 0025', blurred_mostrar)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5. Detección de bordes
edges = cv2.Canny(gray_image, 100, 200)

edges_mostrar = edges.copy()
cv2.putText(edges_mostrar, 'Emily Barraza NC 0025', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.imshow('Bordes Canny - Emily Barraza NC 0025', edges_mostrar)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 6. Umbralización
_, thresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

thresh_mostrar = thresh_image.copy()
cv2.putText(thresh_mostrar, 'Emily Barraza NC 0025', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.imshow('Imagen Umbralizada - Emily Barraza NC 0025', thresh_mostrar)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 7. Contornos
contours, _ = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

cv2.putText(contour_image, 'Emily Barraza NC 0025', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv2.imshow('Contornos - Emily Barraza NC 0025', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 8. Dibujar un rectángulo y agregar texto
annotated_image = image.copy()
cv2.rectangle(annotated_image, (50, 50), (250, 250), (255, 0, 0), 2)
cv2.putText(annotated_image, 'OpenCV Text', (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

cv2.putText(annotated_image, 'Emily Barraza NC 0025', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
cv2.imshow('Imagen Anotada - Emily Barraza NC 0025', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
