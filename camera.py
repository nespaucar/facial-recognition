# Importamos paquetes necesarios

import cv2 # Librería OpenCV para abrir la cámara del dispositivo donde está corriendo el código
from model import FacialExpressionModel # Improtamos el archivo model.py para predecir la emoción en la cara de la imagen
import numpy as np

# Cargamos el haarcascade haarcascade_frontalface_default que nos ayudará a detectar los rostros de la cámara del dispositivo

facec = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_default.xml')

# Cargamos el archivo para predecir que generamos en el entrenamiento del modelo

model = FacialExpressionModel("modelo.json", "pesos_modelo.h5")

# Cargamos el tipo de font de la cámara FONT_HERSHEY_SIMPLEX

font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0) # Abrimos la cámara principal del sistema, si queremos otros, podemos probar con 1 y -1

    def __del__(self):
        self.video.release() # Terminar la sesión de cámara

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read() # Leemos la cámara
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) # Valor necesario para leer la cámara
        faces = facec.detectMultiScale(gray_fr, 1.3, 5) # Para mejorar la detección de rostro

        # Algoritmo para abrir la cámara y retornarlo en archivo .jpeg

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis]) # Línea para predecir la emoción haciendo uso del modelo

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2) # Agregamos el texto al marco
            cv2.rectangle(fr,(x, y), (x + w, y + h), (255, 0, 0), 2) # Para enmarcar el rostro en el vídeo

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes() # Finalmente mostramos la cámara con las configuraciones de enmarcado de rostro
