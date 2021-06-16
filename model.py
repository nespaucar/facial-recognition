# Importamos paquetes necesarios

from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import numpy as np

import tensorflow as tf

# Configuración necesaria para optimizar la sesión de la cámara

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)
set_session(session)

# Clase para predecir la emoción

class FacialExpressionModel(object):

    # Todas las posibles salidas, por orden de lectura de carpetas

    # LISTA_EMOCIONES = ["Molesto", "Disgustado", "De miedo", "Feliz", "Neutral", "Triste", "Sorprendido"]

    # En este caso, solo usaremos la tristeza

    LISTA_EMOCIONES = ["", "", "", "", "", "Triste", ""]

    # Inicializamos el modelo con los archivos obtenidos en el entrenamiento

    def __init__(self, model_json_file, model_weights_file):
     
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)

    # Función para predecir    

    def predict_emotion(self, img):
        global session
        set_session(session)

        # En este paso vamos a obtener una matriz con los porcentajes de las predicciones como [0, 0, 0, 0, 1, 0, 0], el 1 representa el valor más alto

        self.preds = self.loaded_model.predict(img)

        # Los valores coinciden con el arreglo de emociones, seleccionamos la más alta y la retornamos, por ejemplo,  [0, 0, 0, 0, 1, 0, 0] => "Neutral"

        return FacialExpressionModel.LISTA_EMOCIONES[np.argmax(self.preds)]
