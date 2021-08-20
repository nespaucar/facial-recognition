# Importación de paquetes necesarios

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from livelossplot import PlotLossesKeras
from tensorflow.keras.utils import plot_model
from IPython.display import SVG, Image
from livelossplot import PlotLossesKerasTF
from tensorflow.keras import backend as K

# Nos aseguramos de matar todas las sesiones de Keras existentes para empezar con una sesión nueva

K.clear_session()

#Ploteando imágenes

# Aquí recorremos las imágenes de entrenamiento y evaluación y nos aseguramos de que tengan un tamaño de 48x48 pixeles

def plot_example_images(plt):
    img_size = 48
    plt.figure(0, figsize=(12,20)) # Para entrenamiento
    plt2.figure(0, figsize=(12,20)) # Para evaluación
    ctr = 0

    for expression in os.listdir("train/"):
        for i in range(1,6):
            ctr += 1
            plt.subplot(7,5,ctr)
            img = load_img("train/" + expression + "/" +os.listdir("train/" + expression)[i], target_size=(img_size, img_size)) # Cargamos la imagen
            plt.imshow(img, cmap="gray") # La ploteamos a escala de grises

    for expression in os.listdir("test/"):
        for i in range(1,6):
            ctr += 1
            plt2.subplot(7,5,ctr)
            img = load_img("test/" + expression + "/" +os.listdir("test/" + expression)[i], target_size=(img_size, img_size)) # Cargamos la imagen
            plt2.imshow(img, cmap="gray") # La ploteamos a escala de grises

    plt.tight_layout()
    plt2.tight_layout()
    return plt

# Imprimiendo carpetas de entrenamiento y validación

for expression in os.listdir("../train/"):
    print(str(len(os.listdir("../train/" + expression))) + " - Emoción de entrenamiento: " + expression)

for expression in os.listdir("../test/"):
    print(str(len(os.listdir("../test/" + expression))) + " - Emoción de validación: " + expression)

# Inicializamos el tamaño de las imágenes a 48X48 y la cantidad de lote de imágenes para procesar

img_size = 48 # Las imágenes se procesarán en un tamaño inicial de 48x48
batch_size = 64 # Las imágenes se procesarán en paquetes de 64
clases = 7 # Cantidad de carpetas de emociones, posibles predicciones finales
lr = 0.0005 # learning_rate, ajustes de la red neuronal para acercarse a la predicción óptima
epochs = 50 # Cantidad de épocas que va a iterar el modelo

# Empezamos a preprocesar las imágenes
# Generando data para entrenamiento

datagen_train = ImageDataGenerator(horizontal_flip = True)

train_generator = datagen_train.flow_from_directory("../train/", # Carpeta de entrnamiento
                                                    target_size = (img_size, img_size), # Tamaño de 48x48
                                                    color_mode = "grayscale", # Escala de grises
                                                    batch_size = batch_size, # Paquetes de 64
                                                    class_mode = "categorical", # Clasificación categórica por 7 emociones
                                                    shuffle = True)

# Generando data para evaluación

datagen_validation = ImageDataGenerator(horizontal_flip = True)

validation_generator = datagen_validation.flow_from_directory("../test/", # Carpeta de evaluación
                                                    target_size = (img_size, img_size), # Tamaño de 48x48
                                                    color_mode = "grayscale", # Escala de grises
                                                    batch_size = batch_size, # Paquetes de 64
                                                    class_mode = "categorical", # Clasificación categórica por 7 emociones
                                                    shuffle = False)

# Creación del modelo de red neuronal convolucional
# Inicializamos en modelo

model = Sequential() # Secuencial porque va en varias capas apiladas, una depués de otra

# Agregamos las capas al modelo

### Etapa de extracción de Características

# Primera capa con una cantidad de 64 kernels (3, 3), e imágenes de entrada de 48x48

model.add(Conv2D(64, (3, 3), padding = 'same', input_shape = (img_size, img_size, 1)))
model.add(BatchNormalization()) # Normalizamos los paquetes
model.add(Activation('relu')) # Función de activación de RELU
model.add(MaxPooling2D(pool_size = (2, 2))) # Hacemos el Max Pooling
model.add(Dropout(0.25)) # apagamos el 25% de las neuronas para evitar carga

# Segunda capa con una cantidad de 128 kernels (5, 5)

model.add(Conv2D(128, (5, 5), padding = 'same'))
model.add(BatchNormalization()) # Normalizamos los paquetes
model.add(Activation('relu')) # Función de activación de RELU
model.add(MaxPooling2D(pool_size = (2, 2))) # Hacemos el Max Pooling
model.add(Dropout(0.25)) # apagamos el 25% de las neuronas para evitar carga

# Tercera capa con una cantidad de 512 kernels (3, 3)

model.add(Conv2D(512, (3, 3), padding = 'same'))
model.add(BatchNormalization()) # Normalizamos los paquetes
model.add(Activation('relu')) # Función de activación de RELU
model.add(MaxPooling2D(pool_size = (2, 2))) # Hacemos el Max Pooling
model.add(Dropout(0.25)) # apagamos el 25% de las neuronas para evitar carga

# Cuarta capa con una cantidad de 512 kernels (3, 3)

model.add(Conv2D(512, (3, 3), padding = 'same'))
model.add(BatchNormalization()) # Normalizamos los paquetes
model.add(Activation('relu')) # Función de activación de RELU
model.add(MaxPooling2D(pool_size = (2, 2))) # Hacemos el Max Pooling
model.add(Dropout(0.25)) # apagamos el 25% de las neuronas para evitar carga

# En esta capa hacemos una trancisión a la etapa de Clasificación
# Aquí aplanamos las imágenes muy profundas y pequeñas las aplanamos, una dimensión con toda la información

model.add(Flatten())

# Primera capa de densidad con 256 neuronas

model.add(Dense(256))
model.add(BatchNormalization()) # Normalizamos los paquetes
model.add(Activation('relu')) # Función de activación de RELU
model.add(Dropout(0.25)) # Bajamos el 25% de las neuronas para aprender caminos alternos, para adaptarse mejor a información nueva con diferentes emociones

# Segunda capa de densidad con 512 neuronas

model.add(Dense(512))
model.add(BatchNormalization()) # Normalizamos los paquetes
model.add(Activation('relu')) # Función de activación de RELU
model.add(Dropout(0.25)) # Bajamos nuevamente el 25% de las neuronas

# Última capa de densidad con 7 neuronas, las que servirán de predicción final de la emoción

model.add(Dense(clases, activation = 'softmax')) # Aplicamos la función de activación de SOFTMAX para clasificar

# Finalmente compilamos el modelo y ya estaría listo para entrenar el DataSet

opt = Adam(lr = lr)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Imprimimos la estructura del modelo

model.summary()

## Librerías necesarias para imprimir gráfico de cada época

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 2, min_lr = 0.00001, mode = 'auto')

# Guardamos los pesos obtenidos en cada época

checkpoint = ModelCheckpoint("../pesos_modelo.h5", monitor = 'val_accuracy', save_weights_only = True, mode = 'max', verbose = 1)

# Esto para imprimir la gráfica de pérdida y ganancia

callbacks = [PlotLossesKerasTF(), checkpoint, reduce_lr]

## Etapa de Entrenamiento y Evaluación del modelo

history = model.fit(
    batch_size = batch_size,
    x = train_generator,
    epochs = epochs,
    validation_data = validation_generator,
    callbacks = callbacks
)

# Representación de modelo entrenado en JSON, es la estructura del modelo, lo guardamos en model.json para usarlo luego

model_json = model.to_json()
with open("../modelo.json", "w") as json_file:
    json_file.write(model_json)