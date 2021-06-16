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

# Aquí recorremos las imágenes de entrenamiento y evaluación y los redimensionamos a escala de 48x48 pixeles

def plot_example_images(plt):
    img_size = 48
    plt.figure(0, figsize=(12,20))
    plt2.figure(0, figsize=(12,20))
    ctr = 0

    for expression in os.listdir("train/"):
        for i in range(1,6):
            ctr += 1
            plt.subplot(7,5,ctr)
            img = load_img("train/" + expression + "/" +os.listdir("train/" + expression)[i], target_size=(img_size, img_size))
            plt.imshow(img, cmap="gray")

    for expression in os.listdir("test/"):
        for i in range(1,6):
            ctr += 1
            plt2.subplot(7,5,ctr)
            img = load_img("test/" + expression + "/" +os.listdir("test/" + expression)[i], target_size=(img_size, img_size))
            plt2.imshow(img, cmap="gray")

    plt.tight_layout()
    plt2.tight_layout()
    return plt

# Imprimiendo carpetas de entrenamiento y validación

for expression in os.listdir("../train/"):
    print(str(len(os.listdir("../train/" + expression))) + " - Emoción de entrenamiento: " + expression)

for expression in os.listdir("../test/"):
    print(str(len(os.listdir("../test/" + expression))) + " - Emoción de validación: " + expression)

# Inicializamos el tamaño de las imágenes a 48X48 y la cantidad de lote de imágenes para procesar

img_size = 48
batch_size = 64
clases = 7 # Cantidad de carpetas de emociones, posibles predicciones
lr = 0.0005 # learning_rate, ajustes de la red neuronal para acercarse a la predicción óptima
epochs = 50 # Cantidad de épocas que va a iterar el modelo

# Empezamos a preprocesar las imágenes

# Generando data para entrenamiento

datagen_train = ImageDataGenerator(horizontal_flip = True)

train_generator = datagen_train.flow_from_directory("../train/",
                                                    target_size = (img_size, img_size),
                                                    color_mode = "grayscale",
                                                    batch_size = batch_size,
                                                    class_mode = "categorical", # Clasificación categórica por 7 emociones
                                                    shuffle = True)

# Generando data para evaluación

datagen_validation = ImageDataGenerator(horizontal_flip = True)

validation_generator = datagen_validation.flow_from_directory("../test/",
                                                    target_size = (img_size, img_size),
                                                    color_mode = "grayscale",
                                                    batch_size = batch_size,
                                                    class_mode = "categorical", # Clasificación categórica por 7 emociones
                                                    shuffle = False)

# Creación del modelo de red neuronal convolucional

# Inicializamos en modelo

model = Sequential() # Secuencial porque va en varias capas apiladas, una depués de otra

# Agregamos las capas al modelo

model.add(Conv2D(64, (3, 3), padding = 'same', input_shape = (img_size, img_size, 1))) #
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2))) #
model.add(Dropout(0.25))

model.add(Conv2D(128, (5, 5), padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) # Aquí aplanamos las imágenes muy profundas y pequeñas las aplanamos, una dimensión con toda la información

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Bajamos el 25% de las neuronas para aprender caminos alternos, para adaptarse mejor a información nueva con diferentes emociones

model.add(Dropout(0.25))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Bajamos nuevamente el 25% de las neuronas

model.add(Dropout(0.25))

# Esta capa sirve para predecir entre las clases o emociones en el modelo. La probabilidad más alta es la predicción correcta

model.add(Dense(clases, activation = 'softmax'))

opt = Adam(lr = lr)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Imprimimos las estadísticas del modelo

model.summary()

# Entrenando y evaluando el modelo

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 2, min_lr = 0.00001, mode = 'auto')

# Guardamos los pesos obtenidos en cada época

checkpoint = ModelCheckpoint("../pesos_modelo.h5", monitor = 'val_accuracy', save_weights_only = True, mode = 'max', verbose = 1)

# Esto para imprimir la gráfica de pérdida y ganancia

callbacks = [PlotLossesKerasTF(), checkpoint, reduce_lr]

history = model.fit(
    batch_size = batch_size,
    x = train_generator,
    epochs = epochs,
    validation_data = validation_generator,
    callbacks = callbacks
)

# Representación de modelo entrenado en JSON, es la estructura del modelo, lo guardamos en model.json

model_json = model.to_json()
with open("../modelo.json", "w") as json_file:
    json_file.write(model_json)