#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 00:03:15 2017

@author: antonio
"""

from keras.models import Sequential #paquete para inicializar nuestra red, como es una convolucional usamos sequential
from keras.layers import Conv2D #para las capas de convolucón 
from keras.layers import MaxPooling2D #para el paso dos max pooling
from keras.layers import Flatten #para la parte de flatten
from keras.layers import Dense #

#inicializar la red

classifier = Sequential()

#para la convolucion

classifier.add(Conv2D(32, 3, 3, input_shape=(64,64,3), activation='relu'))

#para el pooling

classifier.add(MaxPooling2D(pool_size = (2,2)))

#para mejorar la calidad de la predicción podemos agregar otra capa de colvolución
#como en la capa anterior ya vienen los datos en el pooling, no agregamos el input_shape
classifier.add(Conv2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#flattening

classifier.add(Flatten())

#capa oculta completamente conectada

classifier.add(Dense(output_dim = 128, activation = 'relu'))

#output layer

classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) 
#si fueran más de dos clases debemos usar softmax

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#categorial cross entropy en caso de que sean más clases

#entrenar nuestra CNN

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

#realizar las predicciones

import numpy as np 
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand:dims(test_image, axis=0)

result = classifier.predict(test_image)

training_set.class_indices
if result[0][0] == 1:
	prediction = "dog"
else :
	prediction = "cat"
	pass




