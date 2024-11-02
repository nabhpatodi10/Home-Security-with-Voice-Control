import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, regularizers

training_dir = "Face_Dataset/train/"
testing_dir = "Face_Dataset/test/"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle = True
)

test_generator = test_datagen.flow_from_directory(
    testing_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle = False
)

inputs = layers.Input(shape = (224, 224, 3))

x = EfficientNetB0(include_top=False, weights=None, input_tensor=inputs, pooling='avg')(inputs)
x = layers.Dropout(0.5)(x)
output = layers.Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.001))(x)

model = tf.keras.Model(inputs, output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=5,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)

test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('Test accuracy:', test_acc)

model.save("Model.h5")

with open("training_history.json", "w") as f:
    json.dump(history.history, f)