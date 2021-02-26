import tensorflow as tf
import os
from utils.custom_dataset_loader import Loader

loader = Loader(path=f'{os.getcwd()}/Pictures',
                loading_batch=3200,
                learning_batch=32,
                num_epochs=3)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(6, 3, activation='selu', padding='same', input_shape=(224,224,3)),
    tf.keras.layers.MaxPool2D(16),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])
model.summary()
model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['mae']
    )

while loader.is_going:
    train_dataset = loader.load_next_data()
    model.fit(train_dataset, epochs=1)

