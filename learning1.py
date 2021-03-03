import tensorflow as tf
import os
from utils.custom_dataset_loader import Loader

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(6, 3, activation='relu', padding='same', input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(6, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(6, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(2, ),
        tf.keras.layers.BatchNormalization(axis=1),

        tf.keras.layers.Conv2D(6, 3, activation='relu', padding='same', input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(6, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(6, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.BatchNormalization(axis=1),

        tf.keras.layers.Conv2D(12, 3, activation='selu', padding='same'),
        tf.keras.layers.BatchNormalization(axis=1),
        tf.keras.layers.MaxPool2D(2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredLogarithmicError(),
        metrics=['mae']
    )
    return model


loader = Loader(path=f'{os.getcwd()}/Pictures',
                batch=1000,
                validation_split=0.2,
                num_epochs=20,
                column='calories')
model = create_model()
while loader.is_going:
    x, y , val_x, val_y= loader.load_next_data()
    tf.keras.backend.clear_session()
    model.fit([x],[y], validation_data=(val_x, val_y), batch_size=64, epochs=1) #should be 1 epoch here. Real number of epochs defines by Loader parameters
    weights = model.get_weights()
    model = create_model()
    model.set_weights(weights) # preventing memory leak,  creating new model with old weights



