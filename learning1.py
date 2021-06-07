import tensorflow as tf
import os
from utils.custom_dataset_loader import Loader

def create_model():
    model = tf.keras.Sequential([
        tf.keras.applications.ResNet50(input_shape=(224, 224, 3), include_top=False),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['mae']
    )
    return model



loader = Loader(path=f'{os.getcwd()}/Pictures',
                batch=1600,
                validation_split=0.2,
                num_epochs=60,
                column='calories',
                stop_number=6400)

model = create_model()
while loader.is_going:
    x, y , val_x, val_y= loader.load_next_data()
    print(x.shape, y.shape, val_x.shape, val_y.shape)
    tf.keras.backend.clear_session()
    model.fit([x],[y], validation_data=(val_x, val_y), batch_size=64, epochs=1) #should be 1 epoch here. Real number of epochs defines by Loader parameters
    weights = model.get_weights()
    model = create_model()
    model.set_weights(weights) # preventing memory leak,  creating new model with old weights
model.save('model')


