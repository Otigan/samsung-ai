from .data_base import Calories, init_base
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils import shuffle

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

db_session = init_base()

images_list = []
calories_list = []

def load_data(path):
    for dir in os.listdir(path):
        dir_path = os.path.join(path, dir)
        for img in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img)
            img_from_file = Image.open(img_path)
            image_array = tf.keras.preprocessing.image.img_to_array(img_from_file)
            images_list.append(image_array)
            img_from_file.close()

            image_name = img_path.split('/')[-1]
            image_id = image_name.split('e')[-1].split('.')[0]

            calories = db_session.query(Calories).filter_by(id=int(image_id)).first()
            calories_list.append(calories.calories)

load_data('C:/Project/Python/pyCharmTest2/Pictures')

images_list = np.array(images_list)
calories_list = np.array(calories_list)

print(images_list.shape)
print(calories_list.shape)

image_generator = ImageDataGenerator(rescale=1/255)
train_dataset = image_generator.flow(x=images_list, y=calories_list, batch_size=32)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(6, 3, activation='selu', padding='same', input_shape=(224,224,3)),
    tf.keras.layers.Conv2D(6, 3, activation='selu', padding='same'),
    tf.keras.layers.Conv2D(6, 3, activation='selu', padding='same'),
    tf.keras.layers.MaxPool2D(2),

    tf.keras.layers.Conv2D(12, 3, activation='selu', padding='same'),
    tf.keras.layers.Conv2D(12, 3, activation='selu', padding='same'),
    tf.keras.layers.MaxPool2D(2),

    tf.keras.layers.Conv2D(18, 3, activation='selu', padding='same'),
    tf.keras.layers.Conv2D(18, 3, activation='selu', padding='same'),
    tf.keras.layers.MaxPool2D(2),

    tf.keras.layers.Conv2D(24, 3, activation='selu', padding='same'),
    tf.keras.layers.Conv2D(24, 3, activation='selu', padding='same'),
    tf.keras.layers.MaxPool2D(2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])
model.summary()
model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['mae']
    )

model.fit(train_dataset, epochs=60)

predicted = model.predict(tf.data.Dataset.from_tensor_slices([images_list[4]]).batch(1))
print(f'predicted={predicted}   true={calories_list[4]}')

