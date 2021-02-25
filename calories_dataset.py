import tensorflow as tf
from PIL import Image
from PIL import ImageFile

from database.data_base import Calories, init_base

ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

db_session = init_base()

x_train = []
y_train = []
x_test = []
y_test = []


def load_data(path):
    test = 0
    for dir in os.listdir(path):
        dir_path = os.path.join(path, dir)
        for img in os.listdir(dir_path):

            img_path = os.path.join(dir_path, img)
            img_from_file = Image.open(img_path)
            image_array = tf.keras.preprocessing.image.img_to_array(img_from_file)
            image_name = img_path.split('/')[-1]
            image_id = image_name.split('e')[-1].split('.')[0]

            calories = db_session.query(Calories).filter_by(id=int(image_id)).first()
            if test <= 2900:
                x_test.append(image_array)
                y_test.append(calories.calories)
            else:
                x_train.append(image_array)
                y_train.append(calories.calories)
            img_from_file.close()
            test += 1


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

load_data('D:/samsung/dataset/Pictures')

x_train = x_train[:500]
y_train = y_train[:500]
x_test = x_test[:200]
y_test = y_test[:200]

x_train[:] = [i/255 for i in x_train]
x_test[:] = [i/255 for i in x_test]

print(x_test[:3])

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(2)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(2)

BATCH_SIZE = 1
SHUFFLE_BUFFER_SIZE = 10

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).batch(2)
test_dataset = test_dataset.batch(BATCH_SIZE).batch(2)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(3, 3, activation='relu', padding='same', input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(3, 3, activation='relu', padding='same'),
    tf.keras.layers.Conv2D(3, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.BatchNormalization(axis=1),

    tf.keras.layers.Conv2D(3, 3, activation='relu', padding='same', input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(3, 3, activation='relu', padding='same'),
    tf.keras.layers.Conv2D(3, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.BatchNormalization(axis=1),

    tf.keras.layers.Conv2D(12, 3, activation='selu', padding='same'),
    tf.keras.layers.BatchNormalization(axis=1),
    tf.keras.layers.MaxPool2D(2),



    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='linear')
])
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['mae']
)

model.fit(train_dataset.batch(2), validation_data=test_dataset.batch(2), epochs=60)
