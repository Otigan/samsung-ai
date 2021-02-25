from database.data_base import Calories, init_base
import tensorflow as tf
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

print(len(images_list))
print(len(calories_list))