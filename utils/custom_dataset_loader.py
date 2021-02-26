from .data_base import Calories, init_base
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils import shuffle

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

class Loader():
    def __init__(self, path, batch, num_epochs, validation_split=0.0, table=Calories, column='calories'):
        self.db_session = init_base()
        self.images_list = []
        self.labels_list = []
        self.current_image_index = 0
        self.current_loop = 1
        self.current_walkthrough = 1
        self.path = path
        self.batch = batch
        self.num_epochs = num_epochs
        self.validation_split = validation_split
        self.table = table
        self.column = column
        self.is_going = True
        self.dataset_size = 0
        for dir in os.listdir(self.path):
            dir_path = os.path.join(self.path, dir)
            for img in os.listdir(dir_path):
                self.dataset_size += 1


    def load_next_data(self):
        self.images_list = []
        self.labels_list = []
        temp_image_index = 0
        if self.current_image_index == 0:
            print(f'------------ walkthrough {self.current_walkthrough} --------------------')
        print(f'epoch {self.current_loop}')
        for dir in os.listdir(self.path):
            if self.current_image_index >= self.current_loop * self.batch:
                break
            dir_path = os.path.join(self.path, dir)
            for img in os.listdir(dir_path):
                if self.current_image_index >= self.current_loop*self.batch:
                    break
                if temp_image_index > self.current_image_index:
                    img_path = os.path.join(dir_path, img)
                    img_from_file = Image.open(img_path)
                    image_array = tf.keras.preprocessing.image.img_to_array(img_from_file)
                    self.images_list.append(image_array)
                    img_from_file.close()

                    image_name = img_path.split('/')[-1]
                    image_id = image_name.split('e')[-1].split('.')[0]

                    row = self.db_session.query(self.table).filter_by(id=int(image_id)).first()
                    if self.column == 'calories':
                        self.labels_list.append(row.calories)
                    #if self.column == 'other' ....
                    self.current_image_index += 1
                temp_image_index += 1
        if self.current_image_index == self.dataset_size-1:
            self.current_image_index = 0
            self.current_loop = 1
            if self.current_walkthrough == self.num_epochs:
                self.is_going = False
            else:
                self.current_walkthrough += 1
        else:
            self.current_loop += 1

        self.images_list = np.array(self.images_list)
        self.labels_list = np.array(self.labels_list)
        self.images_list, self.labels_list = shuffle(self.images_list, self.labels_list)

        if self.validation_split == 0.0:
            return self.images_list, self.labels_list
        else:
            return self.images_list[:int(len(self.images_list) * (1 - self.validation_split))],\
                    self.labels_list[:int(len(self.images_list) * (1 - self.validation_split))], \
                    self.images_list[int(len(self.images_list) * (1 - self.validation_split)):], \
                    self.labels_list[int(len(self.images_list) * (1 - self.validation_split)):],