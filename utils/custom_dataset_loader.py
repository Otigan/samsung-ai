from .data_base import FoodImage, Recipe, init_base
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils import shuffle

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import os


class Loader():
    def __init__(self, path, batch, num_epochs, validation_split=0.0, column='calories'):
        self.db_session = init_base()
        self.images_list = []  # Листы для хранения загруженного куска датасета
        self.labels_list = []  #
        self.current_image_index = 0  # Номер рисунка, загружаемого в данный момент
        self.current_loop = 1  # Текущий номер батча загрузки
        self.current_walkthrough = 1  # Текущий проход по датасету (эпоха)
        self.path = path
        self.batch = batch  # Количество рисунков, загружаемых за раз
        self.num_epochs = num_epochs  # Количество эпох (проходов по датасету)
        self.validation_split = validation_split
        self.column = column  # Колонка базы данных, загружаемая в лейбл лист
        self.is_going = True  # Нужное количество эпох (проходов по датасету) не прошло
        self.dataset_size = 0
        for dir in os.listdir(self.path):
            dir_path = os.path.join(self.path, dir)
            for img in os.listdir(dir_path):
                self.dataset_size += 1  # Цикл для определения размера датасета

    def load_next_data(self):
        self.images_list = []
        self.labels_list = []
        temp_image_index = 0  # индекс, обнуляемы при каждом цикле загрузки,
        # для сравнения с последним загруженным элементом
        if self.current_image_index == 0:  # Проход по датасету начался сначала
            print(f'------------ walkthrough {self.current_walkthrough} --------------------')
        print(f'epoch {self.current_loop}')
        for dir in os.listdir(self.path):
            dir_path = os.path.join(self.path, dir)
            if self.current_image_index >= self.current_loop * self.batch: \
                    # Было загружено нужное количество картинок для текущего цикла загрузки
                break
            for img in os.listdir(dir_path):
                if temp_image_index >= self.current_image_index:  # Если дошли до последнего загруженного элемента
                    img_path = os.path.join(dir_path, img)  # Далее обработка изображения
                    img_from_file = Image.open(img_path)
                    image_array = tf.keras.preprocessing.image.img_to_array(img_from_file)
                    image_array = image_array / 255
                    image_array = image_array[:, :, :3] # Убираем лишние каналы
                    self.images_list.append(image_array)
                    img_from_file.close()

                    image_name = img_path.split('/')[-1]
                    image_id = image_name.split('e')[-1].split('.')[0]

                    foreign_key = self.db_session.query(FoodImage).filter_by(id=int(image_id)).first().recipe_id
                    # Выбор соответствующего элемена из базы данных
                    recipe = self.db_session.query(Recipe).filter_by(id=foreign_key).first()
                    if self.column == 'calories':  # Выбор из базы данных, в зависисимости от заданной колонки
                        self.labels_list.append(recipe.calories)
                    elif self.column == 'total_time':
                        self.labels_list.append(recipe.total_time)
                    elif self.column == 'fat_content':
                        self.labels_list.append(recipe.fat_content)
                    elif self.column == 'saturated_fat_content':
                        self.labels_list.append(recipe.saturated_fat_content)
                    elif self.column == 'cholesterol_content':
                        self.labels_list.append(recipe.cholesterol_content)
                    elif self.column == 'sodium_content':
                        self.labels_list.append(recipe.sodium_content)
                    elif self.column == 'carbohydrate_content':
                        self.labels_list.append(recipe.carbohydrate_content)
                    elif self.column == 'fiber_content':
                        self.labels_list.append(recipe.fiber_content)
                    elif self.column == 'sugar_content':
                        self.labels_list.append(recipe.sugar_content)
                    elif self.column == 'protein_content':
                        self.labels_list.append(recipe.protein_content)
                    self.current_image_index += 1
                temp_image_index += 1
                if self.current_image_index >= self.current_loop * self.batch: \
                        # Было загружено нужное количество картинок для текущего цикла загрузки
                    break
        if self.current_image_index >= self.dataset_size - 1:  # Если проход по датасету завершен
            self.current_image_index = 0
            self.current_loop = 1
            if self.current_walkthrough == self.num_epochs:  # Если прошло нужное количество эпох
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
            return self.images_list[:int(len(self.images_list) * (1 - self.validation_split))], \
                   self.labels_list[:int(len(self.images_list) * (1 - self.validation_split))], \
                   self.images_list[int(len(self.images_list) * (1 - self.validation_split)):], \
                   self.labels_list[int(len(self.images_list) * (1 - self.validation_split)):],
