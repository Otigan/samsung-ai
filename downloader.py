from utils.data_base import Calories, init_base
import pandas as pd
import os
import time
from PIL import Image
import PIL
from urllib.request import urlretrieve

db_session = init_base()

columns = ['Images', 'Calories', 'RecipeCategory']
file = pd.read_parquet('recipes.parquet', columns=columns)
file = file[file.Images.str.len() != 0]
row, col = file.shape
print(row, col)

file_images = file.Images.to_numpy()
file_calories = file.Calories.to_numpy()
file_categories = file.RecipeCategory.to_numpy()
images = []
calories = []
categories = []

for i in range(len(file_images)):
    try:
        for j in range(len(file_images[i])):
            images.append(file_images[i][j])
            calories.append(file_calories[i])
            categories.append(file_categories[i])
    except: pass

for i in range(len(images)):
    temp_path = f"Pictures/{categories[i].replace('/','_')}/image{i + 1}.jpg"
    try:
        if not os.path.exists(os.path.dirname(temp_path)):
            os.makedirs(os.path.dirname(temp_path))
        time.sleep(0.01)
        urlretrieve(images[i], temp_path)

        img = Image.open(temp_path)
        img.thumbnail((224, 224), Image.ANTIALIAS)
        image_size= img.size
        width = image_size[0]
        height = image_size[1]
        if (width != height):
            bigside = width if width > height else height
            background = Image.new('RGB', (bigside, bigside), (255, 255, 255))
            offset = (int(round(((bigside - width) / 2), 0)), int(round(((bigside - height) / 2), 0)))
            background.paste(img, offset)
            img.close()
            background.save(temp_path)
            background.close()
        else:
            img.save(temp_path)
            img.close()

    except:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        continue
    else:
        calories_to_db = Calories(id=i+1, calories=calories[i])
        db_session.add(calories_to_db)
        db_session.commit()

