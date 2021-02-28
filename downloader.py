from utils.data_base import Image, Recipe, init_base
import pandas as pd
import os
import time
from PIL import Image
import PIL
from urllib.request import urlretrieve

db_session = init_base()

columns = ['Images', 'Calories', 'RecipeCategory', 'TotalTime', 'Keywords',
           'RecipeIngredientParts', 'FatContent', 'SaturatedFatContent',
           'CholesterolContent', 'SodiumContent', 'CarbohydrateContent',
           'FiberContent', 'SugarContent', 'ProteinContent',
           'RecipeInstructions', 'Name']
file = pd.read_parquet('recipes.parquet', columns=columns)
file = file[file.Images.str.len() != 0 & file.keywords.str.len() !=0]
row, col = file.shape
print(row, col)
del row, col

file_images = file.Images.to_numpy()
file_calories = file.Calories.to_numpy()
file_categories = file.RecipeCategory.to_numpy()
file_total_time = file.TotalTime.to_numpy()
file_keywords = file.Keywords.to_numpy() # may cause a problem. Includes 'NA'
file_ingredients = file.RecipeIngredientParts.to_numpy()
file_fat_content = file.FileContent.to_numpy()
file_saturated_fat_content = file.SaturatedFatContent.to_numpy()
file_cholesterol_content = file.CholesterolContent.to_numpy()
file_sodium_content = file.SodiumContent.to_numpy()
file_carbohydrate_content = file.CarbohydrateContent.to_numpy()
file_fiber_content = file.FiberContent.to_numpy()
file_sugar_content = file.SugarContent.to_numpy()
file_protein_content = file.ProteinContent.to_numpy()
file_instructions = file.RecipeInstructions.to_numpy()
file_name = file.Name.to_numpy()
del file

images = []
calories = []
categories = []
total_time = []
keywords = []
ingredients = []
fat_content = []
saturated_fat_content = []
cholesterol_content = []
sodium_content = []
carbohydrate_content = []
fiber_content = []
sugar_content = []
protein_content = []
instructions = []
name = []

for i in range(len(file_images)):
    try:
        temp_ingredients = ''
        temp_instructions = ''
        for j in range(len(file_keywords[i])):
            temp_keywords = ''
            temp_keywords += f'{file_keywords[i][j]}_'
            temp_keywords = temp_keywords[:-1] # removing last '_'

        for j in range(len(file_ingredients[i])):
            temp_ingredients += f'{file_ingredients[i][j]}_'
            temp_ingredients = temp_ingredients[:-1]
        for j in range(len(file_instructions[i])):
            temp_instructions += f'{file_instructions[i][j]}_'
            temp_instructions = temp_instructions[:-1]
        for j in range(len(file_images[i])):
            images.append(file_images[i][j])
            calories.append(file_calories[i])
            categories.append(file_categories[i])
            keywords.append(temp_keywords)
            ingredients.append(temp_ingredients)
            instructions.append(temp_instructions)
            total_time.append(file_total_time[i])
            fat_content.append(file_fat_content)
            saturated_fat_content.append(file_saturated_fat_content)
            cholesterol_content.append(file_cholesterol_content)
            sodium_content.append(file_sodium_content)
            carbohydrate_content.append(file_carbohydrate_content)
            fiber_content.append(file_fiber_content)
            sugar
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

