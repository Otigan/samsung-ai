from utils.data_base import FoodImage, Recipe, init_base
import pandas as pd
import os
import time
from PIL import Image
from urllib.request import urlretrieve

db_session = init_base()


def image_resize(image):
    image.thumbnail((224, 224), Image.ANTIALIAS)
    image_size = image.size
    width = image_size[0]
    height = image_size[1]
    if (width != height):
        bigside = width if width > height else height
        background = Image.new('RGB', (bigside, bigside), (255, 255, 255))
        offset = (int(round(((bigside - width) / 2), 0)), int(round(((bigside - height) / 2), 0)))
        background.paste(image, offset)
        return background
    else: return image

columns = ['Images', 'Calories', 'RecipeCategory', 'TotalTime', 'Keywords',
           'RecipeIngredientParts', 'FatContent', 'SaturatedFatContent',
           'CholesterolContent', 'SodiumContent', 'CarbohydrateContent',
           'FiberContent', 'SugarContent', 'ProteinContent',
           'RecipeInstructions', 'RecipeIngredientQuantities', 'Name']
file = pd.read_parquet('recipes.parquet', columns=columns)
file = file[file.Images.str.len() != 0]
file = file[file.Keywords.str.len() !=0]
row, col = file.shape
print(row, col)
del row, col

file_images = file.Images.to_numpy()
file_calories = file.Calories.to_numpy()
file_categories = file.RecipeCategory.to_numpy()
file_total_time = file.TotalTime.to_numpy()
file_keywords = file.Keywords.to_numpy() # may cause a problem. Includes 'NA'
file_ingredients = file.RecipeIngredientParts.to_numpy()
file_ingredients_quantity = file.RecipeIngredientQuantities.to_numpy()
file_fat_content = file.FatContent.to_numpy()
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

total_images = 0
for i in range(len(file_images)):
    try:
        for j in range(len(file_images[i])):
            total_images += 1
    except:pass

    try:
        temp_keywords = ''
        for j in range(len(file_keywords[i])):
            temp_keywords += f'{file_keywords[i][j]}_'
        temp_keywords = temp_keywords[:-1]  # removing last '_'
        file_keywords[i] = temp_keywords

        temp_ingredients = ''
        for j in range(len(file_ingredients[i])):
            temp_ingredients += f'{file_ingredients[i][j]}_'
        temp_ingredients = temp_ingredients[:-1]
        file_ingredients[i] = temp_ingredients

        temp_ingredients_quantity = ''
        for j in range(len(file_ingredients_quantity[i])):
            temp_ingredients_quantity += f'{file_ingredients_quantity[i][j]}_'
        temp_ingredients_quantity = temp_ingredients_quantity[:-1]
        file_ingredients_quantity[i] = temp_ingredients_quantity

        temp_instructions = ''
        for j in range(len(file_instructions[i])):
            temp_instructions += f'{file_instructions[i][j]}_'
        temp_instructions = temp_instructions[:-1]
        file_instructions[i] = temp_instructions # Преобразование массивов в строки
    except: pass

def start_downloading(position = 0):
    current_image = 0
    for i in range(len(file_images)):
        if(i >= position): # Если достигли нужной позиции продолжения скачивания
            try:
                recipe_to_db = Recipe(id=i+1, calories=file_calories[i], total_time=file_total_time[i],
                          keywords=file_keywords[i], ingredients=file_ingredients[i],
                          ingredients_quantity=file_ingredients_quantity[i],
                          fat_content=file_fat_content[i],
                          saturated_fat_content=file_saturated_fat_content[i],
                          cholesterol_content=file_cholesterol_content[i],
                          sodium_content=file_sodium_content[i],
                          carbohydrate_content=file_carbohydrate_content[i],
                          fiber_content=file_fiber_content[i],
                          sugar_content=file_sugar_content[i],
                          protein_content=file_protein_content[i],
                          instructions=file_instructions[i],
                          name=file_name[i])
                db_session.add(recipe_to_db)
                db_session.commit()
                for j in range(len(file_images[i])):
                    temp_path = f"Pictures/{file_categories[i].replace('/','_')}/image{current_image + 1}.jpg"
                    try:
                        if not os.path.exists(os.path.dirname(temp_path)):
                            os.makedirs(os.path.dirname(temp_path))
                        time.sleep(0.01)
                        urlretrieve(file_images[i][j], temp_path)

                        img = Image.open(temp_path)
                        img = image_resize(img)
                        img.save(temp_path)
                        img.close()
                    except:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    else:
                        image_to_db = FoodImage(id=current_image+1, recipe_id=i+1)
                        db_session.add(image_to_db)
                        db_session.commit()
                    print(f'{current_image} / {total_images}  ({round(current_image/total_images*100, 1)}%)')
                    current_image += 1
                    if current_image % 1000 == 0:
                        time.sleep(10)
                    if current_image % 20000 == 0:
                        time.sleep(120)
            except:
                try:
                    for j in range(len(file_images[i])): #Если не удалось скачать изображение текущий индекс все равно увеличится
                        current_image += 1
                except:pass
                continue
        else: #Если не достигли нужной позиции рецепта
            try:
                for j in range(len(file_images[i])): # Проход по рецепту
                    current_image += 1 # Увеличиваем текущий индекс
            except: pass


def continue_downloading():
    last_recipe = db_session.query(Recipe).order_by(Recipe.id.desc()).first()
    if last_recipe:
        last_recipe_id = last_recipe.id
        db_session.query(FoodImage).filter_by(recipe_id=last_recipe_id).delete()
        db_session.query(FoodImage).filter_by(recipe_id=last_recipe_id - 1).delete()
        db_session.query(Recipe).filter_by(id=last_recipe_id).delete()
        db_session.query(Recipe).filter_by(id=last_recipe_id-1).delete()
        db_session.commit() # Удаление последних двух записей, которые могли записаться некорректно, из-за прерывания
        start_downloading(position=last_recipe_id-2)
    else:
        start_downloading()


if __name__ == '__main__':
    continue_downloading()