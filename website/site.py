from flask import Flask, render_template, redirect, url_for, request
from utils import image_resize
from PIL import Image
import os
import tensorflow as tf
import json

model = tf.keras.models.load_model('model') # model.save('model')
web_site = Flask(__name__)

@web_site.route('/', methods= ['POST', 'GET'])
def home():
    if request.method == 'GET':
        return render_template('page.html')
    else:
        image = request.files['image']
        image.save(os.path.join('static/uploaded', image.filename))

        img = Image.open(os.path.join('static/uploaded', image.filename))
        img = image_resize(img)
        img.save(os.path.join('static/uploaded', image.filename))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255

        calories = model.predict(tf.data.Dataset.from_tensor_slices([img]).batch(1))
        print(calories[0][0])
        return render_template('page.html',calories=str(calories[0][0])+" Калорий", source=url_for('static', filename='uploaded/'+image.filename))

@web_site.route('/mobile', methods= ['POST'])
def mobile():
    image = None
    try: image = request.files['image']
    except: pass
    print(image)
    image.save(os.path.join('static/uploaded', image.filename))
    img = Image.open(os.path.join('static/uploaded', image.filename))
    img = image_resize(img)
    img.save(os.path.join('static/uploaded', image.filename))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img/255
    calories = model.predict(tf.data.Dataset.from_tensor_slices([img]).batch(1))
    print(calories[0][0])

    to_return = calories[0][0]
    return json.dumps(to_return)

web_site.run(debug=True, host='192.168.100.3')