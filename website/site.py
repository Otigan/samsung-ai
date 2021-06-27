from flask import Flask, render_template, redirect, url_for, request
from utils import image_resize
from PIL import Image
from flask import jsonify
import os
import tensorflow as tf

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
        return render_template('page.html', calories=str(int(calories[0][0]))+" Килокалорий",
                               source=url_for('static', filename='uploaded/'+image.filename))

@web_site.route('/mobile', methods= ['POST'])
def mobile():
    try: image = request.files['image']
    except: return jsonify({'result': 'failure'})
    image.save(os.path.join('static/uploaded', image.filename))
    img = Image.open(os.path.join('static/uploaded', image.filename))
    img = image_resize(img)
    img.save(os.path.join('static/uploaded', image.filename))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img/255
    calories = model.predict(tf.data.Dataset.from_tensor_slices([img]).batch(1))

    to_return = {'result': 'success', 'calories': int(calories[0][0])}
    return jsonify(to_return)

web_site.run(debug=True, host='192.168.100.3')