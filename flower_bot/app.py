#Usage: python app.py
import os
 
from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import numpy as np
import argparse
# import imutils
import cv2
import time
import uuid
import base64

# img_width, img_height = 150, 150
# model_path = './models/model.h5'
# model_weights_path = './models/weights.h5'
# model = load_model(model_path)
# #model.load_weights(model_weights_path)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

def get_as_base64(url):
    return base64.b64encode(request.get(url).content)

# def predict(file):
#     x = load_img(file, target_size=(img_width,img_height))
#     x = img_to_array(x)
#     x = np.expand_dims(x, axis=0)
#     array = model.predict(x)
#     result = array[0]
#     answer = np.argmax(result)
#     if answer == 0:
#         print("Label: Daisy")
#     elif answer == 1:
# 	    print("Label: Rose")
#     elif answer == 2:
# 	    print("Label: Sunflower")
#     return answer

from keras.models import model_from_json

from PIL import Image
import numpy as np
import os
import cv2

import tensorflow as tf
global graph,model

graph = tf.get_default_graph()


# load json and create model


json_file = open('flower_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("flower_model.h5")
print("Loaded model from disk")


def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((50, 50))
    return np.array(image)


def get_flower_name(label):
    if label == 0:
        return "ROSE"
    if label == 1:
        return "DAISY"
    if label == 2:
        return "SUNFLOWER"


def predict_flower(file):
    print("Predicting .................................")
    ar = convert_to_array(file)
    ar = ar / 255
    label = 1
    a = []
    a.append(ar)
    a = np.array(a)
    with graph.as_default():
        #y = model.predict(X)
        score = loaded_model.predict(a, verbose=1)
    print(score)
    label_index = np.argmax(score)
    print(label_index)
    acc = np.max(score)
    flower = get_flower_name(label_index)
    print(flower)
    print("The predicted Flower is a " + flower + " with accuracy =    " + str(acc))
    return flower





def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__,static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='../uploads/template.jpg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = predict_flower(file_path)
            # if result == 0:
            #     label = 'Daisy'
            # elif result == 1:
            #     label = 'Rose'
            # elif result == 2:
            #     label = 'Sunflowers'
            print(result)
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('template.html', label=result, imagesource='../uploads/' + filename)

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=False
    app.run(host='0.0.0.0', port=3000)