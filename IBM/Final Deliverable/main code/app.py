# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:19:28 2022

@author: Mr...Vs..99
"""

from flask import Flask, render_template, request
import os
import numpy as np          # used for numerical analysis
import tensorflow as tf     # to load the trained model
import requests

app = Flask(__name__, template_folder="templates")  # initializing a flask app

model = tf.keras.models.load_model("nutrition.h5")  # Loading the model
print("Loaded model from disk")


@app.route('/')  # route to display the home page
def home():
    return render_template('home.html')  # rendering the home page


@app.route('/image1', methods=['GET', 'POST'])  # routes to the index html
def image1():
    return render_template("image.html")


@app.route('/predict', methods=['GET', 'POST'])  # route to show the predictions in a Web UI
def lanuch():
    if request.method == 'POST':
        f = request.files['file']  # requesting the file
        basepath = os.path.dirname('__file__')  # storing the file directory
        filepath = os.path.join(basepath, "uploads", f.filename)  # storing the file in uploads folder
        f.save(filepath)  # saving the file

        img = tf.keras.preprocessing.image.load_img(filepath, target_size=(64, 64))  # load and reshaping the image
        x = tf.keras.preprocessing.image.img_to_array(img)  # converting image to an array
        x = np.expand_dims(x, axis=0)  # changing the dimensions of the image

        pred = np.argmax(model.predict(x), axis=1)
        print("prediction", pred)  # printing the prediction
        index = ['APPLE', 'BANANA', 'ORANGE', 'PINEAPPLE', 'WATERMELON']

        result = str(index[pred[0]])
        print(result)
        x = result
        result = nutrition(result)
        print(result)

        return render_template("predict.html", showcase= (result), showcase1= (x))


def nutrition(index):

    url = "https://calorieninjas.p.rapidapi.com/v1/nutrition"

    querystring = {"query": index}

    headers = {
        "X-RapidAPI-Key": "85887549f4msh51e7315b280a87ep1f43e0jsn585c940f2ea6",
        "X-RapidAPI-Host": "calorieninjas.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    print(response.text)
    return response.json()['items']


if __name__ == "__main__":
    # running the app
    app.run(debug=False)
