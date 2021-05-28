import os

from flask import Flask, request
from PIL import Image
import tensorflow as tf
import numpy as np

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def home():
    return 'HOME'


@app.route('/digit', methods=['POST', 'GET'])
def function():
    print('primul pas')
    if request.method == 'POST':
        if "digit" in request.files:
            print('este inauntru')
            digit = request.files['digit']

            filename = digit.filename
            digit.save(digit.filename)
            print(digit.filename)

            new_model = tf.keras.models.load_model('my_own_model')
            print('a dat load la model')
            exem = np.loadtxt(digit.filename)
            print(exem[0])
            print(type(exem[0]))
            ex = exem[0].reshape(1,28,28)
            print(ex.shape)
            output = new_model.predict(ex)
            print(output)
            return str(np.argmax(output))
            # with open(digit.filename, "rb") as image:
            #     print('este inauntru in imagine')
            #     f = image.read()
            #     print(f)
            #     output = new_model.predict(f)
            #     return np.argmax(output)
    return 'Digit'


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
