import json
import keras
import random
import numpy as np
import tensorflow as tf
from flask import Flask, request

app = Flask(__name__)
model = keras.models.load_model('MNIST.h5')
feature_model = keras.models.Model(
    model.inputs, [layer.output for layer in model.layers]
)

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0

def get_prediction():
    index = random.randint(0, x_test.shape[0]-1)
    image = x_test[index, :, :]
    image_arr = np.reshape(image, (1, 784))
    return feature_model.predict(image_arr), image, y_test[index]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        preds, image, label = get_prediction()
        final_preds = [p.tolist() for p in preds]
        return json.dumps({
            'prediction': final_preds,
            'image': image.tolist(),
            'label': int(label)
        })
    return 'Welcome to the MNIST Neural Network Visualizer Model Server'

if __name__ == '__main__':
    app.run(debug=True)