import os
from flask import Flask, request, jsonify
import pandas as pd
import json
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)


# to use it when loading the model
def auc(y_true, y_pred):
    auc_score = tf.metrics.auc(y_true, y_pred)[1]
    tf.compat.v1.keras.backend.get_session().run(tf.local_variables_initializer())
    return auc_score


# load the model, and pass in the custom metric function
model_file = 'model.h5'

global graph
graph = tf.compat.v1.get_default_graph()


# model = load_model(model_file, custom_objects={'auc': auc})


@app.route('/')
def home():
    return '<h1>It works!!!</h1>'


@app.route('/json', methods=['POST'])
def predict_json():
    result = {"success": False}
    data = request.get_json(force=True)
    df = pd.DataFrame(data)
    arr = df.iloc[:1, :187].values
    arr1 = arr.reshape(len(arr), 186, 1)
    with graph.as_default():
        model = load_model(model_file, custom_objects={'auc': auc})
        prediction = model.predict(arr1)
        result["success"] = True
        result["prediction"] = str(prediction)

    return jsonify(result)


if __name__ == '__main__':
    port = os.environ.get('PORT')
    if port:
        app.run(host='0.0.0.0', port=int(port), debug=True)
    else:
        app.run(debug=True)
