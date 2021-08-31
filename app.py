import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model_VR = pickle.load(open('gaussian_model', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    # request.form['open']
    vr = model_VR.predict(final_features)
    if(vr==0):
        prediction='Iris-setosa'
    if(vr==1):

        prediction='Iris-versicolor'

    if(vr==2):
        prediction='Iris-versicolor'

    return render_template('index.html', prediction = prediction)
    
if __name__ == "__main__":
    app.run(debug=True)