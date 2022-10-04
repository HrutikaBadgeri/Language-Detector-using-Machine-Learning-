from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import pickle
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

app = Flask(__name__)

#Loading the trainded ML model
model = pickle.load(open('LanguageDetection_model.pkl', 'rb'))
cv = pickle.load(open('CountVectoriser.pkl', 'rb'))
le = pickle.load(open('LabelEncoder.pkl', 'rb'))


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

def model_prediction(text):
    x = cv.transform([text])
    lang = model.predict(x)
    lang = le.inverse_transform(lang)
    return lang[0]

@app.route('/answer', methods=['GET','POST'])
def answer():
    if request.method == 'POST':
        text = request.form.get('language')
        output = model_prediction(text)
        return render_template('index.html', prediction=f'{output}')
    else: 
        print('GET')
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debgug=True)
