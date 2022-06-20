from flask import Flask, render_template,request
import numpy as np
import pickle

#creating constructor
app=Flask(__name__, template_folder='templates', static_folder='static')
model=pickle.load(open('car.pkl', 'rb'))
#print(model)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''v1 = request.form['YearsExperience']'''
    features=[int(x) for x in request.form.values()]
    final_feature=[np.array(features)]
    pred = model.predict(final_feature)
    out=pred

    return render_template('home.html', prediction=out)
if __name__ == '__main__':
    app.run(debug=True)     
