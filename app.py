from flask import Flask,request,jsonify
import pickle
import numpy as np





model = pickle.load(open('SympDetector.pkl', 'rb'))

app= Flask(__name__)

@app.route('/')
def home():
    return "hello world"

@app.route('/predict',methods=['post'])
def predict():
    symptom1= request.form.get('symptom1')
    symptom2= request.form.get('symptom2')
    symptom3 = request.form.get('symptom3')
    symptom4 = request.form.get('symptom4')
    symptom5 = request.form.get('symptom5')
    symptom6 = request.form.get('symptom6')


    input_query = np.array([[symptom1, symptom2, symptom3, symptom4, symptom5, symptom6]])

    result = model.predict(input_query)[0]




    return jsonify({'predicted Disease':str(result)})


if __name__=='__main__':
    app.run(debug=True)
