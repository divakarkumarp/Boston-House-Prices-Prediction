import json
import pickle
#from django.sortcuts import render
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pre-trained BHPPmodel and scalar objects using pickle
BHPPmodel=pickle.load(open('BHPPmodel.pkl', 'rb'))
scalar=pickle.load(open('scaling.pkl', 'rb'))
# Define the home route that renders the home.html template
@app.route('/')
def home():
    return render_template('home.html')
# Define a route for making predictions via API (POST request)
@app.route('/predict_api',methods=['POST'])
def predict_api():
    # Extract JSON data from the request
    data=request.json['data']
    print(data)

    # Convert the received data to a 2D NumPy array and transpose it
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    # Make predictions using the loaded BHPPmodel
    output=BHPPmodel.predict(new_data)
    print(output[0])
    # Return the prediction as a JSON response
    return jsonify(output[0])
# Define a route for making predictions (POST request)
@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=BHPPmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House Price Prediction is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)

           






    