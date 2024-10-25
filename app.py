import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Loading Model
model=pickle.load(open('model.pkl','rb'))
scalar=pickle.load(open('Scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(int(output[0]))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert form data to integers and reshape for the model
        data = [float(x) for x in request.form.values()]  # Using float to handle decimal inputs if needed
        final_input = scalar.transform(np.array(data).reshape(1, -1))
        
        # Make prediction
        output = model.predict(final_input)[0]
        
        # Generate prediction message
        if output == 0:
            txt = "Congrats, you don't have Heart Disease!"
        elif output == 1:
            txt = "Sorry, you may have Heart Disease"
        
    except ValueError:
        # Error message for invalid inputs
        txt = "Please enter valid numeric values for all fields."
    except Exception as e:
        # General error message for unexpected issues
        print(f"Error during prediction: {e}")
        txt = "An error occurred. Please try again."
    
    return render_template("home.html", prediction_txt=txt)

if __name__=="__main__":
    app.run(debug=True)