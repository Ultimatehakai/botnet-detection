'''import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template,Response

#create flask app
app = Flask(__name__)

#load the pickle model
model = joblib.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(X) for X in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    
    return render_template("index.html", prediction_text = "this is safe {}".format(prediction))

if __name__ == " __main__":
    app.run(debug=True)
    '''
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template

# Create Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from form
    features = [float(x) for x in request.form.values()]

    # Reshape features to match model input
    features_array = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features_array)

    # Map prediction to class label
    if prediction == 1:
        prediction_text = "Attack"
    else:
        prediction_text = "Normal"

    return render_template('index.html', prediction_text=prediction_text)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
