from flask import Flask, render_template, request
import joblib
import numpy as np
import pickle

app = Flask(__name__, template_folder='templates')

# Load the model
with open('xgboost1.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def super():
    return render_template('index.html')

@app.route('/index', methods=['POST'])
def index():
    C_wt = float(request.form['C_wt'])
    H_wt = float(request.form['H_wt'])
    O_wt = float(request.form['O_wt'])
    N_wt = float(request.form['N_wt'])
    S_wt = float(request.form['S_wt'])
    Temp_C = float(request.form['Temp_C'])
    Time_min = float(request.form['Time_min'])
    FC_wt = float(request.form['FC_wt'])
    VM_wt = float(request.form['VM_wt'])
    ASH_wt = float(request.form['ASH_wt'])

    # Perform any necessary preprocessing on the input data

    # Make predictions using the loaded model
    input_data = np.array([[C_wt, H_wt, O_wt, N_wt, S_wt, Temp_C, Time_min, FC_wt, VM_wt, ASH_wt]])
    pred = model.predict(input_data)

    # Pass the prediction to the template
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)