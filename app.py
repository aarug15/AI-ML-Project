from flask import Flask, render_template, request
import pickle
import numpy as np

# Load your model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Death.html')

@app.route('/predict', methods=['POST'])
def predict_death():
    # Extract form inputs
    age = float(request.form.get('age'))
    anaemia = float(request.form.get('anaemia'))
    creatinine_phosphokinase = float(request.form.get('cpk'))
    diabetes = float(request.form.get('diabetes'))
    ejection_fraction = float(request.form.get('ejection'))
    high_blood_pressure = float(request.form.get('high_bp'))
    platelets = float(request.form.get('platelets'))
    serum_creatinine = float(request.form.get('serum_c'))
    serum_sodium = float(request.form.get('serum_Na'))
    sex = float(request.form.get('sex'))
    smoking = float(request.form.get('smoking'))
    time = float(request.form.get('time'))

    # Predict
    features = np.array([
        age, anaemia, creatinine_phosphokinase, diabetes,
        ejection_fraction, high_blood_pressure, platelets,
        serum_creatinine, serum_sodium, sex, smoking, time
    ]).reshape(1, -1)

    prediction = model.predict(features)[0]  # Either 0 or 1
    result = f"Predicted Death: {prediction}"

    return render_template('Death.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
