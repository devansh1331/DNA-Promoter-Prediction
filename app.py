from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import joblib
import os
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this for security

# Load trained model
MODEL_PATH = os.path.join('model', 'promoter_model.pkl')
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Import preprocessing functions
from model.preprocessing import preprocess_sequence_for_prediction, validate_dna_sequence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles user input and displays prediction result."""
    try:
        dna_sequence = request.form.get('dna_sequence', '').strip()

        if not dna_sequence:
            flash('Please enter a DNA sequence', 'error')
            return redirect(url_for('index'))

        # Validate sequence
        is_valid, message = validate_dna_sequence(dna_sequence)
        if not is_valid:
            flash(f'Invalid DNA sequence: {message}', 'error')
            return redirect(url_for('index'))

        if model is None:
            flash('Model failed to load!', 'error')
            return redirect(url_for('index'))

        # Preprocess and predict
        processed_sequence = preprocess_sequence_for_prediction(dna_sequence)
        prediction = model.predict(processed_sequence)[0]
        probability = model.predict_proba(processed_sequence)[0]

        # Store result
        result = {
            'sequence': dna_sequence,
            'is_promoter': bool(prediction),
            'confidence': float(max(probability)) * 100,
            'promoter_probability': float(probability[1]) * 100,
            'non_promoter_probability': float(probability[0]) * 100
        }

        return render_template('result.html', result=result)

    except Exception as e:
        flash(f'Prediction error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic predictions."""
    try:
        data = request.get_json()
        if not data or 'sequence' not in data:
            return jsonify({'error': 'No sequence provided'}), 400

        dna_sequence = data['sequence'].strip()

        # Validate sequence
        is_valid, message = validate_dna_sequence(dna_sequence)
        if not is_valid:
            return jsonify({'error': message}), 400

        if model is None:
            return jsonify({'error': 'Model failed to load'}), 500

        # Predict
        processed_sequence = preprocess_sequence_for_prediction(dna_sequence)
        prediction = model.predict(processed_sequence)[0]
        probability = model.predict_proba(processed_sequence)[0]

        result = {
            'sequence': dna_sequence,
            'is_promoter': bool(prediction),
            'confidence': float(max(probability)),
            'promoter_probability': float(probability[1]),
            'non_promoter_probability': float(probability[0])
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask App...")
    print(f"🔬 Model path: {os.path.abspath(MODEL_PATH)}")
    print(f"✅ Model loaded: {model is not None}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
