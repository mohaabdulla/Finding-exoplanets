from flask import Flask, render_template, request, jsonify
import os
import sys
import tempfile
import pandas as pd
from predict import load_and_predict
from train import main as train_main
import argparse
from flask import send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/results.json')
def serve_file():
    with open("./templates/results.json", "r") as f:
        content = f.read()
    return content


@app.route('/explore')
def explore_index():
    # List interactive exploration artifacts
    base = os.path.join(os.path.dirname(__file__), "exploration", "interactive")
    files = []
    if os.path.isdir(base):
        files = sorted([f for f in os.listdir(base) if f.lower().endswith('.html') and f.lower() != 'scatter_matrix.html'])
    return render_template('explore.html', files=files)


@app.route('/exploration/interactive/<path:filename>')
def serve_interactive(filename):
    base = os.path.join(os.path.dirname(__file__), "exploration", "interactive")
    return send_from_directory(base, filename)

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        # Get parameters from request
        data = request.json
        contamination = data.get('contamination', 0.08)
        rf_trees = data.get('rf_trees', 800)
        mc_iters = data.get('mc_iters', 60)
        use_tuning = data.get('tune', True)
        use_smote = data.get('smote', False)
        use_calibration = data.get('calibrate', False)
        use_ensemble = data.get('ensemble', False)
        fast_mode = data.get('fast', False)
        
        # Here you would call your training function with these parameters
        # For now, we'll simulate training
        print(f"Training with params: {data}")
        
        # Uncomment to run actual training:
        # train_main()  # You'll need to modify train.py to accept these params
        
        return jsonify({
            'status': 'success',
            'message': 'Model training completed successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            file.save(tmp_file.name)
            
            # Make predictions using your code
            results_df = load_and_predict(
                data_file=tmp_file.name,
                model_dir="models",
                output_file=None
            )
            
            # Clean up
            os.unlink(tmp_file.name)
            
            if results_df is None:
                return jsonify({'error': 'Prediction failed'}), 500
            
            # Convert results to JSON-friendly format
            results = {
                'predictions': results_df.to_dict('records'),
                'summary': {
                    'total_samples': len(results_df),
                    'confirmed_predictions': int(results_df['prediction'].sum()),
                    'anomalies_detected': int(results_df['is_anomaly'].sum()),
                    'hz_candidates': int(results_df.get('habitable_zone', 0).sum() if 'habitable_zone' in results_df.columns else 0)
                }
            }
            
            return jsonify(results)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    return jsonify({
        'status': 'ready',
        'models_available': os.path.exists('models')
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)