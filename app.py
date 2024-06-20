from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# Load the model
model = joblib.load('car_price_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define all possible columns expected by the model
model_columns = ['make', 'model', 'year', 'mileage', 'condition', 'transmission', 'engine_volume', 'company',
                 'fuel_type']

# Set default values for optional fields
default_values = {
    'transmission': 'unknown',
    'engine_volume': 0.0,
    'company': 'unknown',
    'fuel_type': 'unknown'
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Ensure all required columns are present in the data
        for column in model_columns:
            if column not in data:
                if column in default_values:
                    data[column] = default_values[column]
                else:
                    return jsonify({'error': f'Missing required data: {column}'}), 400

        # Convert data into DataFrame
        input_data = pd.DataFrame([data])

        # Select only the columns expected by the model
        input_data = input_data[model_columns]

        # Make prediction
        prediction = model.predict(input_data)

        # Return the predicted price
        return jsonify({'predicted_price': float(prediction[0])})  # Ensure the prediction is converted to float

    except KeyError as e:
        return jsonify({'error': f'Missing key in JSON data: {e}'}), 400

    except ValueError as e:
        return jsonify({'error': f'Invalid value: {e}'}), 400

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
