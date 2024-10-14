from flask import Flask, request, jsonify
import core

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello! This is the homepage of the seismic data processing app."

# Route to get the CSV data files and return the seismic analysis
@app.route('/process', methods=['POST'])
def process_data():
    file = request.files['file']

    # Verify if the file is a CSV file
    if file.filename.endswith('.csv'):
        # Convert the file to a pandas DataFrame
        dataset = file.read()

        # Call the mathematical core to process the data
        results = core.process_seismic_data(dataset)

        # Return the results as a JSON object
        return jsonify(results)
    else:
        return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400
    
if __name__ == '__main__':
    app.run(debug=True)