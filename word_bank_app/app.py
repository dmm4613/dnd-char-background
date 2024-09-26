from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib, pickle
import random

# Load the cleaned dataset
cleaned_df = pd.read_csv('cleaned_df.csv')

# Load the trained model
with open('trained_pipeline.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Extract the encoders from the dataset
species_encoder = dict(zip(cleaned_df['character_species'].str.lower(), cleaned_df['encoded_species']))
class_encoder = dict(zip(cleaned_df['character_class'].str.lower(), cleaned_df['encoded_class']))

# Create a Flask app
app = Flask(__name__)

@app.route('/')
def home():
    # Render the form page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_cluster():
    # Get form data
    character_name = request.form['character_name']
    character_species = request.form['character_species']
    character_class = request.form['character_class']

    # Encode the input values
    encoded_species = species_encoder.get(character_species.lower(), -1)
    encoded_class = class_encoder.get(character_class.lower(), -1)

    # Check if any encoding failed
    if -1 in [encoded_species, encoded_class]:
        return "Error: Invalid species or class."

    # Prepare feature array and predict the cluster
    input_features = np.array([[encoded_species, encoded_class]])
    predicted_cluster = model.predict(input_features)[0]

    # Filter the cleaned_df dataset based on the predicted cluster
    filtered_df = cleaned_df[cleaned_df['backstory_cluster'] == predicted_cluster]

    # Select a random row from the filtered dataframe
    if not filtered_df.empty:
        random_row = filtered_df.sample(n=1).iloc[0]  # Select a random row
        random_backstory = random_row['character_backstory']  # Get the backstory column
    else:
        random_backstory = "No backstory found for this cluster."

    # Return the backstory as part of the response
    return render_template('result.html', character_name=character_name, backstory=random_backstory)

if __name__ == '__main__':
    app.run(debug=True)