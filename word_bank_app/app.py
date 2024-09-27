import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models import generate_text, sample_with_temperature, clean_generated_text, add_filler_words_smart, generate_structured_text

# Load the cleaned dataset
cleaned_df = pd.read_csv('cleaned_df.csv')

# Load the trained XGBoost pipeline model using pickle
with open('trained_pipeline.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define max_len (make sure this matches the length used when training your seq2seq model)
max_len = 100  # Adjust based on your model's training setup

# Create a Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_cluster():
    # Get form data
    character_name = request.form['character_name']
    character_species = request.form['character_species']
    character_class = request.form['character_class']

    # Encode the input values
    species_encoder = dict(zip(cleaned_df['character_species'].str.lower(), cleaned_df['encoded_species']))
    class_encoder = dict(zip(cleaned_df['character_class'].str.lower(), cleaned_df['encoded_class']))

    encoded_species = species_encoder.get(character_species.lower(), -1)
    encoded_class = class_encoder.get(character_class.lower(), -1)

    # Check if any encoding failed
    if -1 in [encoded_species, encoded_class]:
        return "Error: Invalid species or class."

    # Prepare feature array and predict the cluster
    input_features = np.array([[encoded_species, encoded_class]])
    predicted_cluster = model.predict(input_features)[0]  # Extract the single predicted cluster value

    # Filter the cleaned_df dataset based on the predicted cluster
    filtered_df = cleaned_df[cleaned_df['backstory_cluster'] == predicted_cluster]

    # Select a random row from the filtered dataframe
    if not filtered_df.empty:
        random_row = filtered_df.sample(n=1).iloc[0]  # Select a random row
        random_backstory = random_row['character_backstory']  # Get the backstory column
    else:
        random_backstory = "No backstory found for this cluster."

    # Extract the first 5 words of the random backstory
    first_five_words = ' '.join(random_backstory.split()[:5])

    # Use the first 5 words as the seed_text
    seed_text = first_five_words
    seed_sequence = tokenizer.texts_to_sequences([seed_text])
    seed_sequence = pad_sequences(seed_sequence, maxlen=max_len, padding='post')

    # Generate text using the trained neural network
    generated_text = generate_text(seed_sequence, tokenizer, max_len)
    cleaned_text = clean_generated_text(generated_text)
    filler_text = add_filler_words_smart(cleaned_text)
    punctuated_text = generate_structured_text(filler_text)

    # Return the generated text as part of the response
    return render_template('result.html', character_name=character_name, backstory=random_backstory, generated_text=punctuated_text)

if __name__ == '__main__':
    app.run(debug=True)