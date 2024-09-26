import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the cleaned dataset
cleaned_df = pd.read_csv('cleaned_df.csv')

# Load the trained XGBoost pipeline model using pickle
with open('trained_pipeline.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the trained neural network model (seq2seq)
seq2seq_model = load_model('seq2seq_new_clean_model.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define max_len (make sure this matches the length used when training your seq2seq model)
max_len = 100  # Adjust based on your model's training setup

# Load the trained encoder and decoder models
encoder_model = load_model('encoder_model.h5')
decoder_model = load_model('decoder_model.h5')

# Function to sample with temperature
def sample_with_temperature(predictions, temperature=0.7):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-8) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    return np.random.choice(range(len(predictions)), p=predictions)

# Function to generate text
def generate_text(input_seq, tokenizer, max_len, temperature=0.5):
    # Encode the input sequence to get the initial state
    states_value = encoder_model.predict(input_seq)

    # Prepare the target sequence (start with the "start" token or first word)
    target_seq = np.zeros((1, 1))  # Decoder input starts with a single token
    target_seq[0, 0] = tokenizer.word_index.get('start', 1)  # Use 'start' token or default token

    # Initialize the output sequence
    stop_condition = False
    generated_text = []
    
    while not stop_condition:
        # Predict the next word using decoder model
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample the next word using temperature-based sampling
        sampled_token_index = sample_with_temperature(output_tokens[0, -1, :], temperature)
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')

        # Add the word to the generated text
        generated_text.append(sampled_word)

        # Stop if 'end' token is predicted or sequence is too long
        if sampled_word == 'end' or len(generated_text) >= max_len:
            stop_condition = True

        # Update the target sequence (shift the word) and update states
        target_seq = np.zeros((1, 1))  # Prepare for the next word
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return ' '.join(generated_text)

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

    # Return the generated text as part of the response
    return render_template('result.html', character_name=character_name, backstory=random_backstory, generated_text=generated_text)

if __name__ == '__main__':
    app.run(debug=True)
