import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import random

import nltk
nltk.download('averaged_perceptron_tagger_eng')

# Load the trained encoder and decoder models
encoder_model = load_model('encoder_model.h5')
decoder_model = load_model('decoder_model.h5')

# Function to sample with temperature
def sample_with_temperature(predictions, temperature=0.2):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-8) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    return np.random.choice(range(len(predictions)), p=predictions)

# Function to generate text
def generate_text(input_seq, tokenizer, max_len, temperature=0.2):
    # Encode the input sequence to get the initial state
    states_value = encoder_model.predict(input_seq)

    # Prepare the target sequence using the seed sequence (first four words)
    target_seq = input_seq[:, :1]  # Start with the first word of the input sequence

    # Initialize the output sequence
    stop_condition = False
    generated_text = []

    # If you want to preserve the first four words in the generated text
    # Add them directly to the generated_text list
    for word_id in input_seq[0]:
        if word_id != 0:  # Avoid padding tokens
            generated_text.append(tokenizer.index_word.get(word_id, ''))

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


def clean_generated_text(text):
    # Remove non-alphabetical characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Split text into words and filter out any short or strange words (optional)
    words = text.split()
    common_words = [word for word in words if len(word) > 2]  # Filter out very short words
    
    # Join cleaned words back into a sentence
    return ' '.join(common_words)


# Function to intelligently add filler words using POS tagging
def add_filler_words_smart(generated_text, insertion_prob=0.45):

    # List of filler words categorized by parts of speech
    filler_words_by_pos = {
        'NN': ['the', 'a', 'some', 'this', 'that', 'any', 'one', 'each'],  # Articles and determiners for nouns
        'VB': ['is', 'was', 'has', 'will', 'might', 'can', 'could', 'would', 'should'],  # Auxiliary verbs
        'JJ': ['very', 'quite', 'rather', 'extremely', 'fairly', 'pretty', 'really', 'somewhat'],  # Intensifiers for adjectives
        'IN': ['with', 'by', 'on', 'from', 'under', 'over', 'through', 'in', 'around', 'at'],  # Prepositions
        'RB': ['probably', 'definitely', 'clearly', 'certainly', 'obviously', 'possibly', 'evidently'],  # Adverbs for verbs/adjectives
        'CC': ['and', 'or', 'but', 'so', 'yet'],  # Conjunctions
        'PRP': ['he', 'she', 'it', 'they', 'we', 'you', 'I'],  # Pronouns for general context
        'MD': ['will', 'would', 'can', 'could', 'might', 'must', 'should', 'shall'],  # Modal verbs
    }

    words = nltk.word_tokenize(generated_text)  # Tokenize the text
    pos_tags = nltk.pos_tag(words)  # Get POS tags for the generated text
    new_text = []
    
    for word, pos in pos_tags:
        new_text.append(word)
        
        # If the word's POS is in our filler_words_by_pos dictionary, insert a filler word
        if pos in filler_words_by_pos and random.random() < insertion_prob:
            filler_word = random.choice(filler_words_by_pos[pos])
            new_text.append(filler_word)
    
    return ' '.join(new_text)

# Function to generate structured sentences using POS tagging on enhanced text
def generate_structured_text(enhanced_text, max_sent_len=10):
    words = nltk.word_tokenize(enhanced_text)  # Tokenize the text into words
    pos_tags = nltk.pos_tag(words)  # Get POS tags for the words
    
    generated_sentences = []
    current_sentence = []

    # Iterate through the POS tagged words and structure sentences
    for word, pos in pos_tags:
        current_sentence.append(word)
        
        # Define basic sentence-ending logic based on POS
        # End the sentence if a noun, verb, or adjective has been seen and sentence is long enough
        if pos in ('NN', 'VB') and len(current_sentence) >= max_sent_len:
            generated_sentences.append(' '.join(current_sentence))
            current_sentence = []
        elif pos == 'CC':  # If a conjunction (e.g., "and", "but"), add a comma
            current_sentence.append(',')
            # Do not end the sentence, continue accumulating more words
    
    # Add any remaining words as the last sentence
    if current_sentence:
        generated_sentences.append(' '.join(current_sentence))

    # Join the sentences with punctuation
    punctuated_text = '. '.join(generated_sentences) + '.'
    
    return punctuated_text