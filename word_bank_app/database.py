import random

# Assuming you have a dataset of character backstories in a pandas DataFrame or SQL database

def get_random_text_by_cluster(cluster_number):
    # Example if you're using pandas:
    import pandas as pd
    df = pd.read_csv("character_backstories.csv")  # Replace with your dataset
    
    # Filter by cluster
    filtered_df = df[df['cluster'] == cluster_number]
    
    # Select a random text from the filtered DataFrame
    random_text = random.choice(filtered_df['character_backstory'].tolist())
    return random_text