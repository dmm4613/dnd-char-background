from flask import Flask, render_template, request
from models import predict_cluster, generate_text
from database import get_random_text_by_cluster

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        character_name = request.form["character_name"]
        character_species = request.form["character_species"]
        character_class = request.form["character_class"]
        
        # Step 1: Use XGBoost to predict the cluster
        cluster_number = predict_cluster(character_name, character_species, character_class)
        
        # # Step 2: Query the database to get a random text from the predicted cluster
        # random_text = get_random_text_by_cluster(cluster_number)
        
        # # Step 3: Take the first 4 words and generate text using the NN model
        # first_four_words = " ".join(random_text.split()[:4])
        # generated_text = generate_text(first_four_words)
        
        return render_template("index.html", cluster_number=cluster_number)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)