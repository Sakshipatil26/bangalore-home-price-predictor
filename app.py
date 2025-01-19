import pickle
import json
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


model = pickle.load(open("banglore_home_prices_model.pickle", "rb"))
with open("columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]
    locations = data_columns[3:]  

@app.route("/")
def home():
    return render_template("home.html", locations=locations)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        location = request.form.get("location")
        total_sqft = float(request.form.get("total_sqft"))
        bath = float(request.form.get("bath"))
        bhk = int(request.form.get("bhk"))
        
        # Prepare input array for model
        loc_index = data_columns.index(location.lower()) if location.lower() in data_columns else -1
        x = [0] * len(data_columns)
        x[0] = total_sqft
        x[1] = bath
        x[2] = bhk
        if loc_index >= 0:
            x[loc_index] = 1
        
        # Predict price
        price = model.predict([x])[0]
        return render_template("result.html", price=round(price, 2))
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
