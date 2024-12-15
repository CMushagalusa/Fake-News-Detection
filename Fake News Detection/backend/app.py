import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # Allow frontend requests
import joblib
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all domains for development

# Logging setup
logging.basicConfig(level=logging.INFO)

# Load Models and Vectorizer
try:
    MODELS = {
        "model1": joblib.load("logistic_model.pkl"),  # Logistic Regression
        "model2": joblib.load("decision_tree_model.pkl"),  # Decision Tree
        "model3": joblib.load("gradient_boosting_model.pkl"),  # Gradient Boosting
        "model4": joblib.load("random_forest_model.pkl"),  # Random Forest
    }
    vectorizer = joblib.load("vectorizer.pkl")  # Vectorizer for text preprocessing
except Exception as e:
    logging.error(f"Error loading models or vectorizer: {str(e)}")

# Default Route (Home Page)
@app.route("/")
def home():
    return """
    <h1>Welcome to the Fake News Classifier API</h1>
    <p>To classify news articles, use the <strong>/classify</strong> endpoint.</p>
    <p>Send a POST request with the JSON payload:</p>
    <pre>
    {
      "model": "model1",  # Options: model1, model2, model3, model4
      "news_content": "Enter your news content here."
    }
    </pre>
    """

# News Classification Route
@app.route("/classify", methods=["POST"])
def classify_news():
    try:
        # Check if content-type is JSON
        if not request.is_json:
            return jsonify({"error": "Content-Type must be 'application/json'."}), 415

        # Get the data from the request
        data = request.get_json()
        model_key = data.get("model")  # Selected model
        news_content = data.get("news_content", "").strip()  # Input news content

        # Validate inputs
        if not model_key or not news_content:
            return jsonify({"error": "Both 'model' and 'news_content' are required fields."}), 400

        # Ensure the selected model exists
        model = MODELS.get(model_key)
        if not model:
            return jsonify({"error": f"Invalid model '{model_key}'. Choose from: model1, model2, model3, model4."}), 400

        # Vectorize the input news content
        vectorized_content = vectorizer.transform([news_content])

        # Predict using the selected model
        prediction = model.predict(vectorized_content)
        result = "Not Fake News" if prediction[0] == 1 else "Fake News"

        logging.info(f"Prediction: Model {model_key}, Result: {result}")

        # Return the result
        return jsonify({
            "model": model_key,
            "result": result
        })

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Run the Flask App
if __name__ == "__main__":
    app.run(debug=True)
