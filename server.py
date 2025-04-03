from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import threading

app = Flask(__name__)
CORS(app)

reinforcement_data = []

# Load Model
model = tf.keras.models.load_model("query_classifier_model.keras")

# Load Vectorizer Config & Vocabulary
with open("vectorizer_config.pkl", "rb") as f:
    config = pickle.load(f)

vectorizer = tf.keras.layers.TextVectorization.from_config(config)

with open("vectorizer_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

vectorizer.set_vocabulary(vocab)  # Properly restore vocabulary

# Load Label Encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS preflight request successful"}), 200

    try:
        data = request.get_json()
        
        # Debugging Statement
        print("Received Data:", data)  # Check structure in logs

        # Check if Data is a Dictionary
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid input format. Expected JSON object."}), 400

        if "instances" not in data or not isinstance(data["instances"], list):
            return jsonify({"error": "Invalid input format. 'instances' should be a list."}), 400

        # Extract Queries
        queries = []
        for instance in data["instances"]:
            if isinstance(instance, dict) and "Query" in instance:
                queries.append(instance["Query"])

        if not queries:
            return jsonify({"error": "No valid queries found in input."}), 400

        vectorized_queries = vectorizer(tf.constant(queries))
        predictions = model.predict(vectorized_queries)
        response_data = []
        for i, scores in enumerate(predictions):
            class_labels = label_encoder.classes_.tolist()
            scores = scores.tolist()  # Convert NumPy array to list

            if not class_labels or not scores:
                predicted_class = "text generation"  # Default fallback
            else:
                max_index = scores.index(max(scores))  # Index of max score
                predicted_class = class_labels[max_index] if max_index < len(class_labels) else "text generation"

            response_data.append({"classes": class_labels, "scores": scores, "predicted_label": predicted_class})

        return jsonify({"predictions": response_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reinforce", methods=["POST"])
def reinforce():
    global reinforcement_data

    try:
        data = request.get_json()
        query = data.get("query")
        correct_label = data.get("correct_label")

        if not query or not correct_label:
            return jsonify({"error": "Both 'query' and 'correct_label' are required."}), 400

        # Store new data
        reinforcement_data.append((query, correct_label))

        # Retrain the model after every 10 corrections
        if len(reinforcement_data) >= 10:
            training_thread = threading.Thread(target=retrain_model)
            training_thread.start()
            reinforcement_data = []
            return jsonify({"message": "Feedback stored. Reinforcement training started in the background."})

        return jsonify({"message": "Feedback stored. Model will be updated soon."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def retrain_model():
    global model, vectorizer, label_encoder

    df = pd.read_csv("final_dataset (1).csv")
    X = df["Query"].astype(str).tolist()
    y = df["Label"].tolist()

    # Add Reinforcement Data
    for query, label in reinforcement_data:
        X.append(query)
        y.append(label)

    # Re-encode Labels
    label_encoder.fit(y)
    y_encoded = label_encoder.transform(y)

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    # Re-Vectorize Data
    vectorizer.adapt(X)
    X_vec = vectorizer(np.array(X))

    # Re-train Model (Fine-tune last layers)
    model.fit(X_vec, y_encoded, epochs=1, batch_size=32)

    # Save Updated Model
    model.save("query_classifier_model.keras")

    return "Model retrained successfully."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
