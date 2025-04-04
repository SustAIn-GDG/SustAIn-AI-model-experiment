from gevent import monkey
monkey.patch_all()

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path="query_classifier_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load Vectorizer Config & Vocabulary
with open("vectorizer_config.pkl", "rb") as f:
    config = pickle.load(f)

vectorizer = tf.keras.layers.TextVectorization.from_config(config)

with open("vectorizer_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

vectorizer.set_vocabulary(vocab)  # Restore vocabulary

# Load Label Encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS preflight request successful"}), 200

    try:
        data = request.get_json()

        if not isinstance(data, dict) or "instances" not in data or not isinstance(data["instances"], list):
            return jsonify({"error": "Invalid input format. 'instances' should be a list."}), 400

        queries = [instance["Query"] for instance in data["instances"] if isinstance(instance, dict) and "Query" in instance]

        if not queries:
            return jsonify({"error": "No valid queries found in input."}), 400

        # Vectorize the queries
        vectorized_queries = vectorizer(tf.constant(queries)).numpy().astype(np.float32)

        # Run inference on each query
        response_data = []
        for vec_query in vectorized_queries:
            vec_query = np.expand_dims(vec_query, axis=0)  # Reshape for model input

            interpreter.set_tensor(input_details[0]['index'], vec_query)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            class_labels = label_encoder.classes_.tolist()
            scores = output_data[0].tolist()

            if not class_labels or not scores:
                predicted_class = "text generation"
            else:
                max_index = scores.index(max(scores))
                predicted_class = class_labels[max_index] if max_index < len(class_labels) else "text generation"

            response_data.append({"classes": class_labels, "scores": scores, "predicted_label": predicted_class})

        return jsonify({"predictions": response_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health_check():
    return "Server is running!", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    app.run(host="0.0.0.0", port=port)
