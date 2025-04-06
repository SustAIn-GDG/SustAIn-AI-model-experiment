
# Query Classification API - TFLite Powered Flask Service

This project provides a lightweight and production-ready Flask-based API service that classifies natural language queries using a TensorFlow Lite (TFLite) model. It is designed to be deployed efficiently in constrained environments, including edge devices, with support for cross-origin requests and easy integration into frontend or microservice architectures.

---

## 🚀 Features

- 🔗 **RESTful API** using Flask
- ⚡ **TensorFlow Lite Inference** for fast and efficient classification
- 🧠 **Pre-trained Model** with embedded text vectorization
- 🔍 **Label Prediction** with class probabilities
- 🔒 **CORS Enabled** for cross-domain access
- 📦 **Plug-and-play Deployment** with minimal dependencies

---

## 📁 Project Structure

```
.
├── model.ipynb               # Jupyter notebook for model training or analysis
├── vectorizer_config.pkl     # TextVectorization layer configuration
├── vectorizer_vocab.pkl      # Vocabulary used for vectorization
├── query_classifier_model.tflite  # Trained TFLite classification model
├── label_encoder.pkl         # Fitted LabelEncoder for class label mapping
└── app.py                    # Flask API script
```

---

## 🧠 Model Workflow

1. The text query is received via the `/predict` POST endpoint.
2. The query is vectorized using TensorFlow's `TextVectorization` layer (reconstructed from `vectorizer_config.pkl` and `vectorizer_vocab.pkl`).
3. The pre-processed input is passed to the TFLite model for inference.
4. Output logits are mapped back to labels using the loaded `LabelEncoder`.
5. Predictions (including all class scores and the final label) are returned as JSON.

---

## 📡 API Endpoints

### `GET /`
Health check endpoint to verify server status.

**Response:**
```
200 OK
Server is running!
```

---

### `POST /predict`
Predicts the class of a given query using the pre-trained TFLite model.

**Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "instances": [
    {"Query": "How do I fine-tune a BERT model?"},
    {"Query": "Generate a Python code for bubble sort"}
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "classes": ["classification", "text generation", "summarization"],
      "scores": [0.9, 0.05, 0.05],
      "predicted_label": "classification"
    },
    {
      "classes": ["classification", "text generation", "summarization"],
      "scores": [0.1, 0.85, 0.05],
      "predicted_label": "text generation"
    }
  ]
}
```

**Error Responses:**
- 400 Bad Request: Incorrect or missing input format.
- 500 Internal Server Error: Inference or server issue.

---

## ⚙️ Setup & Deployment

### 1. Install Dependencies
```bash
pip install flask flask-cors tensorflow numpy gevent
```

### 2. Run the Server
```bash
python app.py
```

Server runs at `http://0.0.0.0:8001` by default.

### 3. Environment Variable (Optional)
Set `PORT` to run the server on a different port:
```bash
export PORT=5000
```

---

## 🛠 Dependencies

- Flask
- Flask-CORS
- TensorFlow (Lite)
- NumPy
- Pickle
- gevent (used for patching concurrency)

---

## 🔒 Notes

- Ensure that `query_classifier_model.tflite`, `vectorizer_config.pkl`, `vectorizer_vocab.pkl`, and `label_encoder.pkl` are present in the root directory.
- The model and vectorizer should be trained and saved using a compatible TensorFlow/Keras version.

---

## 🧪 Testing

Use `curl` or Postman to test predictions:

```bash
curl -X POST http://localhost:8001/predict \
-H "Content-Type: application/json" \
-d '{"instances": [{"Query": "What is transfer learning?"}]}'
```

---

## 📄 License

© 2025 Team SustAIn. All rights reserved.

This project and its source code are the intellectual property of Team SustAIn.  
Unauthorized copying, distribution, modification, or usage in any form is strictly prohibited.
