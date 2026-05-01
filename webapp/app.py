from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import sys

# Fix for Windows console
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Disable warnings
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

app = Flask(__name__)

# Global variables
model = None
tokenizer = None

def load_model():
    """Load the PyTorch model"""
    global model, tokenizer
    
    try:
        print("Loading model...")
        
        # Check if we have a saved model
        if os.path.exists("model_full.pt"):
            print("Loading saved model from model_full.pt...")
            # Use weights_only=False since you trust this file
            model = torch.load("model_full.pt", map_location=torch.device('cpu'), weights_only=False)
            model.eval()
            print("[OK] Model loaded from file")
        elif os.path.exists("model_weights.pt"):
            print("Loading model architecture and weights...")
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            # Load state dict with weights_only=False
            state_dict = torch.load("model_weights.pt", map_location=torch.device('cpu'), weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()
            print("[OK] Model loaded from weights")
        else:
            print("Downloading model from Hugging Face...")
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.eval()
            print("[OK] Model downloaded and loaded")
        
        # Load tokenizer
        if os.path.exists("tokenizer"):
            print("Loading saved tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("tokenizer")
            print("[OK] Tokenizer loaded from file")
        else:
            print("Loading tokenizer from Hugging Face...")
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("[OK] Tokenizer downloaded and loaded")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route("/predict", methods=["POST"])
def predict():
    """Predict sentiment of input text"""
    try:
        data = request.get_json()
        
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data.get("text", "")
        
        if not text:
            return jsonify({"error": "Text field is empty"}), 400
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            probabilities = torch.softmax(logits, dim=1).numpy()[0]
        
        # Map prediction to sentiment (0=negative, 1=neutral, 2=positive)
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = sentiment_map[prediction]
        
        result = {
            "sentiment": sentiment,
            "positive": prediction == 2,
            "confidence": float(probabilities[prediction]),
            "probabilities": {
                "negative": float(probabilities[0]),
                "neutral": float(probabilities[1]),
                "positive": float(probabilities[2])
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    })

@app.route("/", methods=["GET"])
def index():
    """API information"""
    return jsonify({
        "service": "Sentiment Analysis API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Send JSON with 'text' field",
            "/health": "GET - Check service status"
        },
        "example": {
            "curl": 'curl -X POST -H "Content-Type: application/json" -d \'{"text": "I love this!"}\' http://localhost:5000/predict'
        }
    })

if __name__ == "__main__":
    print("=" * 50)
    print("Sentiment Analysis Flask App")
    print("=" * 50)
    
    if load_model():
        print("\n[OK] Application ready!")
        print("[OK] Server: http://0.0.0.0:5000")
        print("[OK] Health check: http://localhost:5000/health")
        print("\nPress CTRL+C to stop")
        print("=" * 50)
        
        app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        print("\n[ERROR] Failed to start application")