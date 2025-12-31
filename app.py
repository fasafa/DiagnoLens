# app.py
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
import torch
import os

app = Flask(__name__)

# ---------------------------
# 1. Load Base + LoRA Adapter
# ---------------------------
base_model_name = "unsloth/Llama-3.2-11B-Vision"  # or your original base model name
adapter_path = "LoRa_model"  # your fine-tuned adapter folder

device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔹 Loading base model...")
base_model = AutoModelForVision2Seq.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

print("🔹 Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.to(device)
model.eval()

# Load processor
processor = AutoProcessor.from_pretrained(adapter_path)

# ---------------------------
# 2. API Endpoint
# ---------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        image = Image.open(file.stream).convert("RGB")

        # Preprocess input
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Generate prediction
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100)

        description = processor.decode(output[0], skip_special_tokens=True)

        return jsonify({"diagnosis": description})

    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------------------
# 3. Run Flask
# ---------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
