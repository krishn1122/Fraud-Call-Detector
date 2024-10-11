import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import librosa
import numpy as np

# Load the model and processor
model_name = (
    r"E:\fraude call\wav2vec2-base"  # Replace with your Hugging Face model path
)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)


# Load and process the .wav file
def load_wav_file(file_path):
    speech, sample_rate = librosa.load(file_path, sr=16000)  # Load audio at 16kHz
    return speech, sample_rate


# Predict whether the call is fraudulent or not
def predict(file_path):
    speech, sample_rate = load_wav_file(file_path)
    inputs = processor(
        speech, sampling_rate=sample_rate, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    return predicted_class_id


# Test with your .wav file
file_path = r"E:\fraude call\audio.wav"  # Replace with your .wav file path
predicted_label = predict(file_path)

# Assuming the model has two classes: 0 - Not Fraud, 1 - Fraud
if predicted_label == 0:
    print("The call is not fraudulent.")
else:
    print("The call is fraudulent.")
