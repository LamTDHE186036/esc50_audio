import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import librosa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

au_csv = pd.read_csv("esc50.csv")
idx2label = dict(zip(au_csv['target'], au_csv['category']))

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(idx2label))
model.load_state_dict(torch.load("esc50_resnet18.pth", map_location=device))
model.eval()
model = model.to(device)

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def extract_mel_rgb(path, n_mels=128):
    y, sr = librosa.load(path, sr=44100)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min()) * 255
    mel_db = np.uint8(mel_db)
    mel_db = np.stack([mel_db]*3, axis=-1)
    return mel_db

def predict_audio(path, model):
    mel_img = extract_mel_rgb(path)
    mel_img = transforms.ToPILImage()(mel_img)
    mel_img = tfms(mel_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(mel_img)
        pred_class = logits.argmax(1).item()
    return pred_class

def predict_gradio(file):
    try:
        pred_class = predict_audio(file, model)
        label_name = idx2label[pred_class]
        return f"The above sound belongs to the class: {pred_class} ({label_name})"
    except Exception as e:
        return f"Error: {e}"

demo = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="ESC-50 Audio Classifier",
    description="Upload a .wav file to predict its class."
)

demo.launch()
