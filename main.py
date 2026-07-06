import streamlit as st
import torch
import numpy as np
import joblib
import pandas as pd
from model_architectures import MyModel
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_asset():
    model=MyModel()
    model.load_state_dict(torch.load("animal_classifier.pth",map_location=device))
    model.to(device)
    model.eval()

    label_encoder=joblib.load("label_encoder.pkl")
    return model,label_encoder

model,label_encoder=load_asset()

transform=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32)
])

st.title("PyTorch Animal Face Classifier")
st.text("This app uses Deep Learning (PyTorch) to classify if the given picture is a Cat, Dog, or Wild animal.")
st.sidebar.title("Upload an Image to Predict Cat/Dog/Wild")
image=st.sidebar.file_uploader("Upload an image")

if image is not None and st.sidebar.button("Predict"):
    img=Image.open(image).convert("RGB")
    image=transform(img)
    image=image.unsqueeze(0)
    image=image.to(device)
    st.image(img, caption="Uploaded Image", width=300)

    with torch.no_grad():
        predict=model(image)
        name=torch.argmax(predict,axis=1).item()
    label=label_encoder.inverse_transform([name])
    st.success(f"Predicted Animal: {label[0]}")
else:
    st.error("Please Upload an Image & Then Click Predict")
