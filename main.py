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

