import streamlit as st
import torch
import numpy as np
import joblib
import pandas as pd
from model_architectures import MyModel
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder