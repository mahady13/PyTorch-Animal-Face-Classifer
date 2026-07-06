# 🐾 PyTorch Animal Face Classifier

A deep learning-based web application that classifies animal faces into three distinct categories: **Cat**, **Dog**, and **Wild Animals**. Built using **PyTorch** for the core Convolutional Neural Network (CNN) architecture and **Streamlit** for a clean, interactive user interface.

---

## 🚀 Features
* **Real-time Image Classification:** Upload any animal face image (`.jpg`, `.jpeg`, `.png`) to get instant predictions.
* **Deep Learning Backend:** Powered by a custom 3-layer Convolutional Neural Network (CNN) trained in PyTorch.
* **Modern UI/UX:** A minimalist, mobile-friendly, and responsive sidebar-driven web layout.
* **Optimized Image Pipeline:** Automatic RGB conversion and resizing preprocessing pipelines.

---

## 🛠️ Tech Stack
* **Core Framework:** Python
* **Deep Learning:** PyTorch, Torchvision
* **Web Interface:** Streamlit
* **Data & Image Processing:** Pillow (PIL), NumPy, Pandas
* **Model Serialization:** Joblib

---

## 📁 Project Structure
```text
├── .venv/                   # Local Virtual Environment (Ignored by Git)
├── .idea/                   # PyCharm Settings (Ignored by Git)
├── model_architectures.py   # PyTorch Custom CNN Model Class definition
├── main.py                  # Streamlit Web Application Core Script
├── animal_classifier.pth    # Trained PyTorch Model Weights
├── requirements.txt         # Project Dependencies & Libraries
├── .gitignore               # Untracked Files Configuration
└── README.md                # Project Documentation

```
# 💻 Local Setup & Installation
Follow these steps to run the project locally on your machine:

* 1. Clone the Repository
```
git clone [https://github.com/mahady13/PyTorch-Animal-Face-Classifier.git](https://github.com/mahady13/PyTorch-Animal-Face-Classifier.git)
cd PyTorch-Animal-Face-Classifier
```
* 2. Set Up a Virtual Environment
```
# Create a virtual environment
python -m venv .venv

# Activate it (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate it (Mac/Linux or Git Bash)
source .venv/bin/activate
```
* 3. Install Dependencies
```
pip install -r requirements.txt
```
* 4. Run the Streamlit Application
```
streamlit run main.py
```
---
Once executed, the local server will spin up, and the app will automatically open in your default browser at http://localhost:8501.

# 🧠 Model Architecture Summary
The backend utilizes a sequential CNN design consisting of:

* **Conv2D Layer 1**: Extracts edge features.

* **MaxPooling & ReLU**: Reduces spatial dimensions and introduces non-linearity.

* **Conv2D Layer 2 & 3**: Learns complex facial textures and structural patterns.

* **Fully Connected (Linear) Layers**: Maps flattening features into final class probabilities (Cat, Dog, Wild).

# 🔗 Live Deployment
* This application is fully compatible and optimized for deployment on the Streamlit Community Cloud. Simply connect your GitHub repository, specify main.py as the entry point, and deploy instantly!
---
# 📄 License
* This project is open-source and available under the MIT License. Developed with 💻 by Mohiuddin Mahady.