# Rice-Leaf-Disease-Detection-using-CNN

# 📖 Overview

Rice is one of the most important staple foods worldwide, feeding more than a billion people every day. However, rice crops are highly vulnerable to leaf diseases such as Leaf Smut, Brown Spot, and Bacterial Leaf Blight, which reduce yield and affect food security.

Traditional disease detection methods are manual, time-consuming, and prone to human error. To overcome this, we developed a deep learning–based web application that can automatically detect rice plant diseases from leaf images.

This project uses a Convolutional Neural Network (CNN) trained on rice leaf datasets to classify images into three categories:

🌱 Leaf Smut
🍂 Brown Spot
🦠 Bacterial Leaf Blight

# 🎯 Features

✅ Multi-class classification using CNN
✅ Image pre-processing and augmentation for robust training
✅ Real-time disease detection using Flask Web App
✅ Upload image and get instant prediction with disease class
✅ User-friendly interface with visual feedback

# 🛠️ Tech Stack

- Programming Language: Python
- Deep Learning: TensorFlow / Keras (CNN Model)
- Image Processing: OpenCV
- Web Framework: Flask
- Frontend: HTML, CSS (custom template)
- IDE / Environment: Google Colab, VS Code, Jupyter Notebook

# 📂 Project Structure

```bash
Rice-Leaf-Disease-Detection-using-CNN/
│
├── dataset/                # Collected dataset of rice leaf diseases
├── static/                 # For storing uploaded images
├── templates/              # HTML files (frontend)
│   └── index.html
│
├── rice_leaf_cnn.ipynb     # Model training notebook (Colab)
├── model.h5                # Trained CNN model
├── app.py                  # Flask backend for prediction
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

```
## 🎥 Demo Video  

[![Watch the video](https://img.youtube.com/vi/-BfOz4UrpiM/0.jpg)](https://youtu.be/-BfOz4UrpiM)

# 📊 Results

    1. Achieved high accuracy (>94%) on validation and test datasets.
    2. The model successfully detects rice leaf diseases in real-time through the web app.

# 🚀 Future Improvements

    1. Extend to other crop diseases (wheat, maize, potato, etc.)
    2. Deploy on cloud (Heroku / AWS / GCP) for global accessibility
    3. Add mobile app integration for farmers